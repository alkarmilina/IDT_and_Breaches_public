import pandas as pd
import numpy as np
import os
from thefuzz import fuzz, process
import statsmodels.api as sm

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part3/PRC_Maine_NH_regression.py

"""
Part 3: State-Level Regression and Undisclosed Record Imputation

Recovers national record counts for breaches that only have a
state-level count, by regressing known national counts on their
matching state-level counts (Maine and New Hampshire Attorney General
filings), then applying the resulting multiplier to breaches that don't
have a national match. Finally imputes a value for PRC incidents that
still have an undisclosed record count, using the annual nu baseline.

Setup:
Reads the PRC-HHS integrated dataset from PRC_fill_in_health.py, plus
the raw Maine and New Hampshire breach notification filings. State
filings are first narrowed down to national-scale organizations only,
Maine via a keyword blocklist (excluding schools, local clinics, small
banks, and similar purely local entities), New Hampshire via an
allowlist of specific known national companies, since its dataset is
too small for keyword filtering to work well. Matches to the national
dataset require a fuzzy name match score of at least 70, and the
national count must be at least 5 times the state count, following the
paper's high-confidence match criteria.

Goal:
Produce the paper's state-level regression recovery and final
undisclosed-record imputation.

Outputs:
- data/processed/part3/state_augmented_breaches.csv: state-only
  breaches with their estimated national count.
- data/processed/part3/PRC_augmented.csv: the full augmented PRC
  dataset, national records plus recovered state records, with
  undisclosed counts imputed.
"""


def clean_name(name):
    """Lowercases an organization name, strips punctuation, and strips common business suffixes, to improve fuzzy match quality."""
    if pd.isna(name) or str(name).strip() == "":
        return "unknown_org"
    cleaned = str(name).lower().strip().replace(".", "").replace(",", "")
    suffixes = [" inc", " corp", " llc", " ltd", " corporation", " lp", " group"]
    temp_cleaned = cleaned
    for s in suffixes:
        if len(temp_cleaned.replace(s, "").strip()) > 2:
            temp_cleaned = temp_cleaned.replace(s, "").strip()
    return temp_cleaned


def classify_nh_entity(org_name):
    """Flags New Hampshire filings that belong to a known national-scale company, via a manually curated allowlist."""
    org_lower = str(org_name).lower()
    national_allowlist = [
        "hewlett packard", "pepsi", "toyota", "orbitz", "merrill lynch", "aetna",
        "warby parker", "oxo", "ameriprise", "quest diagnostics", "regal entertainment",
        "alere", "benchmade", "wilton brands", "topps", "djo", "sourcefire",
        "lokai", "la jolla", "baylor", "six red marbles", "workers united",
        "gloria jean", "dutch bros"
    ]
    for n in national_allowlist:
        if n in org_lower: return True
    return False


def classify_maine_entity(org_name):
    """Flags Maine filings as national-scale by excluding organizations with keywords indicating a purely local scope."""
    org_lower = str(org_name).lower()
    local_keywords = [
        "school", "elementary", "high school", "district", "academy", "college", "university",
        "town of", "city of", "village of", "county", "police", "sheriff", "municipality",
        "dental", "dentist", "orthodontic", "family practice", "pediatrics", "clinic",
        "eye care", "optometry", "chiropractic", "veterinary", "animal hospital",
        "surgery center", "medical center", "hospital",
        "credit union", "fcu", "savings bank", "community bank",
        "realty", "real estate", "properties", "condo", "apartments",
        "auto", "motors", "dealership", "nissan of", "toyota of", "ford of",
        "restaurant", "grill", "pizza", "cafe", "bakery", "brewing",
        "plumbing", "heating", "electric", "landscaping", "construction"
    ]
    for keyword in local_keywords:
        if keyword in org_lower: return False
    return True


def load_state_data(path, state_label):
    """Loads one state's breach filings from CSV or Excel, standardizing whichever organization/affected-count/date columns are present."""
    if not os.path.exists(path):
        print(f"Warning: Missing state file {path}")
        return pd.DataFrame()

    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)

    col_map = {
        'org': ['Organization', 'org_name', 'Entity Name', 'Company', 'DATA BREACH NOTICES', '02_03_01_Entity Name', 'Company Whose Data Was Breached'],
        'affected': ['NH_Affected', 'maine_affected', 'Maine residents affected', 'Count', 'Unnamed: 6', 'Number of Maine Residents Affected', '03_01_02_Total number of Maine residents affected'],
        'date': ['Date of Notification', 'Notification Date', 'Date_Reported', 'Date', 'Year', 'year', 'Reported Year', 'Completed Date']
    }

    found_org = [c for c in col_map['org'] if c in df.columns]
    found_affected = [c for c in col_map['affected'] if c in df.columns]
    found_date = [c for c in col_map['date'] if c in df.columns]

    if found_org and found_affected:
        cols_to_keep = [found_org[0], found_affected[0]]
        if found_date: cols_to_keep.append(found_date[0])

        df = df[cols_to_keep].copy()

        if len(df.columns) == 3:
            df.columns = ['Organization', 'State_Affected_X', 'Date_Raw']
        else:
            df.columns = ['Organization', 'State_Affected_X']
            df['Date_Raw'] = np.nan

        df['State'] = state_label
        df['State_Affected_X'] = pd.to_numeric(df['State_Affected_X'], errors='coerce').fillna(0)

        df['Clean_Date'] = pd.to_datetime(df['Date_Raw'], errors='coerce')
        df['Year'] = df['Clean_Date'].dt.year

        # Some sources report a plain year instead of a full date.
        mask_nat = df['Clean_Date'].isna()
        if mask_nat.any():
            df.loc[mask_nat, 'Year'] = pd.to_numeric(df.loc[mask_nat, 'Date_Raw'], errors='coerce')

        return df[df['State_Affected_X'] > 0]

    print(f"  [{state_label}] Warning: Columns not found in {os.path.basename(path)}")
    return pd.DataFrame()


def main():
    print("\n--- Part 3: State-Level Regression and Undisclosed Record Imputation ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    enriched_base_path = os.path.join(root_dir, 'data/processed/part3/prc_hhs_integrated.csv')
    nh_path = os.path.join(root_dir, 'data/raw/PRC_supplements/NH_Manual_Entry_Template.csv')
    me_path_1 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine1.xlsx')
    me_path_2 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine2.xlsx')

    output_recovered = os.path.join(root_dir, "data/processed/part3/state_augmented_breaches.csv")
    output_final_augmented = os.path.join(root_dir, "data/processed/part3/PRC_augmented.csv")

    print("Loading state-level data...")
    nh_df = load_state_data(nh_path, 'NH')
    nh_df['is_national_scale'] = nh_df['Organization'].apply(classify_nh_entity)
    nh_df_filtered = nh_df[nh_df['is_national_scale'] == True].copy()
    print(f"NH data: filtered {len(nh_df)} down to {len(nh_df_filtered)} records.")

    me_combined = pd.concat([load_state_data(me_path_1, 'ME'), load_state_data(me_path_2, 'ME')], ignore_index=True)
    me_combined['is_national_scale'] = me_combined['Organization'].apply(classify_maine_entity)
    me_filtered = me_combined[me_combined['is_national_scale'] == True].copy()
    print(f"Maine data: filtered {len(me_combined)} down to {len(me_filtered)} records.")

    state_df = pd.concat([nh_df_filtered, me_filtered], ignore_index=True)
    state_df = state_df.sort_values('State_Affected_X', ascending=False).drop_duplicates('Organization')
    state_df['clean_org'] = state_df['Organization'].apply(clean_name)

    national_df = pd.read_csv(enriched_base_path)
    print(f"Loaded data from {enriched_base_path}")
    national_df['clean_org'] = national_df['org_name'].apply(clean_name)

    print(f"Matching {len(state_df)} unique state incidents to national base...")

    mega_threshold = 10000000
    national_df_filtered = national_df[national_df['total_affected'] < mega_threshold].copy()

    matches = []
    for idx, s_row in state_df.iterrows():
        match = process.extractOne(s_row['clean_org'], national_df_filtered['clean_org'], scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 70:
            national_val = national_df_filtered[national_df_filtered['clean_org'] == match[0]]['total_affected'].iloc[0]
            if national_val >= s_row['State_Affected_X']:
                matches.append({
                    'Organization': s_row['Organization'],
                    'X_State': s_row['State_Affected_X'],
                    'Y_National': national_val,
                    'original_index': idx
                })

    reg_df = pd.DataFrame(matches)

    if len(reg_df) <= 10:
        print("Error: Insufficient matched pairs for regression.")
        return

    # High-confidence pairs only: national count at least 5x the state count.
    train_df = reg_df[reg_df['Y_National'] >= (5 * reg_df['X_State'])].copy()

    X = train_df['X_State']
    y = train_df['Y_National']
    model = sm.OLS(y, X).fit()
    beta = model.params.iloc[0]

    print("\nRegression results:")
    print(f"  Matches:            {len(reg_df)}")
    print(f"  Training set:       {len(train_df)}")
    print(f"  Beta (multiplier):  {beta:.2f}")
    print(f"  R-squared:          {model.rsquared:.4f}")

    # Apply beta to recover a national estimate for state incidents with no confident national match.
    matched_indices = set(reg_df['original_index'])
    all_indices = set(state_df.index)
    unmatched_indices = list(all_indices - matched_indices)

    unmatched_df = state_df.loc[unmatched_indices].copy()
    unmatched_df['Estimated_National_Count'] = unmatched_df['State_Affected_X'] * beta

    median_year = unmatched_df['Year'].median()
    if pd.isna(median_year): median_year = 2015
    unmatched_df['Year'] = unmatched_df['Year'].fillna(median_year)

    recovered_output = unmatched_df[['Organization', 'State', 'Year', 'Clean_Date', 'State_Affected_X', 'Estimated_National_Count']]
    recovered_output.to_csv(output_recovered, index=False)

    print(f"Applied beta ({beta:.2f}) to {len(unmatched_df)} unmatched national-scale incidents.")
    print(f"Data saved to {output_recovered}")

    # Merge the recovered state incidents into the national dataset.
    national_final = national_df.copy()
    if 'clean_org' in national_final.columns: del national_final['clean_org']
    national_final['source_type'] = 'PRC_HHS'

    state_final = recovered_output.copy()
    state_final = state_final.rename(columns={'Organization': 'org_name', 'Estimated_National_Count': 'total_affected'})

    # Only keep recovered records with a specific reported date.
    state_final['reported_date'] = state_final['Clean_Date']
    before_drop = len(state_final)
    state_final = state_final.dropna(subset=['reported_date'])
    dropped_count = before_drop - len(state_final)

    if dropped_count > 0:
        print(f"Dropped {dropped_count} state records missing specific dates.")

    if 'Clean_Date' in state_final.columns: del state_final['Clean_Date']
    state_final['source_type'] = 'State_Augmented'

    for col in national_final.columns:
        if col not in state_final.columns: state_final[col] = np.nan
    state_final = state_final[national_final.columns]

    prc_augmented = pd.concat([national_final, state_final], ignore_index=True)

    # Final imputation of PRC incidents that still have an undisclosed record count, using
    # the annual nu baseline derived in find_nu.py, split evenly among that year's undisclosed
    # incidents.
    print("Imputing undisclosed counts...")

    annual_nu_prc_only = 1531784  # From find_nu.py. Re-run find_nu.py and update this if the input data changes.

    prc_augmented['Year'] = prc_augmented['Year'].fillna(pd.to_datetime(prc_augmented['reported_date']).dt.year)
    unknown_counts = prc_augmented[prc_augmented['total_affected'] == 0].groupby('Year').size().to_dict()

    print("\n" + "-"*40)
    print(f"{'Year':<10} | {'Undisclosed incidents that year':<20}")
    print("-" * 40)
    for year in sorted(unknown_counts.keys()):
        print(f"{int(year):<10} | {unknown_counts[year]:<20}")
    print("-" * 40 + "\n")

    def impute_final_values(row):
        if row['total_affected'] > 0: return row['total_affected']
        year_count = unknown_counts.get(row['Year'], 1)
        return annual_nu_prc_only / year_count if year_count > 0 else annual_nu_prc_only

    prc_augmented['total_affected_imputed'] = prc_augmented.apply(impute_final_values, axis=1)

    prc_augmented.to_csv(output_final_augmented, index=False)
    print(f"Data saved to {output_final_augmented}")

    total_imputed_vol = prc_augmented['total_affected_imputed'].sum()
    print(f"Original count: {len(national_final)} | Recovered added: {len(state_final)} | Final total: {len(prc_augmented)}")
    print(f"Total volume (incl. imputed): {total_imputed_vol:,.0f}")

    unmatched_sum_X = unmatched_df['State_Affected_X'].sum()
    vol_recovered_absolute = unmatched_sum_X * beta
    vol_per_year = vol_recovered_absolute / 14  # 14-year study period (2008-2021).

    print("\nRecovered volume:")
    print(f"  State victims (unmatched): {unmatched_sum_X:,.0f}")
    print(f"  Beta:                      {beta:.2f}")
    print(f"  Volume added:              {vol_recovered_absolute:,.0f}")
    print(f"  Per year:                  {vol_per_year:,.0f}")


if __name__ == "__main__":
    main()
