import pandas as pd
import numpy as np
import os
from thefuzz import fuzz, process
import statsmodels.api as sm

# To run: python scripts/part3/PRC_Maine_NH_Regression.py

def clean_name(name):
    """Safely cleans organization names for matching."""
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
    """Manually classifies NH entities (Allowlist)."""
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
    """Classifies Maine entities (Blocklist)."""
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

def main():
    root_dir = os.getcwd()
    enriched_base_path = os.path.join(root_dir, 'data/processed/part3/prc_hhs_integrated.csv')
    nh_path = os.path.join(root_dir, 'data/raw/PRC_supplements/NH_Manual_Entry_Template.csv')
    me_path_1 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine1.xlsx')
    me_path_2 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine2.xlsx')
    
    output_recovered = os.path.join(root_dir, "data/processed/part3/state_augmented_breaches.csv")
    output_final_augmented = os.path.join(root_dir, "data/processed/part3/PRC_augmented.csv")

    def load_state_data(path, state_label):
        if not os.path.exists(path):
            print(f"Warning: Missing state file {path}")
            return pd.DataFrame()
        
        # Load file (Headers are now reliably in Row 0)
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        
        # Robust Column Mapping
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
            
            # Rename for standardization
            if len(df.columns) == 3:
                df.columns = ['Organization', 'State_Affected_X', 'Date_Raw']
            else:
                df.columns = ['Organization', 'State_Affected_X']
                df['Date_Raw'] = np.nan

            df['State'] = state_label
            df['State_Affected_X'] = pd.to_numeric(df['State_Affected_X'], errors='coerce').fillna(0)
            
            # --- DATE PARSING ---
            df['Clean_Date'] = pd.to_datetime(df['Date_Raw'], errors='coerce')
            df['Year'] = df['Clean_Date'].dt.year
            
            # Fallback for plain years
            mask_nat = df['Clean_Date'].isna()
            if mask_nat.any():
                df.loc[mask_nat, 'Year'] = pd.to_numeric(df.loc[mask_nat, 'Date_Raw'], errors='coerce')
            
            return df[df['State_Affected_X'] > 0]
            
        print(f"  [{state_label}] Warning: Columns not found in {os.path.basename(path)}")
        return pd.DataFrame()

    print("Loading state-level data...")
    nh_df = load_state_data(nh_path, 'NH')
    nh_df['is_national_scale'] = nh_df['Organization'].apply(classify_nh_entity)
    nh_df_filtered = nh_df[nh_df['is_national_scale'] == True].copy()
    print(f"NH Data: Filtered {len(nh_df)} down to {len(nh_df_filtered)} records.")

    me_combined = pd.concat([load_state_data(me_path_1, 'ME'), load_state_data(me_path_2, 'ME')], ignore_index=True)
    me_combined['is_national_scale'] = me_combined['Organization'].apply(classify_maine_entity)
    me_filtered = me_combined[me_combined['is_national_scale'] == True].copy()
    print(f"Maine Data: Filtered {len(me_combined)} down to {len(me_filtered)} records.")

    state_df = pd.concat([nh_df_filtered, me_filtered], ignore_index=True)
    state_df = state_df.sort_values('State_Affected_X', ascending=False).drop_duplicates('Organization')
    state_df['clean_org'] = state_df['Organization'].apply(clean_name)

    print(f"Matching {len(state_df)} unique state incidents to national base...")
    national_df = pd.read_csv(enriched_base_path)
    national_df['clean_org'] = national_df['org_name'].apply(clean_name)
    
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

    if len(reg_df) > 10:
        # --- [MAIN LOGIC] 5x FILTER ---
        train_df = reg_df[reg_df['Y_National'] > (5 * reg_df['X_State'])].copy()
        
        X = train_df['X_State']
        y = train_df['Y_National']
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[0]
        
        print("\n" + "="*45)
        print(f"--- MULTIPLIER REGRESSION RESULTS ---")
        print(f"Total Matches:              {len(reg_df)}")
        print(f"Training Set (>5x Ratio):   {len(train_df)}")
        print(f"National Multiplier (Beta): {beta:.2f}")
        print(f"R-Squared:                  {model.rsquared:.4f}")
        print("="*45)
        
        # --- RECOVER THE MISSING DATA ---
        matched_indices = set(reg_df['original_index'])
        all_indices = set(state_df.index)
        unmatched_indices = list(all_indices - matched_indices)
        
        unmatched_df = state_df.loc[unmatched_indices].copy()
        unmatched_df['Estimated_National_Count'] = unmatched_df['State_Affected_X'] * beta
        
        # Fallback Year
        median_year = unmatched_df['Year'].median()
        if pd.isna(median_year): median_year = 2015
        unmatched_df['Year'] = unmatched_df['Year'].fillna(median_year)
        
        recovered_output = unmatched_df[['Organization', 'State', 'Year', 'Clean_Date', 'State_Affected_X', 'Estimated_National_Count']]
        recovered_output.to_csv(output_recovered, index=False)
        
        print(f"\n[STEP 1] Augmented Data Generation Complete.")
        print(f"Applied Beta ({beta:.2f}) to {len(unmatched_df)} unmatched national-scale incidents.")
        
        # --- MERGE INTO FINAL DATASET ---
        print(f"\n[STEP 2] Merging National and Recovered State Data...")
        
        national_final = national_df.copy()
        if 'clean_org' in national_final.columns: del national_final['clean_org']
        national_final['source_type'] = 'PRC_HHS'

        state_final = recovered_output.copy()
        state_final = state_final.rename(columns={'Organization': 'org_name', 'Estimated_National_Count': 'total_affected'})
        
        # Strict Date Logic
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

        # --- [STEP 3] FINAL IMPUTATION OF UNKNOWNS (n_u) ---
        print(f"\n[STEP 3] Imputing 'Pure Unknown' records with Annual Baseline...")
        
        ANNUAL_NU_PRC_ONLY = 1531784
        
        prc_augmented['Year'] = prc_augmented['Year'].fillna(pd.to_datetime(prc_augmented['reported_date']).dt.year)
        unknown_counts = prc_augmented[prc_augmented['total_affected'] == 0].groupby('Year').size().to_dict()
        # --- NEW PRINT SECTION FOR I_u,y ---
        print("\n" + "-"*30)
        print(f"{'Year (y)':<10} | {'Undisclosed Count (|I_u,y|)':<20}")
        print("-" * 30)
        for year in sorted(unknown_counts.keys()):
            print(f"{int(year):<10} | {unknown_counts[year]:<20}")
        print("-" * 30 + "\n")
        # ----------------------------------
        
        def impute_final_values(row):
            if row['total_affected'] > 0: return row['total_affected']
            year_count = unknown_counts.get(row['Year'], 1)
            return ANNUAL_NU_PRC_ONLY / year_count if year_count > 0 else ANNUAL_NU_PRC_ONLY

        prc_augmented['total_affected_imputed'] = prc_augmented.apply(impute_final_values, axis=1)

        prc_augmented.to_csv(output_final_augmented, index=False)
        
        total_imputed_vol = prc_augmented['total_affected_imputed'].sum()
        print(f"Successfully created: {output_final_augmented}")
        print(f"Original Count: {len(national_final)} | Recovered Added: {len(state_final)} | Final Total: {len(prc_augmented)}")
        print(f"Total Volume (inc. Imputed): {total_imputed_vol:,.0f}")
        
        # --- CALCULATE ABSOLUTE RECOVERED VOLUME ---
        unmatched_sum_X = unmatched_df['State_Affected_X'].sum()
        vol_recovered_absolute = unmatched_sum_X * beta
        
        # Approx 15 years in dataset
        vol_per_year = vol_recovered_absolute / 14
        
        print("\n" + "="*65)
        print("--- VERIFICATION: RECOVERED VOLUME (ABSOLUTE) ---")
        print(f"Total State Victims (Unmatched):    {unmatched_sum_X:,.0f}")
        print(f"Multiplier (Beta):                  {beta:.2f}")
        print(f"Total Volume Added (Recovered):     {vol_recovered_absolute:,.0f}")
        print(f"Average Volume Per Year:            {vol_per_year:,.0f}")
        print("="*65)

    else:
        print("Error: Insufficient matched pairs for regression.")

if __name__ == "__main__":
    main()