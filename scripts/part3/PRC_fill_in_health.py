import pandas as pd
import os
from thefuzz import fuzz, process

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part3/PRC_fill_in_health.py

"""
Part 3: PRC-HHS Data Integration

Supplements the deduplicated PRC base with HHS Breach Portal data in
three ways: adding HHS-reported breaches that are entirely absent from
PRC as new incident rows, resolving conflicts where both sources report
a different record count by keeping the higher value, and filling in
PRC incidents that originally reported a zero or undisclosed count.

Setup:
Reads the deduplicated PRC base from load_raw_PRC_data.py, and the raw
HHS Breach Portal download. Organization names are cleaned (lowercased,
common suffixes like "inc"/"corp"/"llc" stripped) and matched with
thefuzz's token_sort_ratio, restricted to breaches reported in the same
year, using a 70% match threshold for a high-confidence match.

Goal:
Produce the paper's PRC-HHS integration step.

Outputs:
- data/processed/part3/prc_hhs_integrated.csv: the PRC base with HHS
  record counts merged in and unmatched HHS breaches appended as new
  rows.
"""


def clean_name(name):
    """Lowercases an organization name and strips common business suffixes, to improve fuzzy match quality."""
    if pd.isna(name): return ""
    return str(name).lower().strip().replace(".", "").replace(",", "").replace(" inc", "").replace(" corp", "").replace(" llc", "")


def main():
    print("\n--- Part 3: PRC-HHS Data Integration ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    clean_base_path = os.path.join(root_dir, 'data/raw/PRC/loaded/clean_prc_base.csv')
    hhs_path = os.path.join(root_dir, 'data/raw/PRC_supplements/HHS_breach_report.csv')

    output_dir = os.path.join(root_dir, 'data/processed/part3')
    output_path = os.path.join(output_dir, 'prc_hhs_integrated.csv')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(clean_base_path):
        print(f"Error: {clean_base_path} not found. Run scripts/part3/load_raw_PRC_data.py first.")
        return
    if not os.path.exists(hhs_path):
        print(f"Error: HHS report not found at {hhs_path}")
        return

    prc_df = pd.read_csv(clean_base_path)
    print(f"Loaded data from {clean_base_path}")

    prc_df['total_affected'] = pd.to_numeric(prc_df['total_affected'], errors='coerce').fillna(0)
    prc_df['reported_date'] = pd.to_datetime(prc_df['reported_date'], errors='coerce')
    prc_df['Year'] = prc_df['reported_date'].dt.year

    initial_zeros = (prc_df['total_affected'] == 0).sum()
    print(f"Starting with {initial_zeros} unique events missing record counts.")

    hhs_df = pd.read_csv(hhs_path)
    print(f"Loaded data from {hhs_path}")
    hhs_df['Year'] = pd.to_datetime(hhs_df['Breach Submission Date'], errors='coerce').dt.year
    hhs_df['hhs_count'] = pd.to_numeric(hhs_df['Individuals Affected'], errors='coerce').fillna(0)

    prc_df['clean_org'] = prc_df['org_name'].apply(clean_name)
    hhs_df['clean_org'] = hhs_df['Name of Covered Entity'].apply(clean_name)

    matches_found = []
    dual_value_matches = [0]  # Counter for instances where both sources have values.
    matched_hhs_indices = set()  # Tracks HHS records that were matched to an existing PRC row.

    def find_hhs_match(prc_row, hhs_data, threshold=70):
        if pd.isna(prc_row['Year']):
            return prc_row['total_affected']

        # Match only within the same year to ensure historical accuracy.
        year_subset = hhs_data[hhs_data['Year'] == prc_row['Year']]
        if year_subset.empty:
            return prc_row['total_affected']

        match_result = process.extractOne(prc_row['clean_org'], year_subset['clean_org'], scorer=fuzz.token_sort_ratio)

        if match_result:
            best_match, score, index = match_result
            if score >= threshold:
                matched_hhs_indices.add(index)
                hhs_val = year_subset.loc[index, 'hhs_count']

                if prc_row['total_affected'] > 0 and hhs_val > 0:
                    dual_value_matches[0] += 1

                # Take the max value to capture the upper-bound impact, whether this
                # resolves a conflict between two nonzero values or fills in a zero.
                val = max(prc_row['total_affected'], hhs_val)

                if prc_row['total_affected'] == 0 and val > 0:
                    matches_found.append({
                        'Year': prc_row['Year'],
                        'PRC_Org': prc_row['org_name'],
                        'HHS_Match': best_match,
                        'Score': score,
                        'New_Count': val
                    })
                return val
        return prc_row['total_affected']

    print("Scanning unique events for HHS matches (threshold: 70)...")
    prc_df['total_affected'] = prc_df.apply(lambda row: find_hhs_match(row, hhs_df), axis=1)

    # HHS breaches with no match in PRC are entirely new incidents, appended as new rows.
    unmatched_hhs = hhs_df[~hhs_df.index.isin(matched_hhs_indices)]
    new_rows = pd.DataFrame({
        'org_name': unmatched_hhs['Name of Covered Entity'],
        'total_affected': unmatched_hhs['hhs_count'],
        'reported_date': pd.to_datetime(unmatched_hhs['Breach Submission Date'], errors='coerce'),
        'Year': unmatched_hhs['Year'],
        'source': 'HHS'
    })
    new_hhs_events = len(new_rows)
    prc_df = pd.concat([prc_df, new_rows], ignore_index=True)

    final_zeros = (prc_df['total_affected'] == 0).sum()

    print("\n" + "="*40)
    print("HHS enrichment summary")
    print(f"Initial unique zero-count events: {initial_zeros}")
    print(f"HHS matches found:                {len(matches_found)}")
    print(f"Conflicts resolved by taking max: {dual_value_matches[0]}")
    print(f"New unique HHS events integrated: {new_hhs_events}")
    print(f"Remaining zero-count events:      {final_zeros}")
    print("="*40)

    if matches_found:
        print("\nSample HHS matches (threshold 70)")
        for m in matches_found[:5]:
            print(f"  {m['PRC_Org']} -> {m['HHS_Match']} (score: {m['Score']}, count: {m['New_Count']})")

    prc_df = prc_df.drop(columns=['clean_org'])
    prc_df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
