import pandas as pd
import os
from thefuzz import fuzz, process

# To run this script:
# python scripts/part3/PRC_fill_in_health.py

"""
Part 3: PRC-HHS Data Integration (Fuzzy Name Matching)
Supplements the deduplicated PRC base with HHS record counts using 
the 0.7 (70%) threshold defined in the research methodology.
"""

def main():
    # --- 1. DYNAMIC PATH CONFIGURATION ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Input: Deduplicated output from the previous loading script
    clean_base_path = os.path.join(root_dir, 'data/raw/PRC/loaded/clean_prc_base.csv')
    # Supplement: HHS Breach Portal data
    hhs_path = os.path.join(root_dir, 'data/raw/PRC_supplements/HHS_breach_report.csv')
    
    # Output: Processed integrated file
    output_dir = os.path.join(root_dir, 'data/processed/part3')
    output_path = os.path.join(output_dir, 'prc_hhs_integrated.csv')
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. LOAD DEDUPLICATED PRC BASE ---
    if not os.path.exists(clean_base_path):
        print(f"ERROR: Clean base not found at {clean_base_path}. Please run load_raw_PRC_data.py first.")
        return

    print("Loading deduplicated PRC base data...")
    prc_df = pd.read_csv(clean_base_path)
    
    # Standardize types and dates
    prc_df['total_affected'] = pd.to_numeric(prc_df['total_affected'], errors='coerce').fillna(0)
    prc_df['reported_date'] = pd.to_datetime(prc_df['reported_date'], errors='coerce')
    prc_df['Year'] = prc_df['reported_date'].dt.year

    initial_zeros = (prc_df['total_affected'] == 0).sum()
    print(f"Starting with {initial_zeros} unique events missing record counts.")

    # --- 3. LOAD HHS SUPPLEMENT ---
    if not os.path.exists(hhs_path):
        print(f"ERROR: HHS report not found at {hhs_path}")
        return

    hhs_df = pd.read_csv(hhs_path)
    hhs_df['Year'] = pd.to_datetime(hhs_df['Breach Submission Date'], errors='coerce').dt.year
    hhs_df['hhs_count'] = pd.to_numeric(hhs_df['Individuals Affected'], errors='coerce').fillna(0)

    # --- 4. CLEANING & FUZZY MATCHING LOGIC ---
    def clean_name(name):
        if pd.isna(name): return ""
        # Remove common business suffixes to isolate core entity name
        return str(name).lower().strip().replace(".", "").replace(",", "").replace(" inc", "").replace(" corp", "").replace(" llc", "")

    prc_df['clean_org'] = prc_df['org_name'].apply(clean_name)
    hhs_df['clean_org'] = hhs_df['Name of Covered Entity'].apply(clean_name)

    matches_found = []
    # Counter for instances where both sources have values
    dual_value_matches = [0] 
    # Track HHS records that were matched to PRC
    matched_hhs_indices = set()

    def find_hhs_match(prc_row, hhs_data, threshold=70):
        if pd.isna(prc_row['Year']):
            return prc_row['total_affected']
        
        # Match only within the same year to ensure historical accuracy
        year_subset = hhs_data[hhs_data['Year'] == prc_row['Year']]
        if year_subset.empty: 
            return prc_row['total_affected']
        
        # Using token_sort_ratio as established in Section 3.2
        match_result = process.extractOne(prc_row['clean_org'], year_subset['clean_org'], scorer=fuzz.token_sort_ratio)
        
        if match_result:
            best_match, score, index = match_result
            # Threshold set to 70 per paper's high-confidence match rules 
            if score >= threshold:
                matched_hhs_indices.add(index)
                hhs_val = year_subset.loc[index, 'hhs_count']
                
                # Check if both PRC and HHS have values to track conflicts
                if prc_row['total_affected'] > 0 and hhs_val > 0:
                    dual_value_matches[0] += 1

                # Take the max value to capture upper-bound impact
                val = max(prc_row['total_affected'], hhs_val)

                # Only log as a "found match" for the sanity check if it was originally 0
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

    print(f"Scanning unique events for HHS matches (Threshold: 70)...")
    prc_df['total_affected'] = prc_df.apply(lambda row: find_hhs_match(row, hhs_df), axis=1)

    # Calculate HHS events that had no match in PRC
    new_hhs_events = len(hhs_df) - len(matched_hhs_indices)

    # --- 5. SANITY CHECK & SAVE ---
    final_zeros = (prc_df['total_affected'] == 0).sum()
    
    print("\n" + "="*40)
    print("--- HHS ENRICHMENT SANITY CHECK ---")
    print(f"Initial unique zero-count events: {initial_zeros}")
    print(f"HHS matches successfully found:    {len(matches_found)}")
    print(f"Conflicts resolved by taking max:  {dual_value_matches[0]}")
    print(f"New unique HHS events integrated:  {new_hhs_events}")
    print(f"Remaining zero-count events:       {final_zeros}")
    print("="*40)

    # Show sample results to verify match quality
    if matches_found:
        print("\n--- SAMPLE HHS MATCHES (Threshold 70) ---")
        for m in matches_found[:5]:
            print(f"Matched: {m['PRC_Org']} -> {m['HHS_Match']} (Score: {m['Score']}) | Count: {m['New_Count']}")

    # Drop temporary cleaning column before final save
    prc_df = prc_df.drop(columns=['clean_org'])
    prc_df.to_csv(output_path, index=False)
    print(f"\nSUCCESS: Integrated dataset saved to: {output_path}")

if __name__ == "__main__":
    main()