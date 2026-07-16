import pandas as pd
import numpy as np
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/race_profile_comparison.py

"""
Analysis: Part 2 - Racial Disparities in Identity Theft Characteristics (Table 8)
"""

def weighted_percentage_for_section(df, col_name, code_or_var, weight_col):
    """Calculates the weighted percentage based on the category type."""
    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return 0
    
    # Binary indicator (Theft Types)
    if 'EXISTING' in str(col_name) or 'NEW_ACCT' in str(col_name) or 'OTHER_FRAUD' in str(col_name):
        condition_met_weight = df[df[col_name] == 1][weight_col].sum()
    # Categorical (Discovery/Method)
    else:
        condition_met_weight = df[df[col_name] == code_or_var][weight_col].sum()
        
    return (condition_met_weight / total_weight) * 100

def main():
    # --- 1. Path Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    
    input_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    output_folder = os.path.join(root_dir, 'data/processed/part2')
    output_csv_path = os.path.join(output_folder, 'victim_profile_comparison_by_race.csv')

    os.makedirs(output_folder, exist_ok=True)

    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_parquet(input_path)
        print(f"Data loaded successfully from {input_path}")
    except Exception as e:
        print(f"ERROR: Could not load data. {e}")
        return

    # Configuration
    RACE_CODES = {'White': 1, 'Black': 2, 'Asian': 4}
    WEIGHT_COLUMN = 'FINAL_ITS_WEIGHT'

    theft_categories = {
        'Existing Bank Account Misuse': 'EXISTING_BANK_ACCT_MISUSE_12MO',
        'Existing Credit Card Misuse': 'EXISTING_CC_MISUSE_12MO',
        'Other Existing Account Misuse': 'EXISTING_OTHER_ACCT_MISUSE_12MO',
        'New Account Fraud': 'NEW_ACCT_OPENED_12MO',
        'Other Misuse of Personal Information': 'OTHER_FRAUDULENT_PURPOSE_12MO'
    }

    discovery_categories = {
        'Noticed Problem with Account': 1,
        'Notified by Financial Institution': 2,
        'Notified by Other Person/Organization': 3,
        'Received Bill/Item Not Ordered': 4,
        'Checked Credit Report': 5,
        'Other/Unspecified': 7,
        'Don\'t Know': 8
    }

    theft_method_categories = {
        'Lost or Stolen Physical Item': 1,
        'Stolen During Transaction': 2,
        'Hacking/Computer Theft': 3,
        'Deceived by Scam/Phishing': 4,
        'Data Breach (Company/Employer)': 5,
        'Other': 7
    }

    # Data Type Cleanup
    cols_to_numeric = ['PERSON_RACE', WEIGHT_COLUMN, 'HOW_DISCOVERED_MISUSE', 'THEFT_METHOD'] + \
                      list(theft_categories.values())
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 3. Segmentation ---
    white_df = df[df['PERSON_RACE'] == RACE_CODES['White']].copy()
    black_df = df[df['PERSON_RACE'] == RACE_CODES['Black']].copy()
    asian_df = df[df['PERSON_RACE'] == RACE_CODES['Asian']].copy()
    other_df = df[~df['PERSON_RACE'].isin([1, 2, 4])].copy()

    # --- 4. Profile Generation ---
    analysis_sections = {
        'Type': theft_categories,
        'Discovery': discovery_categories,
        'Method': theft_method_categories
    }
    
    comparison_groups = {
        'White (%)': white_df,
        'Black (%)': black_df,
        'Asian (%)': asian_df,
        'All Other (%)': other_df
    }
    
    final_results = {}
    
    for section_name, categories in analysis_sections.items():
        if section_name == 'Discovery': col_name = 'HOW_DISCOVERED_MISUSE'
        elif section_name == 'Method': col_name = 'THEFT_METHOD'
        else: col_name = None 

        for display_name, code_or_var in categories.items():
            key = f"{section_name}: {display_name}"
            final_results[key] = {}
            
            for group_name, group_df in comparison_groups.items():
                calc_col = code_or_var if section_name == 'Type' else col_name
                calc_code = 1 if section_name == 'Type' else code_or_var
                
                pct = weighted_percentage_for_section(group_df, calc_col, calc_code, WEIGHT_COLUMN)
                final_results[key][group_name] = pct
            
            # Differences vs. All Other
            ref_pct = final_results[key]['All Other (%)']
            final_results[key]['Diff (White - Other)'] = final_results[key]['White (%)'] - ref_pct
            final_results[key]['Diff (Black - Other)'] = final_results[key]['Black (%)'] - ref_pct
            final_results[key]['Diff (Asian - Other)'] = final_results[key]['Asian (%)'] - ref_pct

    # --- 5. Saving and Display ---
    results_df = pd.DataFrame.from_dict(final_results, orient='index')
    results_df.to_csv(output_csv_path)
    print(f"Analysis complete. Results saved to: {output_csv_path}")

    # Formatting for terminal
    cols_to_show = ['White (%)', 'Black (%)', 'Asian (%)', 'All Other (%)', 
                    'Diff (White - Other)', 'Diff (Black - Other)', 'Diff (Asian - Other)']
    formatted_df = results_df[cols_to_show].copy()
    
    for col in formatted_df.columns:
        if 'Diff' in col:
            formatted_df[col] = formatted_df[col].map('{:+.2f}%'.format)
        else:
            formatted_df[col] = formatted_df[col].map('{:.2f}%'.format)
    
    print("\n" + "="*110)
    print("Victim Profile Comparison: Race/Ethnicity (Table 8 Preview)")
    print("="*110)
    print(formatted_df.head(15).to_string())

if __name__ == '__main__':
    main()