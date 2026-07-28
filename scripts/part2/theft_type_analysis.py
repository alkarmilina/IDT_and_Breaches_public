import pandas as pd
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/theft_type_analysis.py

"""
Part 2: Incident Characteristics by Year

Builds a year-by-year summary of theft type, discovery method, and how
victims' information was obtained, each expressed as a weighted count
and percentage of that year's total victims. Averaging each category's
percentage across the six years produces the paper's incident summary
table.

Setup:
Reads the harmonized victim dataset from part1. Codes for discovery
method and theft method match the harmonization dictionary. Theft type
uses inclusive filtering, a victim who experienced two types counts in
both, plus a separate "Multiple Types" row tracks that overlap.

Goal:
Produce the year-by-year data behind the paper's incident summary
table.

Outputs:
- data/processed/part2/combined_theft_analysis_by_year.csv: one row per
  category (theft type, discovery method, or obtainment method), one
  pair of columns (count, percentage) per year.
"""


def analyze_theft_data_by_year():
    print("\n--- Part 2: Incident Characteristics by Year ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    harmonized_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    output_folder = os.path.join(root_dir, 'data/processed/part2')
    output_csv_path = os.path.join(output_folder, 'combined_theft_analysis_by_year.csv')

    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_parquet(harmonized_data_path)
        print(f"Loaded data from {harmonized_data_path}")
    except FileNotFoundError:
        print(f"Error: {harmonized_data_path} not found. Run scripts/part1/harmonize_its.py first.")
        return

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

    df['HOW_DISCOVERED_MISUSE'] = pd.to_numeric(df['HOW_DISCOVERED_MISUSE'], errors='coerce')
    df['THEFT_METHOD'] = pd.to_numeric(df['THEFT_METHOD'], errors='coerce')
    for var in theft_categories.values():
        df[var] = pd.to_numeric(df[var], errors='coerce')

    years_to_analyze = sorted(df['year'].unique())
    total_victims_by_year = df.groupby('year')['FINAL_ITS_WEIGHT'].sum()

    results_theft_type = pd.DataFrame(index=list(theft_categories.keys()) + ['Multiple Types'])
    results_discovery = pd.DataFrame(index=discovery_categories.keys())
    results_theft_method = pd.DataFrame(index=theft_method_categories.keys())

    for year in years_to_analyze:
        year_df = df[df['year'] == year].copy()
        total_year_victims = total_victims_by_year.get(year, 0)

        if total_year_victims == 0: continue

        # Theft type
        for display_name, variable_name in theft_categories.items():
            weighted_count = year_df[year_df[variable_name] == 1]['FINAL_ITS_WEIGHT'].sum()
            results_theft_type.loc[display_name, f'{year} (N)'] = weighted_count
            results_theft_type.loc[display_name, f'{year} (%)'] = (weighted_count / total_year_victims) * 100

        theft_vars = list(theft_categories.values())
        year_df['type_count'] = (year_df[theft_vars] == 1).sum(axis=1)
        multiple_count = year_df[year_df['type_count'] > 1]['FINAL_ITS_WEIGHT'].sum()
        results_theft_type.loc['Multiple Types', f'{year} (N)'] = multiple_count
        results_theft_type.loc['Multiple Types', f'{year} (%)'] = (multiple_count / total_year_victims) * 100

        # Discovery method
        for display_name, code in discovery_categories.items():
            count = year_df[year_df['HOW_DISCOVERED_MISUSE'] == code]['FINAL_ITS_WEIGHT'].sum()
            results_discovery.loc[display_name, f'{year} (N)'] = count
            results_discovery.loc[display_name, f'{year} (%)'] = (count / total_year_victims) * 100

        # Theft method (how information was obtained)
        for display_name, code in theft_method_categories.items():
            count = year_df[year_df['THEFT_METHOD'] == code]['FINAL_ITS_WEIGHT'].sum()
            results_theft_method.loc[display_name, f'{year} (N)'] = count
            results_theft_method.loc[display_name, f'{year} (%)'] = (count / total_year_victims) * 100

    combined_results = pd.concat(
        [results_theft_type, results_discovery, results_theft_method],
        keys=['Theft Type', 'Discovery Method', 'Obtainment Method'],
        names=['Section', 'Category']
    )

    combined_results.to_csv(output_csv_path)
    print(f"Data saved to {output_csv_path}")

    preview_df = combined_results.copy()
    for col in preview_df.columns:
        if '(N)' in col:
            preview_df[col] = preview_df[col].map('{:,.0f}'.format)
        else:
            preview_df[col] = preview_df[col].map('{:.1f}%'.format)

    print("\nTrend analysis preview:")
    print(preview_df.head(15))


if __name__ == '__main__':
    analyze_theft_data_by_year()
