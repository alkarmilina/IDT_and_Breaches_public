import pandas as pd
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/educational_profile_comparison.py

"""
Part 2: Victim Profile Comparison by Education Level

Compares identity theft victims with a Bachelor's or Graduate/Professional
degree against all other victims, across theft type, discovery method,
and theft method, to test whether the two groups' overrepresentation
among victims is driven by disproportionate exposure to certain kinds
of fraud.

Setup:
Reads the harmonized victim dataset from part1. Codes for education,
discovery method, and theft method match the harmonization dictionary.

Goal:
For each of the three education groups, compute the weighted percentage
of victims falling into each theft type, discovery method, and theft
method category, and the difference between each high-education group
and the baseline (all other victims).

Outputs:
- data/processed/part2/victim_profile_comparison_by_education.csv: one
  row per category, one column per education group plus the two
  difference-from-baseline columns.
"""


def calculate_weighted_profile(df, theft_cats, disc_cats, method_cats, weight_col='FINAL_ITS_WEIGHT'):
    """
    Computes the weighted percentage of df falling into each theft type,
    discovery method, and theft method category, plus the share of
    victims who experienced more than one theft type.

    Args:
        df (pd.DataFrame): victims in one education group.
        theft_cats (dict): display name -> theft type indicator column.
        disc_cats (dict): display name -> HOW_DISCOVERED_MISUSE code.
        method_cats (dict): display name -> THEFT_METHOD code.
        weight_col (str): survey weight column.

    Returns:
        pd.Series: weighted percentage, indexed by category label.
    """
    if df.empty:
        return pd.Series(dtype=float)

    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return pd.Series(dtype=float)

    profile = {}
    profile_index_order = []

    # Theft types
    theft_vars = list(theft_cats.values())
    for display_name, variable_name in theft_cats.items():
        idx = f'Type: {display_name}'
        profile_index_order.append(idx)
        weighted_count = df[df[variable_name] == 1][weight_col].sum()
        profile[idx] = (weighted_count / total_weight) * 100

    # Multiple types
    profile_index_order.append('Type: Multiple Types')
    df = df.copy()
    df['theft_type_count'] = (df[theft_vars] == 1).sum(axis=1)
    multiple_weighted_count = df[df['theft_type_count'] > 1][weight_col].sum()
    profile['Type: Multiple Types'] = (multiple_weighted_count / total_weight) * 100

    # Discovery methods
    for display_name, category_code in disc_cats.items():
        idx = f'Discovery: {display_name}'
        profile_index_order.append(idx)
        weighted_count = df[df['HOW_DISCOVERED_MISUSE'] == category_code][weight_col].sum()
        profile[idx] = (weighted_count / total_weight) * 100

    # Theft methods
    for display_name, category_code in method_cats.items():
        idx = f'Method: {display_name}'
        profile_index_order.append(idx)
        weighted_count = df[df['THEFT_METHOD'] == category_code][weight_col].sum()
        profile[idx] = (weighted_count / total_weight) * 100

    return pd.Series(profile, index=profile_index_order)


def main():
    print("\n--- Part 2: Victim Profile Comparison by Education Level ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    input_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    output_folder = os.path.join(root_dir, 'data/processed/part2')
    output_csv_path = os.path.join(output_folder, 'victim_profile_comparison_by_education.csv')

    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_parquet(input_path)
        print(f"Loaded data from {input_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Education codes, as defined in harmonization_dictionary.json.
    bachelors_code = 6
    grad_code = 7
    other_codes = [1, 2, 3, 4, 5]

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

    cols_to_numeric = ['PERSON_EDUCATION', 'FINAL_ITS_WEIGHT', 'HOW_DISCOVERED_MISUSE', 'THEFT_METHOD'] + \
                      list(theft_categories.values())
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    grad_victims = df[df['PERSON_EDUCATION'] == grad_code].copy()
    bach_victims = df[df['PERSON_EDUCATION'] == bachelors_code].copy()
    other_victims = df[df['PERSON_EDUCATION'].isin(other_codes)].copy()

    grad_profile = calculate_weighted_profile(grad_victims, theft_categories, discovery_categories, theft_method_categories)
    bach_profile = calculate_weighted_profile(bach_victims, theft_categories, discovery_categories, theft_method_categories)
    other_profile = calculate_weighted_profile(other_victims, theft_categories, discovery_categories, theft_method_categories)

    comparison_df = pd.DataFrame({
        'Graduate/Professional': grad_profile,
        'Bachelor\'s Degree': bach_profile,
        'All Other Victims': other_profile
    })

    comparison_df['Diff (Grad - Other)'] = comparison_df['Graduate/Professional'] - comparison_df['All Other Victims']
    comparison_df['Diff (Bach - Other)'] = comparison_df['Bachelor\'s Degree'] - comparison_df['All Other Victims']

    comparison_df.to_csv(output_csv_path)
    print(f"Analysis complete. Results saved to {output_csv_path}")

    print("\n" + "="*80)
    print("Education Profile Comparison (Preview)")
    print("="*80)
    print(comparison_df.head(10).to_string())


if __name__ == '__main__':
    main()
