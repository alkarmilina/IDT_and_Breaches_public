import pandas as pd
import numpy as np
import json
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/demographics_table.py

"""
Part 2: Weighted Demographic Summary of Victims

Builds weighted summaries of the victim population by year, age group,
sex, race, education, and household income, each summed by survey
weight and expressed as a percentage of the total weighted victim
population. The year breakdown is the source for the paper's
victimization-by-year table. The age, sex, race, education, and income
breakdowns are the victim-side of the paper's victim-versus-Census
demographic comparison.

Setup:
Reads the harmonized victim dataset from part1, and the harmonization
dictionary for demographic category labels and their display order.

Goal:
For each demographic characteristic, compute the weighted count and
weighted percentage of victims in each category, averaged over 2008-2021.

Outputs:
- data/processed/part2/demographic_summary.csv: one row per category,
  across all demographic characteristics.
"""


def generate_weighted_summary(df, variable_name, weight_variable='FINAL_ITS_WEIGHT', category_order=None):
    """Sums survey weights by category for one variable, and expresses each category as a percentage of the total weighted population."""
    df_filtered = df.dropna(subset=[variable_name, weight_variable])

    weighted_counts = df_filtered.groupby(variable_name)[weight_variable].sum()

    if category_order:
        existing_categories = [cat for cat in category_order if cat in weighted_counts.index]
        weighted_counts = weighted_counts.reindex(existing_categories)

    total_weight = weighted_counts.sum()
    weighted_percentages = (weighted_counts / total_weight) * 100

    summary_df = pd.DataFrame({
        'Characteristic': variable_name,
        'Category': weighted_counts.index,
        'Weighted N': weighted_counts.values,
        'Weighted %': weighted_percentages.values
    })

    return summary_df


def main():
    print("\n--- Part 2: Weighted Demographic Summary of Victims ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    input_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    dict_path = os.path.join(root_dir, 'config/harmonization_dictionary.json')

    try:
        df = pd.read_parquet(input_path)
        print(f"Loaded {len(df):,} victims")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Run scripts/part1/harmonize_its.py first.")
        return

    try:
        with open(dict_path, 'r') as f:
            harmonization_plan = json.load(f)['VARIABLE_HARMONIZATION']
        print(f"Loaded data from {dict_path}")
    except (FileNotFoundError, KeyError):
        print(f"Error: {dict_path} not found or is missing the VARIABLE_HARMONIZATION key.")
        return

    all_summaries = []

    if 'year' in df.columns and 'FINAL_ITS_WEIGHT' in df.columns:
        yearly_summary = generate_weighted_summary(df, 'year')
        yearly_summary['Characteristic'] = 'Year of Victimization'
        all_summaries.append(yearly_summary)

    if 'PERSON_AGE' in df.columns:
        df['PERSON_AGE'] = pd.to_numeric(df['PERSON_AGE'], errors='coerce')
        age_bins = [15, 29, 49, 64, np.inf]
        age_labels = ['16-29', '30-49', '50-64', '65+']
        df['Age Group'] = pd.cut(df['PERSON_AGE'], bins=age_bins, labels=age_labels, right=True)
        all_summaries.append(generate_weighted_summary(df, 'Age Group', category_order=age_labels))

    variables_to_summarize = {
        'PERSON_SEX': 'Victim Sex',
        'PERSON_RACE': 'Victim Race/Ethnicity',
        'PERSON_EDUCATION': 'Victim Educational Attainment',
        'HOUSEHOLD_INCOME': 'Victim Household Income'
    }

    for var_name, title in variables_to_summarize.items():
        if var_name in df.columns and var_name in harmonization_plan:
            # Cast to string so values line up with the harmonization dictionary's string-keyed code map.
            df[var_name] = df[var_name].astype(str).str.replace('.0', '', regex=False)

            mapping = harmonization_plan[var_name]['proposed_harmonized_map']
            code_to_label_map = {str(v): k for k, v in mapping.items()}

            sorted_items = sorted(mapping.items(), key=lambda item: item[1])
            category_order = [item[0] for item in sorted_items]

            df[title] = df[var_name].map(code_to_label_map)

            if var_name == 'PERSON_RACE' and 'PERSON_HISPANIC_ORIGIN' in df.columns:
                h_map = harmonization_plan['PERSON_HISPANIC_ORIGIN']['proposed_harmonized_map']
                hispanic_yes_code = str(h_map.get('Yes')).replace('.0', '')
                df.loc[df['PERSON_HISPANIC_ORIGIN'].astype(str).str.replace('.0', '', regex=False) == hispanic_yes_code, title] = 'Hispanic'

            all_summaries.append(generate_weighted_summary(df, title, category_order=category_order))

    if not all_summaries:
        print("\nNo data to summarize. Exiting.")
        return

    final_table = pd.concat(all_summaries, ignore_index=True)

    output_dir = os.path.join(root_dir, "data/processed/part2")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demographic_summary.csv")
    final_table.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")

    final_table['Weighted N'] = final_table['Weighted N'].map('{:,.0f}'.format)
    final_table['Weighted %'] = final_table['Weighted %'].map('{:.2f}%'.format)

    print("\nDemographic summary (2008-2021):")
    print(final_table.to_csv(index=False))


if __name__ == '__main__':
    main()
