import pandas as pd
import numpy as np
import os
import json

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/calculate_social_cost.py

"""
Part 2: Social Cost of Identity Theft

Computes the per-victim social cost of identity theft, made up of
out-of-pocket loss, legal cost, lost time cost, and healthcare cost, as
defined in the paper's social cost model. Produces the year-by-year
cost table and the demographic breakdowns of social cost by age,
education, income, and race.

Setup:
Reads the harmonized victim dataset from part1, and the harmonization
dictionary for demographic category labels. All monetary values are
adjusted to constant 2021 dollars using CPI, and lost time is valued at
the year's inflation-adjusted average hourly wage. Legal, medical,
therapy, and medication costs are fixed per-incident costs, applied to
whoever reported using that kind of help.

Goal:
Produce the paper's year-by-year social cost table and demographic
social cost breakdowns.

Outputs:
- data/processed/part2/social_cost/social_cost_analysis.csv: one row
  per year, the year-by-year social cost table.
- data/processed/part2/social_cost/social_cost_by_<demographic>.csv:
  one file per demographic (age group, education, income, race), the
  average social cost per victim within each group.
"""

# Economic constants (constant 2021 dollars), from the paper's CPI, wage, and fixed cost tables.
CPI_VALUES = {2008: 215.297, 2012: 233.165, 2014: 236.736, 2016: 240.007, 2018: 251.107, 2021: 270.970}
CPI_TARGET_YEAR_VALUE = 270.970
ADJUSTED_HOURLY_WAGES = {2008: 26.67, 2012: 27.03, 2014: 27.73, 2016: 28.64, 2018: 28.83, 2021: 29.92}
ADJUSTED_FIXED_COSTS = {"legal": 444.65, "medical": 133.60, "therapy": 88.92, "medication": 53.96}


def weighted_average(df, value_col, weight_col):
    """Survey-weighted mean of value_col, dropping rows missing either value_col or weight_col."""
    good_data = df.dropna(subset=[value_col, weight_col]).copy()
    good_data[value_col] = pd.to_numeric(good_data[value_col], errors='coerce')
    good_data = good_data.dropna(subset=[value_col])
    return np.average(good_data[value_col], weights=good_data[weight_col]) if not good_data.empty else 0


def prepare_demographic_columns(df, harmonization_plan):
    """Adds age group bins and human-readable labels for race, education, and income, using the harmonization dictionary's code maps."""
    df = df.copy()
    if 'PERSON_AGE' in df.columns:
        df['PERSON_AGE'] = pd.to_numeric(df['PERSON_AGE'], errors='coerce')
        df['Age Group'] = pd.cut(df['PERSON_AGE'], bins=[15, 29, 49, 64, np.inf],
                                 labels=['16-29', '30-49', '50-64', '65+'], right=True)

    variables_to_map = {
        'PERSON_RACE': 'Victim Race/Ethnicity',
        'PERSON_EDUCATION': 'Victim Educational Attainment',
        'HOUSEHOLD_INCOME': 'Victim Household Income'
    }
    for var_name, title in variables_to_map.items():
        if var_name in df.columns and var_name in harmonization_plan:
            mapping = harmonization_plan[var_name]['proposed_harmonized_map']
            code_to_label_map = {str(v).replace('.0',''): k for k, v in mapping.items()}
            df[title] = df[var_name].astype(str).str.replace('.0','', regex=False).map(code_to_label_map)

            if var_name == 'PERSON_RACE' and 'PERSON_HISPANIC_ORIGIN' in df.columns:
                h_map = harmonization_plan['PERSON_HISPANIC_ORIGIN']['proposed_harmonized_map']
                his_code = str(h_map.get('Yes')).replace('.0','')
                df.loc[df['PERSON_HISPANIC_ORIGIN'].astype(str).str.replace('.0','', regex=False) == his_code, title] = 'Hispanic'
    return df


def run_longitudinal_analysis(df):
    """Builds the year-by-year social cost table, one row per survey year with the four cost components and the total per-victim and national social cost."""
    results = []
    years = sorted(df['year'].unique())
    for year in years:
        y_df = df[df['year'] == year].copy()
        total_v = y_df['FINAL_ITS_WEIGHT'].sum()
        if total_v == 0: continue

        cpi_r = CPI_TARGET_YEAR_VALUE / CPI_VALUES.get(year, CPI_TARGET_YEAR_VALUE)
        oop = weighted_average(y_df, 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT', 'FINAL_ITS_WEIGHT') * cpi_r
        time = weighted_average(y_df, 'HOURS_SPENT_RESOLVING_PROBLEMS', 'FINAL_ITS_WEIGHT') * ADJUSTED_HOURLY_WAGES.get(year, 0)

        # Financial/legal costs
        legal = (y_df[y_df['CONTACT_HIRED_LAWYER'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['legal']) / total_v

        # Health costs
        med = y_df[y_df['HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['medical']
        thr = y_df[y_df['HELP_TYPE_COUNSELING'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['therapy']
        rx = y_df[y_df['HELP_TYPE_MEDICATION'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['medication']
        health = (med + thr + rx) / total_v

        total_p_v = oop + legal + time + health
        results.append({
            'Year': year,
            'Total Weighted Victims': total_v,
            'Avg. Out-of-Pocket Loss ($)': oop,
            'Avg. Legal Cost ($)': legal,
            'Avg. Lost Time Cost ($)': time,
            'Avg. Healthcare Cost ($)': health,
            'Total Social Cost per Victim ($)': total_p_v,
            'Total National Social Cost ($)': total_p_v * total_v
        })
    return pd.DataFrame(results)


def run_demographic_analysis(df, group_by_col):
    """Averages the total social cost per victim within each level of group_by_col, pooled across all years."""
    df_clean = df.dropna(subset=[group_by_col]).copy()
    results = []
    for name in sorted(df_clean[group_by_col].unique()):
        g_df = df_clean[df_clean[group_by_col] == name]
        total_w_v = g_df['FINAL_ITS_WEIGHT'].sum()
        if total_w_v == 0: continue

        total_nat_cost = 0
        for year in df['year'].unique():
            y_df = g_df[g_df['year'] == year]
            if y_df['FINAL_ITS_WEIGHT'].sum() == 0: continue
            cpi_r = CPI_TARGET_YEAR_VALUE / CPI_VALUES.get(year, CPI_TARGET_YEAR_VALUE)

            oop = (pd.to_numeric(y_df['OUT_OF_POCKET_LOSS_RECENT_INCIDENT'], errors='coerce').fillna(0) * cpi_r * y_df['FINAL_ITS_WEIGHT']).sum()
            time = (pd.to_numeric(y_df['HOURS_SPENT_RESOLVING_PROBLEMS'], errors='coerce').fillna(0) * ADJUSTED_HOURLY_WAGES.get(year, 0) * y_df['FINAL_ITS_WEIGHT']).sum()
            legal = y_df[y_df['CONTACT_HIRED_LAWYER'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['legal']
            health = (y_df[y_df['HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['medical']) + \
                     (y_df[y_df['HELP_TYPE_COUNSELING'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['therapy']) + \
                     (y_df[y_df['HELP_TYPE_MEDICATION'].astype(str).str.contains('1')]['FINAL_ITS_WEIGHT'].sum() * ADJUSTED_FIXED_COSTS['medication'])
            total_nat_cost += (oop + time + legal + health)

        results.append({group_by_col: name, 'Avg. Social Cost per Victim (2008-2021, 2021$)': total_nat_cost / total_w_v})
    return pd.DataFrame(results)


def main():
    print("\n--- Part 2: Social Cost of Identity Theft ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    input_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    dict_path = os.path.join(root_dir, 'config/harmonization_dictionary.json')

    output_folder = os.path.join(root_dir, 'data/processed/part2/social_cost')
    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_parquet(input_path)
        print(f"Loaded data from {input_path}")
        with open(dict_path, 'r') as f:
            plan = json.load(f)['VARIABLE_HARMONIZATION']
        print(f"Loaded data from {dict_path}")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("Building longitudinal social cost table...")
    master_results = run_longitudinal_analysis(df)
    master_results.to_csv(os.path.join(output_folder, 'social_cost_analysis.csv'), index=False)

    df = prepare_demographic_columns(df, plan)

    demographics = ['Age Group', 'Victim Educational Attainment', 'Victim Household Income', 'Victim Race/Ethnicity']
    for col in demographics:
        print(f"Analyzing demographic burden: {col}...")
        res = run_demographic_analysis(df, col)
        safe_col = col.lower().replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_folder, f"social_cost_by_{safe_col}.csv")
        res.to_csv(output_path, index=False)

    print(f"\nSocial cost analysis complete. Data saved to {output_folder}")


if __name__ == '__main__':
    main()
