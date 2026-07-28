import pandas as pd
import numpy as np
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/social_cost_breach_vs_nonbreach.py

"""
Part 2: Social Cost by Breach Notification Status

Compares social cost between victims who were notified their data was
exposed in a breach and victims who were not, and separately compares
victims whose breached data included their Social Security Number
against victims whose breach did not include an SSN.

Setup:
Reads the harmonized victim dataset from part1. Uses the same social
cost model, out-of-pocket loss, legal cost, lost time cost, and
healthcare cost, as the year-by-year social cost analysis, applied
within each breach-status group instead of within each year.

Goal:
Produce the paper's breach-notification and SSN-breach social cost
comparisons.

Outputs:
- data/processed/part2/breach_cost_analysis/social_cost_breach_vs_nonbreach.csv
- data/processed/part2/breach_cost_analysis/social_cost_ssn_vs_nonssn.csv
"""

# Economic constants (constant 2021 dollars), from the paper's CPI, wage, and fixed cost tables.
ADJUSTED_HOURLY_WAGES = {
    2008: 26.67, 2012: 27.03, 2014: 27.73,
    2016: 28.64, 2018: 28.83, 2021: 29.92
}
ADJUSTED_FIXED_COSTS = {
    "legal": 444.65, "medical": 133.60,
    "therapy": 88.92, "medication": 53.96
}
CPI_VALUES = {
    2008: 215.297, 2012: 233.165, 2014: 236.736,
    2016: 240.007, 2018: 251.107, 2021: 270.970
}
CPI_TARGET_YEAR_VALUE = 270.970


def weighted_average(df, value_col, weight_col):
    """Survey-weighted mean of value_col, dropping rows missing either value_col or weight_col."""
    good_data = df.dropna(subset=[value_col, weight_col])
    if good_data.empty:
        return 0
    return np.average(good_data[value_col], weights=good_data[weight_col])


def calculate_social_cost_for_group(df_group, year, group_name):
    """
    Computes the four social cost components and the total per-victim
    and national social cost for one group of victims in one year.

    Args:
        df_group (pd.DataFrame): victims in one group (e.g. breach
            notified) for one year.
        year (int): survey year, used to look up that year's CPI ratio
            and hourly wage.
        group_name (str): label attached to the result row.

    Returns:
        dict or None: cost breakdown for this group and year, or None
        if the group has no weighted victims that year.
    """
    total_victims = df_group['FINAL_ITS_WEIGHT'].sum()
    if total_victims == 0:
        return None

    nominal_avg_oop_loss = weighted_average(df_group, 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT', 'FINAL_ITS_WEIGHT')
    cpi_ratio = CPI_TARGET_YEAR_VALUE / CPI_VALUES.get(year, CPI_TARGET_YEAR_VALUE)
    adjusted_avg_oop_loss = nominal_avg_oop_loss * cpi_ratio

    # Financial/legal costs
    lawyer_victims_df = df_group[df_group['CONTACT_HIRED_LAWYER'] == 1]
    total_weighted_lawyer_victims = lawyer_victims_df['FINAL_ITS_WEIGHT'].sum()
    avg_legal_cost = (total_weighted_lawyer_victims * ADJUSTED_FIXED_COSTS['legal']) / total_victims

    avg_lost_hours = weighted_average(df_group, 'HOURS_SPENT_RESOLVING_PROBLEMS', 'FINAL_ITS_WEIGHT')
    avg_time_cost = avg_lost_hours * ADJUSTED_HOURLY_WAGES.get(year, 0)

    # Health costs
    medical_victims = df_group[df_group['HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL'] == 1]['FINAL_ITS_WEIGHT'].sum()
    therapy_victims = df_group[df_group['HELP_TYPE_COUNSELING'] == 1]['FINAL_ITS_WEIGHT'].sum()
    medication_victims = df_group[df_group['HELP_TYPE_MEDICATION'] == 1]['FINAL_ITS_WEIGHT'].sum()

    total_health_cost = (medical_victims * ADJUSTED_FIXED_COSTS['medical']) + \
                        (therapy_victims * ADJUSTED_FIXED_COSTS['therapy']) + \
                        (medication_victims * ADJUSTED_FIXED_COSTS['medication'])
    avg_health_cost = total_health_cost / total_victims

    total_social_cost_per_victim = adjusted_avg_oop_loss + avg_legal_cost + avg_time_cost + avg_health_cost
    total_national_cost = total_social_cost_per_victim * total_victims

    return {
        'Year': year,
        'Group': group_name,
        'Total Weighted Victims': total_victims,
        'Avg. Out-of-Pocket Loss ($)': adjusted_avg_oop_loss,
        'Avg. Legal Cost ($)': avg_legal_cost,
        'Avg. Lost Time Cost ($)': avg_time_cost,
        'Avg. Healthcare Cost ($)': avg_health_cost,
        'Total Social Cost per Victim ($)': total_social_cost_per_victim,
        'Total National Social Cost ($)': total_national_cost
    }


def main():
    print("\n--- Part 2: Social Cost by Breach Notification Status ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    output_folder = os.path.join(root_dir, 'data/processed/part2/breach_cost_analysis')
    harmonized_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    breach_comparison_output_csv = os.path.join(output_folder, 'social_cost_breach_vs_nonbreach.csv')
    ssn_comparison_output_csv = os.path.join(output_folder, 'social_cost_ssn_vs_nonssn.csv')

    os.makedirs(output_folder, exist_ok=True)
    try:
        df = pd.read_parquet(harmonized_data_path)
        print(f"Loaded data from {harmonized_data_path}")
    except FileNotFoundError:
        print(f"Error: {harmonized_data_path} not found. Run scripts/part1/harmonize_its.py first.")
        return

    numeric_cols = [
        'year', 'FINAL_ITS_WEIGHT', 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT',
        'CONTACT_HIRED_LAWYER', 'HOURS_SPENT_RESOLVING_PROBLEMS',
        'HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL', 'HELP_TYPE_COUNSELING',
        'HELP_TYPE_MEDICATION', 'NOTIFIED_OF_DATA_BREACH', 'BREACH_INCLUDED_SSN'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column {col} not found. It will be treated as zero.")
            df[col] = 0

    years = sorted(df['year'].unique())

    print("\nAnalysis 1: breach vs. non-breach victims")
    breach_results = []
    for year in years:
        year_df = df[df['year'] == year]

        breach_victims = year_df[year_df['NOTIFIED_OF_DATA_BREACH'] == 1]
        non_breach_victims = year_df[year_df['NOTIFIED_OF_DATA_BREACH'] == 2]

        breach_cost = calculate_social_cost_for_group(breach_victims, year, "Breach Victim")
        if breach_cost:
            breach_results.append(breach_cost)

        non_breach_cost = calculate_social_cost_for_group(non_breach_victims, year, "Non-Breach Victim")
        if non_breach_cost:
            breach_results.append(non_breach_cost)

    breach_df = pd.DataFrame(breach_results)

    print("\nAnalysis 2: SSN vs. non-SSN breach victims")
    ssn_results = []
    for year in years:
        year_df = df[(df['year'] == year) & (df['NOTIFIED_OF_DATA_BREACH'] == 1)]

        ssn_victims = year_df[year_df['BREACH_INCLUDED_SSN'] == 1]
        non_ssn_victims = year_df[year_df['BREACH_INCLUDED_SSN'] == 2]

        ssn_cost = calculate_social_cost_for_group(ssn_victims, year, "SSN Breach")
        if ssn_cost:
            ssn_results.append(ssn_cost)

        non_ssn_cost = calculate_social_cost_for_group(non_ssn_victims, year, "Non-SSN Breach")
        if non_ssn_cost:
            ssn_results.append(non_ssn_cost)

    ssn_df = pd.DataFrame(ssn_results)

    print("\nComparative analysis results")

    if not breach_df.empty:
        print("Social cost, breach victims vs. non-breach victims (2021 dollars):")
        breach_pivot = breach_df.pivot(index='Year', columns='Group', values='Total Social Cost per Victim ($)')
        print(breach_pivot.to_string(float_format='${:,.2f}'))
        breach_df.to_csv(breach_comparison_output_csv, index=False, float_format='%.2f')
        print(f"\nData saved to {breach_comparison_output_csv}")
    else:
        print("No data available for breach vs. non-breach comparison.")

    if not ssn_df.empty:
        print("\n\nSocial cost, SSN breach victims vs. non-SSN breach victims (2021 dollars):")
        ssn_pivot = ssn_df.pivot(index='Year', columns='Group', values='Total Social Cost per Victim ($)')
        print(ssn_pivot.to_string(float_format='${:,.2f}'))
        ssn_df.to_csv(ssn_comparison_output_csv, index=False, float_format='%.2f')
        print(f"\nData saved to {ssn_comparison_output_csv}")
    else:
        print("\nNo data available for SSN breach vs. non-SSN breach comparison.")


if __name__ == '__main__':
    main()
