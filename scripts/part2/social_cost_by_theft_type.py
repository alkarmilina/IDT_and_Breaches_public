import pandas as pd
import numpy as np
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/social_cost_by_theft_type.py

"""
Analysis: Social Cost by Identity Theft Type (Inclusive Filtering)

This script calculates the complete social cost profile for each of the five
NCVS-ITS theft type categories using inclusive filtering: a victim is included
in a type's group if they experienced that type, regardless of whether they also
experienced other types simultaneously. This mirrors the SSN vs. non-SSN
framing in social_cost_breach_vs_nonbreach.py.

Because theft types are not mutually exclusive, a single victim may appear in
more than one group. The 'Multiple Types' row isolates victims who experienced
more than one type to provide context on the overlap share.

Key Operations:
    1. For each year × each of the 5 theft type columns, filter to victims
       where that column == 1 and compute the full 4-component social cost.
    2. Adds a 'Multiple Types' row (victims with type_count > 1) for context.
    3. Reports overlap share (% of year's victims in multiple types) so readers
       understand that group totals do not sum to the all-victim total.
    4. National cost estimates use FINAL_ITS_WEIGHT throughout.

Setup/Inputs:
    - its_victims.parquet: The standardized victim-level dataset from Part 1.
    - Economic constants match those used in social_cost_breach_vs_nonbreach.py.

Outputs:
    - data/processed/part2/breach_cost_analysis/social_cost_by_theft_type.csv:
      Per-victim and national cost breakdown by theft type and year.
"""

# --- PART 1: CONFIGURATION ---

OUTPUT_FOLDER = 'data/processed/part2/breach_cost_analysis'
HARMONIZED_DATA_PATH = 'data/processed/part1/its_victims.parquet'
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, 'social_cost_by_theft_type.csv')

ADJUSTED_HOURLY_WAGES = {
    2008: 26.67, 2012: 27.03, 2014: 27.73,
    2016: 28.64, 2018: 28.83, 2021: 29.92
}
ADJUSTED_FIXED_COSTS = {
    "legal": 444.65, "medical": 86.38,
    "therapy": 86.38, "medication": 53.96
}
CPI_VALUES = {
    2008: 215.297, 2012: 233.165, 2014: 236.736,
    2016: 240.007, 2018: 251.107, 2021: 270.970
}
CPI_TARGET_YEAR_VALUE = 270.970

THEFT_TYPES = {
    'Existing Bank Account Misuse':       'EXISTING_BANK_ACCT_MISUSE_12MO',
    'Existing Credit Card Misuse':        'EXISTING_CC_MISUSE_12MO',
    'Other Existing Account Misuse':      'EXISTING_OTHER_ACCT_MISUSE_12MO',
    'New Account Fraud':                  'NEW_ACCT_OPENED_12MO',
    'Other Misuse of Personal Info':      'OTHER_FRAUDULENT_PURPOSE_12MO',
}

# --- PART 2: CALCULATION ENGINE ---

def weighted_average(df, value_col, weight_col):
    good_data = df.dropna(subset=[value_col, weight_col])
    if good_data.empty:
        return 0
    return np.average(good_data[value_col], weights=good_data[weight_col])


def calculate_social_cost_for_group(df_group, year, group_name, total_year_victims):
    """
    Calculates the complete social cost for a victim subset.
    total_year_victims is the weighted count for the full year, used to
    compute the overlap share for multi-type reporting.
    """
    total_victims = df_group['FINAL_ITS_WEIGHT'].sum()
    if total_victims == 0:
        return None

    # Dimension 1: Out-of-Pocket Loss (CPI-adjusted)
    nominal_avg_oop = weighted_average(df_group, 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT', 'FINAL_ITS_WEIGHT')
    cpi_ratio = CPI_TARGET_YEAR_VALUE / CPI_VALUES.get(year, CPI_TARGET_YEAR_VALUE)
    adjusted_avg_oop = nominal_avg_oop * cpi_ratio

    # Dimension 2: Legal Cost
    lawyer_victims = df_group[df_group['CONTACT_HIRED_LAWYER'] == 1]['FINAL_ITS_WEIGHT'].sum()
    avg_legal_cost = (lawyer_victims * ADJUSTED_FIXED_COSTS['legal']) / total_victims

    # Dimension 3: Lost Time Cost
    avg_lost_hours = weighted_average(df_group, 'HOURS_SPENT_RESOLVING_PROBLEMS', 'FINAL_ITS_WEIGHT')
    avg_time_cost = avg_lost_hours * ADJUSTED_HOURLY_WAGES.get(year, 0)

    # Dimension 4: Healthcare Cost
    medical_victims  = df_group[df_group['HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL'] == 1]['FINAL_ITS_WEIGHT'].sum()
    therapy_victims  = df_group[df_group['HELP_TYPE_COUNSELING'] == 1]['FINAL_ITS_WEIGHT'].sum()
    medication_victims = df_group[df_group['HELP_TYPE_MEDICATION'] == 1]['FINAL_ITS_WEIGHT'].sum()
    avg_health_cost = (
        (medical_victims  * ADJUSTED_FIXED_COSTS['medical']) +
        (therapy_victims  * ADJUSTED_FIXED_COSTS['therapy']) +
        (medication_victims * ADJUSTED_FIXED_COSTS['medication'])
    ) / total_victims

    total_cost_per_victim = adjusted_avg_oop + avg_legal_cost + avg_time_cost + avg_health_cost
    total_national_cost = total_cost_per_victim * total_victims

    overlap_share = (total_victims / total_year_victims * 100) if total_year_victims > 0 else None

    return {
        'Year':                            year,
        'Theft Type':                      group_name,
        'Total Weighted Victims':          total_victims,
        'Share of Year Victims (%)':       overlap_share,
        'Avg. Out-of-Pocket Loss ($)':     adjusted_avg_oop,
        'Avg. Legal Cost ($)':             avg_legal_cost,
        'Avg. Lost Time Cost ($)':         avg_time_cost,
        'Avg. Healthcare Cost ($)':        avg_health_cost,
        'Total Social Cost per Victim ($)': total_cost_per_victim,
        'Total National Social Cost ($)':  total_national_cost,
    }


# --- PART 3: MAIN EXECUTION ---

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Loading harmonized data from '{HARMONIZED_DATA_PATH}'...")
    try:
        df = pd.read_parquet(HARMONIZED_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{HARMONIZED_DATA_PATH}'.")
        return
    print("Data loaded successfully.")

    # Coerce all relevant columns to numeric
    numeric_cols = [
        'year', 'FINAL_ITS_WEIGHT', 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT',
        'CONTACT_HIRED_LAWYER', 'HOURS_SPENT_RESOLVING_PROBLEMS',
        'HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL', 'HELP_TYPE_COUNSELING',
        'HELP_TYPE_MEDICATION',
        *THEFT_TYPES.values(),
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in data — treating as zero.")
            df[col] = 0

    # Pre-compute multi-type flag (used per year below)
    theft_vars = list(THEFT_TYPES.values())
    df['_type_count'] = (df[theft_vars] == 1).sum(axis=1)

    years = sorted(df['year'].unique())
    results = []

    for year in years:
        year_df = df[df['year'] == year]
        total_year_victims = year_df['FINAL_ITS_WEIGHT'].sum()

        print(f"\n--- Year {year} (N={total_year_victims:,.0f} weighted victims) ---")

        # Inclusive filter: one group per theft type
        for display_name, col in THEFT_TYPES.items():
            group_df = year_df[year_df[col] == 1]
            row = calculate_social_cost_for_group(group_df, year, display_name, total_year_victims)
            if row:
                results.append(row)
                print(f"  {display_name}: {row['Total Weighted Victims']:>12,.0f} victims  "
                      f"({row['Share of Year Victims (%)']:.1f}%)  "
                      f"-> ${row['Total Social Cost per Victim ($)']:>8,.2f}/victim")

        # Multi-type victims (for overlap disclosure)
        multi_df = year_df[year_df['_type_count'] > 1]
        multi_row = calculate_social_cost_for_group(multi_df, year, 'Multiple Types', total_year_victims)
        if multi_row:
            results.append(multi_row)
            print(f"  Multiple Types:            {multi_row['Total Weighted Victims']:>12,.0f} victims  "
                  f"({multi_row['Share of Year Victims (%)']:.1f}% overlap)  "
                  f"-> ${multi_row['Total Social Cost per Victim ($)']:>8,.2f}/victim")

    # --- Output ---
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, float_format='%.2f')
    print(f"\nResults saved to '{OUTPUT_CSV}'")

    # Summary pivot: Total Social Cost per Victim by theft type × year
    print("\n\n--- SUMMARY: Total Social Cost per Victim ($) by Theft Type and Year ---")
    pivot = out_df.pivot(index='Theft Type', columns='Year', values='Total Social Cost per Victim ($)')
    print(pivot.to_string(float_format='${:,.2f}'))


if __name__ == '__main__':
    main()
