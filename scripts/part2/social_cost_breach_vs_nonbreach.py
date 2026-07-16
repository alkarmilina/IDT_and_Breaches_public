import pandas as pd
import numpy as np
import os 

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/social_cost_breach_vs_nonbreach.py

"""
Analysis: Comparative Social Cost Modeling (Breach & SSN Impact)

This script implements the three-dimensional social cost model to quantify the 
total economic burden of identity theft. It performs comparative analysis to 
determine how breach notification and SSN exposure influence the severity of 
financial, professional, and health-related harms.

Key Operations:
    1. Monetization Engine: Sums four cost pillars:
        - Adjusted Out-of-Pocket (OOP) Loss (CPI-U adjusted to 2021 dollars).
        - Legal Costs (Fixed average per hiring victim).
        - Opportunity Cost (Lost hours x year-specific average private wage).
        - Healthcare Costs (Monetized medical, therapy, and medication visits).
    2. Group Comparison: Segregates the ITS victim data into 'Breach vs. Non-Breach' 
       and 'SSN vs. Non-SSN' cohorts for longitudinal comparison.
    3. National Scaling: Uses the FINAL_ITS_WEIGHT to scale per-victim costs to 
       national social burden estimates.

Setup/Inputs:
    - its_victims.parquet: The standardized victim-level dataset.
    - Economic Constants: Hardcoded hourly wages (Table 14) and fixed service 
      costs (Table 15) from the paper.

Outputs:
    - data/processed/part2/breach_cost_analysis/social_cost_breach_vs_nonbreach.csv: 
      Supports Section 6.2 and Appendix Table 11.
    - data/processed/part2/breach_cost_analysis/social_cost_ssn_vs_nonssn.csv: 
      Supports Section 6.2 and Appendix Table 12.
"""

# --- PART 1: CONFIGURATION AND SETUP ---
# This section contains all predefined data and economic assumptions for the model.

# NEW CONFIGURATION FOR FOLDER STRUCTURE
OUTPUT_FOLDER = 'data/processed/part2/breach_cost_analysis'

# 1a. Input/Output Paths
# Pointing to the specific harmonized victim file from Part 1
HARMONIZED_DATA_PATH = 'data/processed/part1/its_victims.parquet'

BREACH_COMPARISON_OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, 'social_cost_breach_vs_nonbreach.csv')
SSN_COMPARISON_OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, 'social_cost_ssn_vs_nonssn.csv')

# 1b. Economic Constants (All values adjusted to 2021 dollars)
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

# --- PART 2: THE REUSABLE CALCULATION ENGINE ---

def weighted_average(df, value_col, weight_col):
    """Helper function to calculate a weighted average, handling NaNs."""
    good_data = df.dropna(subset=[value_col, weight_col])
    if good_data.empty:
        return 0
    return np.average(good_data[value_col], weights=good_data[weight_col])

def calculate_social_cost_for_group(df_group, year, group_name):
    """
    Calculates the complete social cost for a specific dataframe subset (a group).
    This function is the core of our comparative analysis.
    """
    total_victims = df_group['FINAL_ITS_WEIGHT'].sum()
    if total_victims == 0:
        return None

    # --- Dimension 1: Direct Financial & Professional Costs ---
    nominal_avg_oop_loss = weighted_average(df_group, 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT', 'FINAL_ITS_WEIGHT')
    cpi_ratio = CPI_TARGET_YEAR_VALUE / CPI_VALUES.get(year, CPI_TARGET_YEAR_VALUE)
    adjusted_avg_oop_loss = nominal_avg_oop_loss * cpi_ratio
    
    lawyer_victims_df = df_group[df_group['CONTACT_HIRED_LAWYER'] == 1]
    total_weighted_lawyer_victims = lawyer_victims_df['FINAL_ITS_WEIGHT'].sum()
    avg_legal_cost = (total_weighted_lawyer_victims * ADJUSTED_FIXED_COSTS['legal']) / total_victims

    # --- Dimension 2: Lost Time & Productivity Cost ---
    avg_lost_hours = weighted_average(df_group, 'HOURS_SPENT_RESOLVING_PROBLEMS', 'FINAL_ITS_WEIGHT')
    avg_time_cost = avg_lost_hours * ADJUSTED_HOURLY_WAGES.get(year, 0)

    # --- Dimension 3: Health & Wellbeing Costs ---
    medical_victims = df_group[df_group['HELP_TYPE_VISITED_MEDICAL_PROFESSIONAL'] == 1]['FINAL_ITS_WEIGHT'].sum()
    therapy_victims = df_group[df_group['HELP_TYPE_COUNSELING'] == 1]['FINAL_ITS_WEIGHT'].sum()
    medication_victims = df_group[df_group['HELP_TYPE_MEDICATION'] == 1]['FINAL_ITS_WEIGHT'].sum()
    
    total_health_cost = (medical_victims * ADJUSTED_FIXED_COSTS['medical']) + \
                        (therapy_victims * ADJUSTED_FIXED_COSTS['therapy']) + \
                        (medication_victims * ADJUSTED_FIXED_COSTS['medication'])
    avg_health_cost = total_health_cost / total_victims

    # --- Final Calculations ---
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

# --- PART 3: MAIN EXECUTION AND ANALYSIS ---

def main():
    """
    Main function to run the comparative social cost analysis.
    """
    # Ensure the output folder exists before proceeding
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Output folder '{OUTPUT_FOLDER}' ensured to exist.")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    print(f"Loading harmonized data from '{HARMONIZED_DATA_PATH}'...")
    try:
        df = pd.read_parquet(HARMONIZED_DATA_PATH)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{HARMONIZED_DATA_PATH}'.")
        return

    # Convert relevant columns to numeric, handling potential errors
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
            print(f"Warning: Column '{col}' not found. It will be treated as zero.")
            df[col] = 0

    years = sorted(df['year'].unique())

    # --- ** DIAGNOSTIC CHECK ** ---
    print("\n--- Diagnostic Check for Lawyer Hires in Breach Victims ---")
    for year in years:
        breach_year_df = df[(df['year'] == year) & (df['NOTIFIED_OF_DATA_BREACH'] == 1)]
        if not breach_year_df.empty:
            ssn_victims_lawyer_count = len(breach_year_df[(breach_year_df['BREACH_INCLUDED_SSN'] == 1) & (breach_year_df['CONTACT_HIRED_LAWYER'] == 1)])
            non_ssn_victims_lawyer_count = len(breach_year_df[(breach_year_df['BREACH_INCLUDED_SSN'] == 2) & (breach_year_df['CONTACT_HIRED_LAWYER'] == 1)])
            print(f"  Year {year}:")
            print(f"    -> SSN Breach Victims who hired a lawyer (raw count): {ssn_victims_lawyer_count}")
            print(f"    -> Non-SSN Breach Victims who hired a lawyer (raw count): {non_ssn_victims_lawyer_count}")
    print("-----------------------------------------------------------\n")
    
    # --- Analysis 1: Breach vs. Non-Breach ---
    print("--- Running Analysis 1: Breach vs. Non-Breach Victims ---")
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
    
    # --- Analysis 2: SSN Breach vs. Non-SSN Breach ---
    print("\n--- Running Analysis 2: SSN vs. Non-SSN Breach Victims ---")
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

    # --- PART 4: OUTPUT ---
    print("\n\n--- COMPARATIVE ANALYSIS RESULTS ---\n")
    
    if not breach_df.empty:
        print("Table 1: Social Cost Comparison - Breach vs. Non-Breach Victims (All costs in 2021 Dollars)")
        breach_pivot = breach_df.pivot(index='Year', columns='Group', values='Total Social Cost per Victim ($)')
        print(breach_pivot.to_string(float_format='${:,.2f}'))
        breach_df.to_csv(BREACH_COMPARISON_OUTPUT_CSV, index=False, float_format='%.2f')
        print(f"\nFull results for breach comparison saved to '{BREACH_COMPARISON_OUTPUT_CSV}'")
    else:
        print("No data available for Breach vs. Non-Breach comparison.")

    if not ssn_df.empty:
        print("\n\nTable 2: Social Cost Comparison - SSN vs. Non-SSN Breach Victims (All costs in 2021 Dollars)")
        ssn_pivot = ssn_df.pivot(index='Year', columns='Group', values='Total Social Cost per Victim ($)')
        print(ssn_pivot.to_string(float_format='${:,.2f}'))
        ssn_df.to_csv(SSN_COMPARISON_OUTPUT_CSV, index=False, float_format='%.2f')
        print(f"\nFull results for SSN comparison saved to '{SSN_COMPARISON_OUTPUT_CSV}'")
    else:
        print("\nNo data available for SSN Breach vs. Non-SSN Breach comparison.")

if __name__ == '__main__':
    main()