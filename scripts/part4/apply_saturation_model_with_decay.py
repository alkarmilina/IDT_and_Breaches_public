import pandas as pd
import numpy as np
import os

# To run this script:
# python scripts/part4/apply_saturation_model_with_decay.py

"""
Part 4: Reservoir Risk Modeling (OPTIMIZED)
Converts raw breach counts into "Active Risk Units" using the optimized 
Reservoir Model parameters (Lambda=0.60, Beta=61k) learned from Part 5.
"""

def main():
    root_dir = os.getcwd()
    input_path = os.path.join(root_dir, 'data/processed/part3/prc_hhs_integrated.csv')
    output_dir = os.path.join(root_dir, 'data/processed/part4')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'PRC_Data_After_Saturation_Model_With_Decay.csv')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # --- 1. LOAD DATA ---
    df = pd.read_csv(input_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'])
    df['Month'] = df['reported_date'].dt.to_period('M')
    df['Year'] = df['reported_date'].dt.year

    # --- 2. CONFIGURATION (Optimized Parameters) ---
    # Derived from 'scripts/part5/learn_reservoir_model.py'
    LAMBDA_DECAY = 0.5983       # 59.8% of risk drains every month
    BETA_SENSITIVITY = 61631.22 # Sensitivity to log-scale input
    BASELINE_C = 100000.00      # Background risk constant
    
    # Data Cleaning Constants
    ANNUAL_NU = 1549465         # Imputation for unknown breach sizes (from Paper)
    
    # Known Mega-Breach Corrections (Ensuring we capture the true scale of inputs)
    mega_shocks = {
        '2008-05': 12500000, 
        '2009-01': 130000000, 
        '2011-04': 77000000,
        '2013-12': 40000000, 
        '2014-09': 56000000, 
        '2017-09': 143000000,
    }

    # --- 3. IMPUTE MISSING VALUES ---
    # Impute 0-count breaches
    unknown_counts_per_year = df[df['total_affected'] == 0].groupby('Year').size().to_dict()
    
    def impute_records(row):
        if row['total_affected'] > 0: 
            return row['total_affected']
        # Distribute the annual unknown baseline across the unknown incidents
        year = row['Year']
        n_unknown = unknown_counts_per_year.get(year, 1)
        return ANNUAL_NU / n_unknown

    df['total_affected_imputed'] = df.apply(impute_records, axis=1)

    # --- 4. AGGREGATE TO MONTHLY TIMELINE ---
    # We need a continuous timeline to run the decay loop properly
    full_idx = pd.period_range(start='2008-01', end='2022-01', freq='M')
    timeline_df = pd.DataFrame({'Month': full_idx})
    
    # Sum up the imputed records by month
    monthly_sums = df.groupby('Month')['total_affected_imputed'].sum().reset_index()
    merged = pd.merge(timeline_df, monthly_sums, on='Month', how='left').fillna(0)
    
    # Apply Mega Shocks corrections (Ensure manual overrides are respected)
    merged['Month_Str'] = merged['Month'].dt.strftime('%Y-%m')
    
    def apply_shocks(row):
        manual_val = mega_shocks.get(row['Month_Str'], 0)
        return max(row['total_affected_imputed'], manual_val)
    
    merged['Final_Input_Records'] = merged.apply(apply_shocks, axis=1)

    # --- 5. RUN RESERVOIR MODEL ---
    print("\n" + "="*60)
    print(f"RUNNING RESERVOIR MODEL (Lambda={LAMBDA_DECAY:.4f})")
    print("="*60)
    
    risk_pool = []
    current_stock = 0.0
    
    # We use log1p to dampen the massive orders-of-magnitude differences
    # (e.g. 100M records vs 10k records)
    log_inputs = np.log1p(merged['Final_Input_Records'].values)
    
    for t, inflow_log in enumerate(log_inputs):
        # 1. New Risk enters (scaled by Beta)
        new_risk = inflow_log * BETA_SENSITIVITY
        
        # 2. Old Risk drains (decay) + New Risk adds
        current_stock = current_stock * (1 - LAMBDA_DECAY) + new_risk
        
        # 3. Add Baseline
        total_active_risk = current_stock + BASELINE_C
        risk_pool.append(total_active_risk)

    merged['Um_Unique_Individuals'] = risk_pool # Naming column to match plotting script expectations
    
    # --- 6. DIAGNOSTICS & SAVING ---
    # Print a few key dates to verify logic
    key_dates = ['2013-12', '2014-01', '2014-02', '2017-09', '2017-10']
    print(f"{'Month':<10} | {'Raw Input':<15} | {'Active Risk Pool (Output)':<25}")
    print("-" * 60)
    
    for _, row in merged[merged['Month_Str'].isin(key_dates)].iterrows():
        print(f"{row['Month_Str']:<10} | {row['Final_Input_Records']:>15,.0f} | {row['Um_Unique_Individuals']:>25,.0f}")

    # Save
    # We verify column names match what plotting scripts expect
    merged['Month_Str'] = merged['Month'].dt.strftime('%Y-%m')
    final_output = merged[['Month_Str', 'Final_Input_Records', 'Um_Unique_Individuals']]
    
    final_output.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Optimized model data saved to {output_file}")

if __name__ == "__main__":
    main()