import pandas as pd
import numpy as np
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part4/apply_saturation_model.py

"""
Part 4: Dynamic Saturation & Flow Modeling
Converts raw breach counts into "Newly Unique Records" (c_t) by modeling 
population saturation (mu) and identity density (gamma) over time (2008-2021).
"""

def main():
    root_dir = os.getcwd()
    
    # --- 1. PATH CONFIGURATION ---
    input_path = os.path.join(root_dir, 'data/processed/part3/prc_hhs_integrated.csv')
    pop_path = os.path.join(root_dir, 'data/raw/external/CNP16OV.csv')
    
    # Updated Descriptive Output Filename
    output_dir = os.path.join(root_dir, 'data/processed/part4')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'PRC_Data_After_Saturation_Model.csv')

    if not os.path.exists(input_path):
        print(f"ERROR: Missing input data at {input_path}")
        return

    # --- 2. LOAD & PREP DATA ---
    df = pd.read_csv(input_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'])
    df['Month'] = df['reported_date'].dt.to_period('M')
    df['Year'] = df['reported_date'].dt.year

    # Load Real Population Data
    if os.path.exists(pop_path):
        pop_df = pd.read_csv(pop_path)
        # Handle FRED date format
        pop_df['observation_date'] = pd.to_datetime(pop_df['observation_date'])
        pop_df['Year'] = pop_df['observation_date'].dt.year
        ANNUAL_POP_16 = pop_df.groupby('Year')['CNP16OV'].mean().to_dict()
    else:
        print("Warning: Population file not found at data/raw/external/CNP16OV.csv")
        ANNUAL_POP_16 = {y: 260000 for y in range(2008, 2026)}

    # --- 3. PARAMETERS (YOUR LOCAL CONSTANTS) ---
    BETA = 141.94         # National record multiplier from your Part 3 regression
    THETA = 1.75          # Unreported breach scaling factor (θ)
    ANNUAL_NU = 1549465   # Your Refined Undisclosed Baseline (n_u)
    GAMMA_BASE = 0.8     # Base identity density (gamma_0)
    OVERLAP_DISCOUNT = 0.8 # Redundancy discount for mega-breaches

    mega_shocks = {
        '2008-05': 12500000, '2009-01': 130000000, '2011-04': 77000000,
        '2013-12': 40000000, '2014-09': 56000000, '2017-09': 143000000,
    }

    # --- 4. INCIDENT-PROPORTIONAL IMPUTATION ---
    unknown_counts_per_year = df[df['total_affected'] == 0].groupby('Year').size().to_dict()

    def impute_records(row):
        if row['total_affected'] > 0:
            return row['total_affected']
        year = row['Year']
        n_unknown = unknown_counts_per_year.get(year, 1)
        return ANNUAL_NU / n_unknown

    df['total_affected_imputed'] = df.apply(impute_records, axis=1)

    # --- 5. THE SATURATION LOOP ---
    C_prev = 0 
    annual_dynamic_params = {}
    diagnostics = []

    print("\n" + "="*95)
    print("RUNNING DYNAMIC SATURATION MODEL (Indexing: t=1 is 2008)")
    print(f"{'Year (t)':<8} | {'Pop (N_t)':<13} | {'Prev Stock (C_t-1)':<18} | {'Sat (mu_t-1)':<12} | {'Gamma_t':<8}")
    print("-" * 95)

    for year in sorted(df['Year'].unique()):
        if pd.isna(year) or year < 2008 or year > 2022: continue
        
        pop_16 = ANNUAL_POP_16.get(year, 260000) * 1000 
        
        # 1. mu_{t-1} (Start of year)
        mu_prev_raw = C_prev / (pop_16 * THETA)
        mu_prev_capped = min(mu_prev_raw, 0.95)
        
        # 2. gamma_t
        gamma_t = GAMMA_BASE * (1 - mu_prev_capped)
        annual_dynamic_params[year] = gamma_t
        
        # 3. flow c_t
        yearly_df = df[df['Year'] == year]
        r_t = yearly_df['total_affected_imputed'].sum() * THETA
        c_t = r_t * gamma_t
        
        # 4. Stock C_t (End of year)
        C_t = C_prev + c_t
        
        # 5. mu_t (End of year)
        mu_curr_raw = C_t / (pop_16 * THETA)
        mu_curr_capped = min(mu_curr_raw, 0.95)
        
        print(f"{year:<8.0f} | {pop_16:>13,.0f} | {C_prev:>18,.0f} | {mu_prev_capped:>12.2%} | {gamma_t:>8.3f}")
        
        diagnostics.append({
            'year': year,
            'N_t': pop_16,
            'r_t': r_t,
            'gamma_t': gamma_t,
            'c_t': c_t,
            'mu_prev_raw': mu_prev_raw,
            'mu_prev_capped': mu_prev_capped,
            'mu_curr_raw': mu_curr_raw,
            'mu_curr_capped': mu_curr_capped,
            'C_t_end': C_t
        })

        C_prev = C_t

    df['Dynamic_Gamma'] = df['Year'].map(annual_dynamic_params)

    # --- STEP-BY-STEP VARIABLE PRINTOUT ---
    print("\n" + "="*165)
    print("STEP-BY-STEP MODEL PRINTOUTS")
    print(f"{'Year':<5} | {'r_t':<12} | {'mu_t-1(Raw)':<11} | {'mu_t-1(Cap)':<11} | {'gamma_t':<7} | {'c_t (New)':<12} | {'C_t (End)':<12} | {'mu_t(Raw)':<10} | {'mu_t(Cap)'}")
    print("-" * 165)
    for d in diagnostics:
        print(f"{d['year']:<5.0f} | {d['r_t']:>12,.0f} | {d['mu_prev_raw']:>11.2%} | {d['mu_prev_capped']:>11.2%} | {d['gamma_t']:>7.4f} | {d['c_t']:>12,.0f} | {d['C_t_end']:>12,.0f} | {d['mu_curr_raw']:>10.2%} | {d['mu_curr_capped']:>10.2%}")

    # --- 6. MONTHLY AGGREGATION & FINAL SCALING ---
    monthly_summary = df.groupby(['Month', 'Dynamic_Gamma'])['total_affected_imputed'].sum().reset_index()
    monthly_summary['Month_Str'] = monthly_summary['Month'].dt.strftime('%Y-%m')

    def finalize_um(row):
        total = row['total_affected_imputed']
        if row['Month_Str'] in mega_shocks:
            total = max(total, mega_shocks[row['Month_Str']])
        
        total_scaled = total * THETA
        identity_units = total_scaled * row['Dynamic_Gamma']
        
        if total_scaled > 5000000:
            identity_units *= OVERLAP_DISCOUNT
            
        return identity_units

    monthly_summary['Um_Unique_Individuals'] = monthly_summary.apply(finalize_um, axis=1)

    # --- 7. SAVE ---
    monthly_summary.to_csv(output_file, index=False)
    print("\n" + "="*80)
    print(f"{os.path.basename(output_file)} saved.")

if __name__ == "__main__":
    main()