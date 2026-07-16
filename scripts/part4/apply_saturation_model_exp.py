import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# To run this script: python scripts/part4/apply_saturation_model_exp.py

def main():
    root_dir = os.getcwd()
    
    # --- 1. PATH CONFIGURATION ---
    input_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    pop_path = os.path.join(root_dir, 'data/raw/external/CNP16OV.csv')
    
    output_dir = os.path.join(root_dir, 'data/processed/part4')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'PRC_Data_After_Saturation_Model_Exp.csv')

    if not os.path.exists(input_path):
        print(f"ERROR: Missing input data at {input_path}")
        return

    # --- 2. LOAD DATA ---
    print(f"Loading Golden Record from: {input_path}")
    df = pd.read_csv(input_path)
    
    # --- FIX: Handle mixed date formats (YYYY-MM-DD vs YYYY-MM-DD HH:MM:SS) ---
    df['reported_date'] = pd.to_datetime(df['reported_date'], format='mixed', errors='coerce')
    
    # Drop rows where date parsing failed completely (should be 0 or very few)
    if df['reported_date'].isna().sum() > 0:
        print(f"Warning: Dropping {df['reported_date'].isna().sum()} rows with unparseable dates.")
        df = df.dropna(subset=['reported_date'])

    df['Month'] = df['reported_date'].dt.to_period('M')
    df['Year'] = df['reported_date'].dt.year

    # Load Population Data
    if os.path.exists(pop_path):
        pop_df = pd.read_csv(pop_path)
        pop_df['observation_date'] = pd.to_datetime(pop_df['observation_date'])
        pop_df['Year'] = pop_df['observation_date'].dt.year
        ANNUAL_POP_16 = pop_df.groupby('Year')['CNP16OV'].mean().to_dict()
    else:
        print("Warning: Population file not found. Using default constant.")
        ANNUAL_POP_16 = {y: 260000 for y in range(2000, 2030)}

    # --- 3. PARAMETERS ---
    THETA = 1.75           
    GAMMA_BASE = 0.8       
    OVERLAP_DISCOUNT = 0.8 
    
    # --- SATURATION SETTINGS ---
    IDENTITIES_PER_PERSON = 15

    mega_shocks = {
        '2008-05': 12500000, '2009-01': 130000000, '2011-04': 77000000,
        '2013-12': 40000000, '2014-09': 56000000, '2017-09': 143000000,
    }

    # --- 4. IMPUTATION CHECK ---
    if 'total_affected_imputed' not in df.columns:
        print("WARNING: 'total_affected_imputed' column missing. Using raw 'total_affected'.")
        df['total_affected_imputed'] = df['total_affected'].fillna(0)
    else:
        print("Using pre-imputed 'total_affected_imputed' column.")

    # --- 5. SATURATION LOOP ---
    C_prev = 0 
    annual_dynamic_params = {}
    diagnostics = []

    print("\n" + "="*120)
    print(f"RUNNING SATURATION + HARDENING MODEL")
    print(f"Identities/Person: {IDENTITIES_PER_PERSON}")
    print(f"{'Year':<6} | {'Pop (N_t)':<13} | {'Capacity':<15} | {'Sat (Start)':<12} | {'Eff. Gamma':<10} | {'Utility':<10}")
    print("-" * 120)

    # Filter for valid years
    valid_years = sorted([y for y in df['Year'].unique() if pd.notna(y) and 2008 <= y <= 2022])

    for year in valid_years:
        
        # 1. Standard Saturation Math
        pop_16 = ANNUAL_POP_16.get(year, 260000) * 1000 
        total_capacity = pop_16 * IDENTITIES_PER_PERSON
        mu_start = C_prev / total_capacity
        
        yearly_df = df[df['Year'] == year]
        r_t = yearly_df['total_affected_imputed'].sum() * THETA
        
        if total_capacity > 0:
            K = (r_t * GAMMA_BASE) / total_capacity
        else:
            K = 0

        gap_start = 1.0 - mu_start
        decay_factor = np.exp(-K)
        mu_end = 1.0 - (gap_start * decay_factor)
        c_t = (mu_end - mu_start) * total_capacity
        C_prev += c_t
        
        # 2. Calculate Base Gamma (Saturation Only)
        if r_t > 0:
            gamma_saturation = c_t / r_t
        else:
            gamma_saturation = GAMMA_BASE * gap_start
        
        utility_factor = 1.0
            
        gamma_final = gamma_saturation * utility_factor

        annual_dynamic_params[year] = gamma_final
        
        print(f"{year:<6.0f} | {pop_16:>13,.0f} | {total_capacity:>15,.0f} | {mu_start:>12.2%} | {gamma_final:>10.4f} | {utility_factor:>10.4f}")
        
        diagnostics.append({'year': year, 'mu_end': mu_end})

    df['Dynamic_Gamma'] = df['Year'].map(annual_dynamic_params)

    # --- 6. SAVE CSV ---
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
    monthly_summary.to_csv(output_file, index=False)
    print(f"\nSaved CSV to {output_file}")
    
    # --- 7. PLOT ---
    years = [d['year'] for d in diagnostics]
    mus = [d['mu_end'] for d in diagnostics]
    plt.figure(figsize=(10, 6))
    plt.plot(years, mus, marker='o', color='teal', linewidth=2, label=f'T={IDENTITIES_PER_PERSON}')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title("Identity Saturation Over Time")
    plt.xlabel("Year")
    plt.ylabel("Saturation Level (Mu)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.show()

if __name__ == "__main__":
    main()