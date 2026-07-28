import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part4/apply_saturation_model_exp.py

"""
Part 4: Dynamic Saturation Model

Converts the annual augmented PRC record counts into an estimated
number of unique individuals compromised each month, accounting for
market saturation, the fact that as more records are exposed over
time, a growing share of them belong to people already compromised
before. Records exposed are scaled by theta to correct for
under-reporting, and the saturation index mu tracks the share of the
potential victim pool already compromised, updated year over year so
mu asymptotically approaches 1 but never exceeds it.

Setup:
Reads the augmented PRC dataset from PRC_Maine_NH_regression.py, and
annual U.S. population (age 16+) data from the BLS. theta=1.75 and
gamma_base=0.8 follow the paper's saturation model. identities_per_person
(Y) is set to 1, following the paper's base model, where each person
has exactly one identity unit that can be compromised.

Goal:
Produce the paper's dynamic saturation model, converting the raw
exposed-record supply into an estimated unique-individual supply for
use in the breach-to-victim conversion analysis.

Outputs:
- data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv: monthly
  estimated unique individuals compromised.
- plots/part4/saturation_over_time.png/.pdf: the saturation index over
  the study period.
"""


def save_dual_formats(output_folder, base_name):
    """Saves the current matplotlib figure as both PNG and PDF under the same base file name."""
    plt.savefig(os.path.join(output_folder, f"{base_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, f"{base_name}.pdf"), bbox_inches='tight')


def main():
    print("\n--- Part 4: Dynamic Saturation Model ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    input_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    pop_path = os.path.join(root_dir, 'data/raw/external/CNP16OV.csv')

    output_dir = os.path.join(root_dir, 'data/processed/part4')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'PRC_Data_After_Saturation_Model_Exp.csv')

    plots_dir = os.path.join(root_dir, 'plots/part4')
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Error: Missing input data at {input_path}. Run scripts/part3/PRC_Maine_NH_regression.py first.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded data from {input_path}")

    df['reported_date'] = pd.to_datetime(df['reported_date'], format='mixed', errors='coerce')

    if df['reported_date'].isna().sum() > 0:
        print(f"Warning: Dropping {df['reported_date'].isna().sum()} rows with unparseable dates.")
        df = df.dropna(subset=['reported_date'])

    df['Month'] = df['reported_date'].dt.to_period('M')
    df['Year'] = df['reported_date'].dt.year

    if os.path.exists(pop_path):
        pop_df = pd.read_csv(pop_path)
        print(f"Loaded data from {pop_path}")
        pop_df['observation_date'] = pd.to_datetime(pop_df['observation_date'])
        pop_df['Year'] = pop_df['observation_date'].dt.year
        annual_pop_16 = pop_df.groupby('Year')['CNP16OV'].mean().to_dict()
    else:
        print("Warning: Population file not found. Using default constant.")
        annual_pop_16 = {y: 260000 for y in range(2000, 2030)}

    theta = 1.75
    gamma_base = 0.8
    identities_per_person = 1  # Y in the paper's model. Each person has one identity unit that can be compromised.

    if 'total_affected_imputed' not in df.columns:
        print("Warning: total_affected_imputed column missing. Using raw total_affected.")
        df['total_affected_imputed'] = df['total_affected'].fillna(0)
    else:
        print("Using pre-imputed total_affected_imputed column.")

    C_prev = 0
    annual_dynamic_params = {}
    diagnostics = []

    print("\n" + "="*100)
    print("Saturation model")
    print(f"Identities/Person: {identities_per_person}")
    print(f"{'Year':<6} | {'Pop (N_t)':<13} | {'Capacity':<15} | {'Sat (Start)':<12} | {'Eff. Gamma':<10}")
    print("-" * 100)

    valid_years = sorted([y for y in df['Year'].unique() if pd.notna(y) and 2008 <= y <= 2022])

    for year in valid_years:
        # CNP16OV is reported in thousands of persons.
        pop_16 = annual_pop_16.get(year, 260000) * 1000
        total_capacity = pop_16 * identities_per_person
        mu_start = C_prev / total_capacity

        yearly_df = df[df['Year'] == year]
        r_t = yearly_df['total_affected_imputed'].sum() * theta

        K = (r_t * gamma_base) / total_capacity if total_capacity > 0 else 0

        gap_start = 1.0 - mu_start
        decay_factor = np.exp(-K)
        mu_end = 1.0 - (gap_start * decay_factor)
        c_t = (mu_end - mu_start) * total_capacity
        C_prev += c_t

        # Effective identity density realized this year, new unique individuals per
        # theta-scaled record exposed. Falls back to the closed-form
        # gamma0 * (1 - mu_start) when no records were exposed that year, to avoid
        # dividing by zero.
        if r_t > 0:
            gamma_final = c_t / r_t
        else:
            gamma_final = gamma_base * gap_start

        annual_dynamic_params[year] = gamma_final

        print(f"{year:<6.0f} | {pop_16:>13,.0f} | {total_capacity:>15,.0f} | {mu_start:>12.2%} | {gamma_final:>10.4f}")

        diagnostics.append({'year': year, 'mu_end': mu_end})

    df['Dynamic_Gamma'] = df['Year'].map(annual_dynamic_params)

    monthly_summary = df.groupby(['Month', 'Dynamic_Gamma'])['total_affected_imputed'].sum().reset_index()
    monthly_summary['Month_Str'] = monthly_summary['Month'].dt.strftime('%Y-%m')

    def finalize_um(row):
        total_scaled = row['total_affected_imputed'] * theta
        return total_scaled * row['Dynamic_Gamma']

    monthly_summary['Um_Unique_Individuals'] = monthly_summary.apply(finalize_um, axis=1)
    monthly_summary.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")

    years = [d['year'] for d in diagnostics]
    mus = [d['mu_end'] for d in diagnostics]
    plt.figure(figsize=(10, 6))
    plt.plot(years, mus, marker='o', color='teal', linewidth=2, label=f'Y={identities_per_person}')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title("Identity Saturation Over Time")
    plt.xlabel("Year")
    plt.ylabel("Saturation Level (Mu)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    save_dual_formats(plots_dir, 'saturation_over_time')
    plt.close()
    print(f"\nData saved to {plots_dir}")


if __name__ == "__main__":
    main()
