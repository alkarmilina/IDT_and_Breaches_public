import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part5/monthly_data_breach_conversion.py

"""
Part 5: Monthly Breach-to-Victim Conversion Rate (Saturation Model)

Plots monthly IDT victims against the saturation model's estimated
number of unique individuals compromised, and the resulting
breach-to-victim conversion rate. This is the saturation-model
comparison to the paper's primary alpha-discounted conversion rate,
using the saturation model's supply (Y=1) instead of a simple
discounted record count.

Setup:
Reads monthly victim counts from the ITS victims-only and all-respondents
datasets from part1, and the saturation model's monthly supply from
apply_saturation_model_exp.py, which already has theta applied.

Goal:
Produce the saturation-model overlay and conversion rate plots.

Outputs:
- plots/part5/overlay.pdf: monthly records exposed vs. IDT victims.
- plots/part5/conversion_rate.pdf: the resulting conversion rate.
- data/processed/part5/conversion_data_saturation.csv: the merged
  monthly data behind both plots.
"""

SMOOTH_WINDOW = 6
CUTOFF_DATE = '2022-01-01'

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 35,
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 25,
    'figure.titlesize': 40
})


def get_its_monthly_data(its_data_path):
    """Monthly weighted victim counts from the ITS victims-only dataset, by discovery month."""
    df = pd.read_parquet(its_data_path)
    quarter_to_month = {1.0: 2, 2.0: 5, 3.0: 8, 4.0: 11}
    df['DISCOVERY_MONTH'] = pd.to_numeric(df['DISCOVERY_MONTH'], errors='coerce')
    df['INTERVIEW_QUARTER'] = pd.to_numeric(df['INTERVIEW_QUARTER'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    missing_mask = df['DISCOVERY_MONTH'].isnull()
    df.loc[missing_mask, 'DISCOVERY_MONTH'] = df.loc[missing_mask, 'INTERVIEW_QUARTER'].map(quarter_to_month)

    analysis_df = df.dropna(subset=['DISCOVERY_MONTH', 'year', 'FINAL_ITS_WEIGHT']).copy()
    analysis_df['Year_Month'] = (
        analysis_df['year'].astype(int).astype(str) + '-' +
        analysis_df['DISCOVERY_MONTH'].astype(int).astype(str).str.zfill(2)
    )
    monthly = analysis_df.groupby(['Year_Month'])['FINAL_ITS_WEIGHT'].sum().reset_index()
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Victims'}, inplace=True)
    return monthly


def get_gold_data(its_all_path):
    """Monthly weighted counts of respondents notified of a data breach, from the all-respondents ITS dataset."""
    df = pd.read_parquet(its_all_path)
    quarter_to_month = {1.0: 2, 2.0: 5, 3.0: 8, 4.0: 11}
    df['DISCOVERY_MONTH'] = pd.to_numeric(df['DISCOVERY_MONTH'], errors='coerce')
    df['INTERVIEW_QUARTER'] = pd.to_numeric(df['INTERVIEW_QUARTER'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    missing_mask = df['DISCOVERY_MONTH'].isnull()
    df.loc[missing_mask, 'DISCOVERY_MONTH'] = df.loc[missing_mask, 'INTERVIEW_QUARTER'].map(quarter_to_month)

    df = df.dropna(subset=['year', 'FINAL_ITS_WEIGHT']).copy()
    df['DISCOVERY_MONTH'] = df['DISCOVERY_MONTH'].fillna(1)

    df['Year_Month'] = (
        df['year'].astype(int).astype(str) + '-' +
        df['DISCOVERY_MONTH'].astype(int).astype(str).str.zfill(2)
    )
    notified = df[df['NOTIFIED_OF_DATA_BREACH'] == 1].groupby(['Year_Month'])['FINAL_ITS_WEIGHT'].sum().reset_index()
    notified.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Notified'}, inplace=True)
    return notified


def main():
    print("\n--- Part 5: Monthly Breach-to-Victim Conversion Rate (Saturation Model) ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    its_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    its_all_path = os.path.join(root_dir, 'data/processed/part1/its_full.parquet')
    final_supply_path = os.path.join(root_dir, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    csv_output_dir = os.path.join(root_dir, 'data/processed/part5')
    plots_dir = os.path.join(root_dir, 'plots/part5')
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    its_df = get_its_monthly_data(its_data_path)
    print(f"Loaded data from {its_data_path}")
    gold_df = get_gold_data(its_all_path)
    print(f"Loaded data from {its_all_path}")
    um_df = pd.read_csv(final_supply_path)
    print(f"Loaded data from {final_supply_path}")

    its_combined = pd.merge(its_df, gold_df, on='Year_Month', how='outer')
    merged_df = pd.merge(its_combined, um_df, left_on='Year_Month', right_on='Month_Str', how='outer')
    merged_df['Date'] = pd.to_datetime(merged_df['Month_Str'])
    merged_df.sort_values('Date', inplace=True)
    merged_df = merged_df[merged_df['Date'] < CUTOFF_DATE].copy()
    merged_df.reset_index(drop=True, inplace=True)

    # Log-linear interpolation across months with no reported victims/notifications.
    for col in ['Estimated_Victims', 'Estimated_Notified']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').replace(0, np.nan)
        merged_df[col] = np.exp(np.log(merged_df[col].astype(float)).interpolate(method='linear'))
        merged_df[col] = merged_df[col].ffill().bfill()

    # Um_Unique_Individuals already has theta applied in apply_saturation_model_exp.py.
    merged_df['Um_Final'] = merged_df['Um_Unique_Individuals']

    merged_df['C_t_raw'] = (merged_df['Estimated_Victims'] / merged_df['Um_Final']) * 100_000
    merged_df['C_t_smooth'] = merged_df['C_t_raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

    print("Plotting supply/demand overlay...")
    _, ax = plt.subplots(figsize=(16, 12))
    ax.plot(merged_df['Date'], merged_df['Um_Final'], color='tomato',
            label='Records Exposed', alpha=0.8, linestyle='-', linewidth=4)
    ax.plot(merged_df['Date'], merged_df['Estimated_Victims'], color='teal',
            label='IDT Victims', linestyle='--', linewidth=5)

    ax.set_title("Month-to-Month Breach Records and Victim IDT Reports", pad=20)
    ax.set_yscale('log')
    ax.set_ylabel('Count (Log Scale)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(loc='upper left', frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overlay.pdf'), dpi=300)
    plt.close()

    print("Plotting conversion rate...")
    _, ax = plt.subplots(figsize=(16, 12))
    ax.plot(merged_df['Date'], merged_df['C_t_raw'], color='purple', alpha=0.25,
            linewidth=2, label='Monthly Rate')
    ax.plot(merged_df['Date'], merged_df['C_t_smooth'], color='purple',
            linewidth=6, label=f'{SMOOTH_WINDOW}-Mo Moving Avg')

    ax.set_title("Month-to-Month Breach Victim to IDT Conversion Rate", pad=20)
    ax.set_yscale('log')
    ax.set_ylabel('Victims per 100k Records')

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, alpha=0.5, which='both', linestyle='--')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'conversion_rate.pdf'), dpi=300)
    plt.close()

    print(f"\nData saved to {plots_dir}")

    output_path = os.path.join(csv_output_dir, 'conversion_data_saturation.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


if __name__ == '__main__':
    main()
