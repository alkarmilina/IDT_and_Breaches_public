import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part5/monthly_data_breach_conversion_beta.py

"""
Part 5: Monthly Breach-to-Victim Conversion Rate (Alpha-Discounted)

The paper's primary breach-to-victim conversion model. Records exposed
each month decay in relevance over time (roughly 20% of a record's
utility is lost per month), so the supply of exploitable records in a
given month is a discounted cumulative sum of past months' exposure,
not just that month's raw total. Fits a log-quadratic curve to the
resulting conversion rate over time.

Setup:
Reads monthly victim counts from the ITS victims-only dataset from
part1, and monthly raw records exposed from the augmented PRC dataset
from part3. Heartland's 2009 breach is added manually since it predates
the PRC/HHS/state augmentation sources. ALPHA=0.80 is the paper's
monthly discount factor.

Goal:
Produce the paper's primary alpha-discounted conversion rate plot and
log-quadratic fit.

Outputs:
- plots/part5/conversion_rate_beta.png/.pdf: the conversion rate over
  time, with a log-quadratic fit and mega-breach events annotated.
- data/processed/part5/conversion_data.csv: the merged monthly data
  behind the plot. Consumed by calculate_megabreach_cost_extended.py
  and the part6 Wilcoxon tests.
"""

SMOOTH_WINDOW = 6
CUTOFF_DATE = '2022-01-01'
START_DATE = '2008-01-01'
ALPHA = 0.80  # Monthly discount factor.

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


def get_raw_breach_monthly(raw_breach_path):
    """Monthly raw records exposed from the augmented PRC dataset, with Heartland's 2009 breach added manually and zero-count months interpolated."""
    df = pd.read_csv(raw_breach_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])

    heartland = pd.DataFrame({
        'reported_date': [pd.Timestamp('2009-01-20')],
        'total_affected': [130000000]
    })
    df = pd.concat([df, heartland], ignore_index=True)

    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Year_Month')['total_affected'].sum().reset_index()
    monthly.rename(columns={'total_affected': 'Raw_Records_Exposed'}, inplace=True)

    monthly = monthly.sort_values('Year_Month').reset_index(drop=True)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].replace(0, np.nan)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].interpolate(method='linear')
    return monthly


def format_sci(val):
    """Formats a float in LaTeX scientific notation, e.g. 1.41 \\times 10^{-4}."""
    s = "{:.4e}".format(val)
    base, exp = s.split('e')
    return f"{base} \\times 10^{{{int(exp)}}}"


def save_plot(fig, output_folder, name):
    """Saves a matplotlib figure as both PNG and PDF under the same base file name."""
    fig.savefig(os.path.join(output_folder, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_folder, f'{name}.pdf'), bbox_inches='tight')


def main():
    print("\n--- Part 5: Monthly Breach-to-Victim Conversion Rate (Alpha-Discounted) ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    its_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    raw_breach_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    csv_output_dir = os.path.join(root_dir, 'data/processed/part5')
    plots_dir = os.path.join(root_dir, 'plots/part5')
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    its_df = get_its_monthly_data(its_data_path)
    print(f"Loaded data from {its_data_path}")
    raw_df = get_raw_breach_monthly(raw_breach_path)
    print(f"Loaded data from {raw_breach_path}")

    merged_df = pd.merge(its_df, raw_df, on='Year_Month', how='outer')
    merged_df['Date'] = pd.to_datetime(merged_df['Year_Month'])
    merged_df = merged_df[(merged_df['Date'] < CUTOFF_DATE) & (merged_df['Date'] >= START_DATE)].sort_values('Date').reset_index(drop=True)

    merged_df['Estimated_Victims'] = np.exp(np.log(merged_df['Estimated_Victims'].astype(float)).interpolate(method='linear'))
    merged_df['Estimated_Victims'] = merged_df['Estimated_Victims'].ffill().bfill()

    # D_t = sum of alpha^(t-k) * M_k, the alpha-discounted cumulative record supply.
    D_t_values = []
    current_sum = 0
    for r in merged_df['Raw_Records_Exposed']:
        current_sum = r + (ALPHA * current_sum)
        D_t_values.append(current_sum)

    merged_df['D_t'] = D_t_values

    merged_df['C_t_raw'] = (merged_df['Estimated_Victims'] / merged_df['D_t']) * 100_000
    merged_df['C_t_smooth'] = merged_df['C_t_raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

    # Months since Jan 2008.
    merged_df['t'] = np.arange(len(merged_df))
    fit_data = merged_df.dropna(subset=['C_t_smooth']).copy()
    x_time = fit_data['t'].values
    y_log_rate = np.log(fit_data['C_t_smooth'].values)

    # ln(C_t) = a*t^2 + b*t + c
    coeffs = np.polyfit(x_time, y_log_rate, 2)
    poly_log = np.poly1d(coeffs)
    merged_df['C_t_fit'] = np.exp(poly_log(merged_df['t']))

    print("Plotting conversion rate...")
    fig1, ax1 = plt.subplots(figsize=(16, 12))

    ax1.plot(merged_df['Date'], merged_df['C_t_raw'], color='mediumpurple', linewidth=4, alpha=0.7, label='Monthly rate')
    ax1.plot(merged_df['Date'], merged_df['C_t_smooth'], color='indigo', linewidth=6, label='6-Mo moving avg')
    ax1.plot(merged_df['Date'], merged_df['C_t_fit'], color='cyan', linewidth=8, label='Fit')

    ax1.set_yscale('log')
    ax1.set_title("Estimated Conversion Rate of Cumulative Records to IDT", pad=20)
    ax1.set_ylabel('Victims per 100k Discounted Records')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')

    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    events = [('Equifax', '2017-09-01', (40, 40)), ('Yahoo', '2016-12-01', (-40, 40)), ('Target', '2013-12-01', (-40, 40)), ('Heartland', '2009-01-01', (40, 40))]
    for label, date_str, offset in events:
        event_date = pd.to_datetime(date_str)
        y_val = merged_df.loc[merged_df['Date'] == event_date, 'C_t_smooth'].values
        if len(y_val) > 0:
            ax1.annotate(label, xy=(event_date, y_val[0]), xytext=offset, textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='black', lw=2), fontsize=25, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    eqn_text = rf"$\ln(\mathcal{{C}}_t) = ({format_sci(coeffs[0])})t^2 + ({format_sci(coeffs[1])})t + {coeffs[2]:.2f}$"
    ax1.text(0.05, 0.05, f"Fit Equation:\n{eqn_text}",
             transform=ax1.transAxes, fontsize=24, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_plot(fig1, plots_dir, 'conversion_rate_beta')
    plt.close(fig1)
    print(f"\nData saved to {plots_dir}")

    merged_df['Month_Str'] = merged_df['Year_Month']  # For downstream scripts.
    output_path = os.path.join(csv_output_dir, 'conversion_data.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Fit coefficients: a={coeffs[0]:.4e}, b={coeffs[1]:.4e}, c={coeffs[2]:.4f}")


if __name__ == '__main__':
    main()
