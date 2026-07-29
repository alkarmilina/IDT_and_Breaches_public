import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part5/breach_type_conversion.py

"""
Part 5: HACK-Specific Breach-to-Victim Conversion Rate

The aggregate conversion rate treats all breach types uniformly, but
the PRC dataset classifies each incident by its method of compromise.
This computes the conversion rate restricted to HACK (hacking or
malware intrusion) breaches only, the type behind all four of the
paper's mega-breach case studies, using the same alpha-discounted
cumulative pool as the aggregate rate, restricted to HACK records.

Setup:
Reads monthly victim counts from the ITS victims-only dataset from
part1, and monthly HACK-type records exposed from the augmented PRC
dataset from part3. ALPHA=0.80 and THETA=1.75 match the aggregate
conversion rate model.

Goal:
Produce the HACK-specific conversion rate and its log-quadratic fit.

Outputs:
- data/processed/part5/breach_type_conversion.csv: monthly HACK-type
  discounted record pool and conversion rate.
- data/processed/part5/breach_type_fit_coeffs.csv: the log-quadratic
  fit coefficients.
- plots/part5/breach_type_nU.png/.pdf: the HACK-type discounted
  cumulative record pool over time.
- plots/part5/breach_type_conversion.png/.pdf: the HACK-specific
  conversion rate, with a log-quadratic fit and mega-breach events
  annotated.
"""

ALPHA = 0.80  # Monthly discount factor.
SMOOTH_WINDOW = 6
CUTOFF_DATE = '2022-01-01'
START_DATE = '2008-01-01'
THETA = 1.75  # Under-reporting scaler, matches the saturation model.

TYPE_MAP = {'HACK': 'HACK'}
FOCUS_TYPES = ['HACK']
TYPE_COLORS = {'HACK': '#5B2D8E'}

# Heartland (Jan 2009) is a HACK breach absent from PRC — added manually.
HACK_MEGA_SHOCKS = {'2009-01': 130_000_000}

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 30, 'axes.titlesize': 30, 'axes.labelsize': 30,
    'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'legend.fontsize': 22, 'figure.titlesize': 35,
})


def load_its_monthly(its_data_path):
    """Monthly weighted victim counts from the ITS victims-only dataset, by discovery month."""
    df = pd.read_parquet(its_data_path)
    q2m = {1.0: 2, 2.0: 5, 3.0: 8, 4.0: 11}
    for c in ['DISCOVERY_MONTH', 'INTERVIEW_QUARTER', 'year']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    mask = df['DISCOVERY_MONTH'].isnull()
    df.loc[mask, 'DISCOVERY_MONTH'] = df.loc[mask, 'INTERVIEW_QUARTER'].map(q2m)
    df = df.dropna(subset=['DISCOVERY_MONTH', 'year', 'FINAL_ITS_WEIGHT'])
    df['Year_Month'] = (df['year'].astype(int).astype(str) + '-' +
                        df['DISCOVERY_MONTH'].astype(int).astype(str).str.zfill(2))
    monthly = df.groupby('Year_Month')['FINAL_ITS_WEIGHT'].sum().reset_index()
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Victims'}, inplace=True)
    return monthly


def load_prc_by_type(prc_path):
    """Loads the augmented PRC dataset and labels each incident's breach type, keeping only types in TYPE_MAP."""
    df = pd.read_csv(prc_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])
    df['label'] = df['group_org_breach_type'].map(TYPE_MAP)
    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    return df


def build_full_date_index():
    """A complete month-by-month index spanning the study period, used to fill in months with no incidents of a given type."""
    idx = pd.period_range(start=START_DATE, end=CUTOFF_DATE, freq='M')
    return pd.DataFrame({'Year_Month': idx.strftime('%Y-%m')})


def get_monthly_records(df_prc, label):
    """Monthly raw record counts for one breach-type label, zero-filled for months with no incidents of that type."""
    sub = df_prc[df_prc['label'] == label].copy()
    monthly = (sub.groupby('Year_Month')['total_affected_imputed']
                  .sum()
                  .reset_index()
                  .rename(columns={'total_affected_imputed': 'Raw_Records'}))

    full = build_full_date_index()
    monthly = full.merge(monthly, on='Year_Month', how='left').fillna(0)
    monthly = monthly.sort_values('Year_Month').reset_index(drop=True)

    if label == 'HACK':
        for ym, shock in HACK_MEGA_SHOCKS.items():
            monthly.loc[monthly['Year_Month'] == ym, 'Raw_Records'] += shock

    return monthly


def discounted_cumsum(series, alpha):
    """Alpha-discounted cumulative record pool for a single breach type, theta-scaled for under-reporting."""
    result = []
    current = 0.0
    for v in series:
        current = v * THETA + alpha * current
        result.append(current)
    return result


def fit_log_quadratic(months_idx, rate_smoothed):
    """Fits ln(rate) = a*t^2 + b*t + c. Returns (None, None) if fewer than 5 valid points."""
    valid = ~np.isnan(rate_smoothed)
    if valid.sum() < 5:
        return None, None
    coeffs = np.polyfit(months_idx[valid], np.log(rate_smoothed[valid]), 2)
    return np.poly1d(coeffs), coeffs


def format_sci(val):
    """Formats a float in LaTeX scientific notation, e.g. 1.41 \\times 10^{-4}."""
    s = f'{val:.4e}'
    base, exp = s.split('e')
    return f'{base} \\times 10^{{{int(exp)}}}'


def main():
    print("\n--- Part 5: HACK-Specific Breach-to-Victim Conversion Rate ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    its_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    prc_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    output_csv_dir = os.path.join(root_dir, 'data/processed/part5')
    output_plot_dir = os.path.join(root_dir, 'plots/part5')
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    its_df = load_its_monthly(its_data_path)
    print(f"Loaded data from {its_data_path}")
    prc_df = load_prc_by_type(prc_path)
    print(f"Loaded data from {prc_path}")
    full_df = build_full_date_index()

    # Interpolate ITS victims onto the full monthly grid.
    base = full_df.merge(its_df, on='Year_Month', how='left')
    base['Date'] = pd.to_datetime(base['Year_Month'])
    base = base[(base['Date'] >= START_DATE) & (base['Date'] < CUTOFF_DATE)].reset_index(drop=True)
    base['Estimated_Victims'] = pd.to_numeric(base['Estimated_Victims'], errors='coerce').replace(0, np.nan)
    base['Estimated_Victims'] = np.exp(np.log(base['Estimated_Victims']).interpolate(method='linear')).ffill().bfill()
    base['t'] = np.arange(len(base))

    all_rows = []
    fit_rows = []

    for label in FOCUS_TYPES:
        print(f"Processing {label}...")
        monthly_rec = get_monthly_records(prc_df, label)
        merged = base.copy()
        merged = merged.merge(monthly_rec, on='Year_Month', how='left').fillna({'Raw_Records': 0})
        merged['D_t'] = discounted_cumsum(merged['Raw_Records'].values, ALPHA)

        # Avoid dividing by a near-zero pool in the earliest months.
        min_pool = merged['D_t'].replace(0, np.nan).quantile(0.05)
        merged['D_t_safe'] = merged['D_t'].clip(lower=max(min_pool, 1))

        merged['C_t_raw'] = (merged['Estimated_Victims'] / merged['D_t_safe']) * 100_000
        merged['C_t_smooth'] = merged['C_t_raw'].rolling(SMOOTH_WINDOW, center=True).mean()
        merged['Label'] = label

        poly, coeffs = fit_log_quadratic(merged['t'].values, merged['C_t_smooth'].values)
        if poly is not None:
            merged['C_t_fit'] = np.exp(poly(merged['t'].values))
            fit_rows.append({
                'breach_type': label,
                'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2],
            })
            print(f"  ln(C) = ({format_sci(coeffs[0])})t^2 + ({format_sci(coeffs[1])})t + {coeffs[2]:.4f}")
        else:
            merged['C_t_fit'] = np.nan

        all_rows.append(merged[['Year_Month', 'Date', 't', 'Label',
                                 'Raw_Records', 'D_t', 'C_t_raw',
                                 'C_t_smooth', 'C_t_fit']])

    combined = pd.concat(all_rows, ignore_index=True)
    output_csv_path = os.path.join(output_csv_dir, 'breach_type_conversion.csv')
    combined.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

    fit_df = pd.DataFrame(fit_rows)
    fit_coeffs_path = os.path.join(output_csv_dir, 'breach_type_fit_coeffs.csv')
    fit_df.to_csv(fit_coeffs_path, index=False)
    print(f"Data saved to {fit_coeffs_path}")

    print("Plotting discounted cumulative records...")
    fig, ax = plt.subplots(figsize=(16, 10))
    for label in FOCUS_TYPES:
        sub = combined[combined['Label'] == label].copy()
        sub = sub[sub['D_t'] > 0]
        ax.plot(sub['Date'], sub['D_t'], label=label,
                color=TYPE_COLORS[label], linewidth=4)
    ax.set_yscale('log')
    ax.set_ylabel(r'Discounted Cumulative Records ($n_U$)')
    ax.set_title(r'Discounted Cumulative Exposed Records ($n_U$), HACK Breaches')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: (f'{x/1e9:.1f}B' if x >= 1e9
                      else f'{x/1e6:.0f}M' if x >= 1e6
                      else f'{x/1e3:.0f}K' if x >= 1000
                      else f'{int(x)}')))
    ax.legend(frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(output_plot_dir, 'breach_type_nU.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(output_plot_dir, 'breach_type_nU.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Plotting conversion rate...")
    fig, ax = plt.subplots(figsize=(16, 11))
    for label in FOCUS_TYPES:
        sub = combined[combined['Label'] == label]
        c = TYPE_COLORS[label]
        ax.plot(sub['Date'], sub['C_t_smooth'], color=c, linewidth=4,
                alpha=0.75, label=f'{label} (6-mo avg)')
        if 'C_t_fit' in sub.columns:
            ax.plot(sub['Date'], sub['C_t_fit'], color=c, linewidth=3,
                    linestyle='--', alpha=0.9, label=f'{label} fit')

    btrans = blended_transform_factory(ax.transData, ax.transAxes)
    events = [('Heartland', '2009-01-01'), ('Target', '2013-12-01'), ('Equifax', '2017-09-01')]
    for ev_name, ev_date in events:
        xdate = pd.to_datetime(ev_date)
        ax.plot([xdate, xdate], [-0.09, 1.0], transform=btrans,
                color='#333333', linestyle=':', linewidth=2.5, clip_on=False)
        ax.text(xdate, -0.13, ev_name, transform=btrans,
                fontsize=20, ha='center', va='top',
                color='#333333', clip_on=False, fontweight='bold')

    ax.set_yscale('log')
    ax.set_ylabel(r'IDT Victims per 100k Discounted Records ($n_U$)')
    ax.set_title('HACK-Specific Conversion Rate ($\\mathcal{C}_t$)')
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    # Extra bottom margin so the below-axis event labels don't get clipped.
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(os.path.join(output_plot_dir, 'breach_type_conversion.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(output_plot_dir, 'breach_type_conversion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nData saved to {output_plot_dir}")


if __name__ == '__main__':
    main()
