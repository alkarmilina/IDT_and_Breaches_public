import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# python scripts/part5/breach_type_conversion.py
#
# Computes breach-type-specific conversion rates (nU) and updates the
# upper-bound social cost estimates for Heartland, Target, and Equifax
# using the HACK-specific log-quadratic fit.
#
# Outputs
# -------
# plots/part5/breach_type_nU.pdf/.png          -- nU by type over time
# plots/part5/breach_type_conversion.pdf/.png  -- conversion rate by type + fits
# data/processed/part5/breach_type_conversion.csv
# data/processed/part5/breach_type_fit_coeffs.csv
# data/processed/part5/breach_type_case_study_costs.csv  (updated upper bounds)

# ── PATHS ──────────────────────────────────────────────────────────────────────
ITS_DATA_PATH      = 'data/processed/part1/its_victims.parquet'
PRC_PATH           = 'data/processed/part3/PRC_augmented.csv'
SOCIAL_COST_PATH   = 'data/processed/part2/social_cost/social_cost_analysis.csv'
OUTPUT_CSV_DIR     = 'data/processed/part5'
OUTPUT_PLOT_DIR    = os.path.join('plots', 'part5')
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# ── PARAMETERS ─────────────────────────────────────────────────────────────────
ALPHA         = 0.80   # monthly discount factor α (Eq. 1 in paper)
SMOOTH_WINDOW = 6      # months for moving-average smoothing
CUTOFF_DATE   = '2022-01-01'
START_DATE    = '2008-01-01'
THETA         = 1.75   # under-reporting scaler (matches saturation model)
DISC_LAG      = 2      # discovery lag used in upper-bound projections

# Breach types to analyse (merge PORT into PHYS since both are physical)
# UNKN excluded — cannot be attributed to a meaningful category
PHYS_TYPES = {'PHYS', 'PORT'}
TYPE_MAP   = {'HACK': 'HACK', 'PHYS': 'PHYS/PORT', 'PORT': 'PHYS/PORT',
              'DISC': 'DISC', 'INSD': 'INSD'}
FOCUS_TYPES = ['HACK', 'PHYS/PORT', 'DISC', 'INSD']

TYPE_COLORS = {
    'HACK':     '#5B2D8E',   # deep purple  (matches existing paper purple)
    'PHYS/PORT':'#E07B39',   # amber
    'DISC':     '#2E8B57',   # sea green
    'INSD':     '#C0392B',   # red
}

# Heartland (Jan 2009) is a HACK breach absent from PRC — add it manually.
# All other major breaches (Target, Equifax, etc.) are already in PRC_augmented.
HACK_MEGA_SHOCKS = {
    '2009-01': 130_000_000,
}

# Case studies — all HACK
CASE_STUDIES = [
    {'name': 'Heartland', 'date': '2009-01-01', 'size': 130_000_000},
    {'name': 'Target',    'date': '2013-12-01', 'size':  40_000_000},
    {'name': 'Equifax',   'date': '2017-09-01', 'size': 147_000_000},
]
SOCIAL_COST_INTERP_T    = [0,  48,  72,  96, 120, 156]
SOCIAL_COST_INTERP_VALS = [1110.31, 921.38, 853.41, 245.27, 244.40, 295.73]

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 30, 'axes.titlesize': 30, 'axes.labelsize': 30,
    'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'legend.fontsize': 22, 'figure.titlesize': 35,
})

# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_its_monthly():
    df = pd.read_parquet(ITS_DATA_PATH)
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


def load_prc_by_type():
    df = pd.read_csv(PRC_PATH)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])
    df['label'] = df['group_org_breach_type'].map(TYPE_MAP)
    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    return df


def build_full_date_index():
    idx = pd.period_range(start=START_DATE, end=CUTOFF_DATE, freq='M')
    return pd.DataFrame({'Year_Month': idx.strftime('%Y-%m')})


def get_monthly_records(df_prc, label):
    """Return monthly raw record counts for one breach-type label."""
    sub = df_prc[df_prc['label'] == label].copy()
    monthly = (sub.groupby('Year_Month')['total_affected_imputed']
                  .sum()
                  .reset_index()
                  .rename(columns={'total_affected_imputed': 'Raw_Records'}))

    if label == 'HACK':
        for ym, shock in HACK_MEGA_SHOCKS.items():
            row = monthly[monthly['Year_Month'] == ym]
            if row.empty:
                monthly = pd.concat([monthly, pd.DataFrame({'Year_Month': [ym], 'Raw_Records': [shock]})],
                                    ignore_index=True)
            else:
                monthly.loc[monthly['Year_Month'] == ym, 'Raw_Records'] = \
                    monthly.loc[monthly['Year_Month'] == ym, 'Raw_Records'].values[0] + shock

    full = build_full_date_index()
    monthly = full.merge(monthly, on='Year_Month', how='left').fillna(0)
    monthly = monthly.sort_values('Year_Month').reset_index(drop=True)
    return monthly


def discounted_cumsum(series, alpha):
    """D_t^τ = Σ α^(t-k) · θ · M_k^τ  (type-specific Eq. 1 variant)"""
    result = []
    current = 0.0
    for v in series:
        current = v * THETA + alpha * current
        result.append(current)
    return result


def fit_log_quadratic(months_idx, rate_smoothed):
    """Fit ln(rate) = a*t^2 + b*t + c; return poly1d and coefficients."""
    valid = ~np.isnan(rate_smoothed)
    if valid.sum() < 5:
        return None, None
    coeffs = np.polyfit(months_idx[valid], np.log(rate_smoothed[valid]), 2)
    return np.poly1d(coeffs), coeffs


def format_sci(val):
    s = f'{val:.4e}'
    base, exp = s.split('e')
    return f'{base} \\times 10^{{{int(exp)}}}'


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading ITS and PRC data...')
    its_df  = load_its_monthly()
    prc_df  = load_prc_by_type()
    full_df = build_full_date_index()

    # Interpolate ITS victims onto full monthly grid
    base = full_df.merge(its_df, on='Year_Month', how='left')
    base['Date'] = pd.to_datetime(base['Year_Month'])
    base = base[(base['Date'] >= START_DATE) & (base['Date'] < CUTOFF_DATE)].reset_index(drop=True)
    base['Estimated_Victims'] = pd.to_numeric(base['Estimated_Victims'], errors='coerce').replace(0, np.nan)
    base['Estimated_Victims'] = np.exp(np.log(base['Estimated_Victims']).interpolate(method='linear')).ffill().bfill()
    base['t'] = np.arange(len(base))

    # Per-type conversion rate table
    all_rows  = []
    fit_rows  = []
    polys     = {}   # label -> poly1d

    # ── TYPE-SPECIFIC LOOP ──────────────────────────────────────────────────────
    for label in FOCUS_TYPES:
        print(f'  Processing {label}...')
        monthly_rec = get_monthly_records(prc_df, label)
        merged = base.copy()
        merged = merged.merge(monthly_rec, on='Year_Month', how='left').fillna({'Raw_Records': 0})
        merged['D_t'] = discounted_cumsum(merged['Raw_Records'].values, ALPHA)

        # Avoid division by near-zero
        min_pool = merged['D_t'].replace(0, np.nan).quantile(0.05)
        merged['D_t_safe'] = merged['D_t'].clip(lower=max(min_pool, 1))

        merged['C_t_raw']    = (merged['Estimated_Victims'] / merged['D_t_safe']) * 100_000
        merged['C_t_smooth'] = merged['C_t_raw'].rolling(SMOOTH_WINDOW, center=True).mean()
        merged['Label']      = label

        poly, coeffs = fit_log_quadratic(merged['t'].values, merged['C_t_smooth'].values)
        if poly is not None:
            merged['C_t_fit'] = np.exp(poly(merged['t'].values))
            polys[label] = (poly, coeffs)
            fit_rows.append({
                'breach_type': label,
                'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2],
            })
            print(f'    ln(C) = ({format_sci(coeffs[0])})t² + ({format_sci(coeffs[1])})t + {coeffs[2]:.4f}')
        else:
            merged['C_t_fit'] = np.nan

        all_rows.append(merged[['Year_Month', 'Date', 't', 'Label',
                                 'Raw_Records', 'D_t', 'C_t_raw',
                                 'C_t_smooth', 'C_t_fit']])

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(f'{OUTPUT_CSV_DIR}/breach_type_conversion.csv', index=False)

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(f'{OUTPUT_CSV_DIR}/breach_type_fit_coeffs.csv', index=False)
    print(f'\nFit coefficients:\n{fit_df.to_string(index=False)}')

    # ── FIGURE 1: nU (D_t) BY TYPE ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))
    for label in FOCUS_TYPES:
        sub = combined[combined['Label'] == label].copy()
        sub = sub[sub['D_t'] > 0]
        ax.plot(sub['Date'], sub['D_t'], label=label,
                color=TYPE_COLORS[label], linewidth=4)
    ax.set_yscale('log')
    ax.set_ylabel(r'Discounted Cumulative Records ($n_U$)')
    ax.set_title(r'Discounted Cumulative Exposed Records ($n_U$) by Breach Type')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: (f'{x/1e9:.1f}B' if x >= 1e9
                      else f'{x/1e6:.0f}M' if x >= 1e6
                      else f'{x/1e3:.0f}K' if x >= 1000
                      else f'{int(x)}')))
    ax.legend(frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_PLOT_DIR}/breach_type_nU.pdf', bbox_inches='tight')
    fig.savefig(f'{OUTPUT_PLOT_DIR}/breach_type_nU.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── FIGURE 2: CONVERSION RATE BY TYPE ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 11))
    for label in FOCUS_TYPES:
        sub = combined[combined['Label'] == label]
        c   = TYPE_COLORS[label]
        ax.plot(sub['Date'], sub['C_t_smooth'], color=c, linewidth=4,
                alpha=0.75, label=f'{label} (6-mo avg)')
        if 'C_t_fit' in sub.columns:
            ax.plot(sub['Date'], sub['C_t_fit'], color=c, linewidth=3,
                    linestyle='--', alpha=0.9, label=f'{label} fit')

    # Event lines + labels below the x-axis
    # blended transform: x in data coords, y in axes fraction (allows clip_on=False)
    from matplotlib.transforms import blended_transform_factory
    btrans = blended_transform_factory(ax.transData, ax.transAxes)

    events = [('Heartland', '2009-01-01'), ('Target', '2013-12-01'), ('Equifax', '2017-09-01')]
    for ev_name, ev_date in events:
        xdate = pd.to_datetime(ev_date)
        # Vertical line extending from slightly below axis up to top of plot
        ax.plot([xdate, xdate], [-0.09, 1.0], transform=btrans,
                color='#333333', linestyle=':', linewidth=2.5, clip_on=False)
        # Label sits below the x-axis tick labels
        ax.text(xdate, -0.13, ev_name, transform=btrans,
                fontsize=20, ha='center', va='top',
                color='#333333', clip_on=False, fontweight='bold')

    ax.set_yscale('log')
    ax.set_ylabel(r'IDT Victims per 100k Discounted Records ($n_U$)')
    ax.set_title('Breach-Type-Specific Conversion Rate ($\\mathcal{C}_t$)')
    ax.legend(frameon=True, ncol=2, fontsize=18, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    # Extra bottom margin so below-axis labels don't get clipped
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(f'{OUTPUT_PLOT_DIR}/breach_type_conversion.pdf', bbox_inches='tight')
    fig.savefig(f'{OUTPUT_PLOT_DIR}/breach_type_conversion.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figures saved to {OUTPUT_PLOT_DIR}')

    # ── UPPER-BOUND CASE STUDIES WITH HACK-SPECIFIC FIT ────────────────────────
    if 'HACK' not in polys:
        print('ERROR: HACK fit unavailable — cannot update case study upper bounds.')
        return

    hack_poly, hack_coeffs = polys['HACK']
    print(f'\nHACK fit: ln(C) = ({format_sci(hack_coeffs[0])})t² + '
          f'({format_sci(hack_coeffs[1])})t + {hack_coeffs[2]:.4f}')

    hack_dates = base.copy()
    hack_dates['C_hack'] = np.exp(hack_poly(hack_dates['t'].values))

    results = []
    for cs in CASE_STUDIES:
        T_date = pd.to_datetime(cs['date'])
        T_row  = hack_dates[hack_dates['Date'] == T_date]
        if T_row.empty:
            print(f"WARNING: {cs['name']} date not in date index — skipping.")
            continue
        T_idx  = T_row.index[0]
        lag    = T_idx + DISC_LAG

        total_victims   = 0.0
        total_cost      = 0.0
        for k in range(lag, len(hack_dates)):
            months_since = k - T_idx
            fresh_records = cs['size'] * (ALPHA ** months_since)
            t_k           = hack_dates.iloc[k]['t']
            conv_rate     = np.exp(hack_poly(t_k)) / 100_000
            monthly_vic   = fresh_records * conv_rate
            sc_per_victim = np.interp(t_k, SOCIAL_COST_INTERP_T, SOCIAL_COST_INTERP_VALS)
            total_victims += monthly_vic
            total_cost    += monthly_vic * sc_per_victim

        results.append({
            'Breach':                      cs['name'],
            'Month':                       cs['date'][:7],
            'Records Exposed':             cs['size'],
            'Total Projected Victims':     round(total_victims),
            'Upper Bound Social Cost ($)': round(total_cost, 2),
            'Upper Bound Social Cost ($B)': round(total_cost / 1e9, 3),
        })
        print(f"\n{cs['name']} (HACK-specific fit):")
        print(f"  Projected Victims:      {total_victims:,.0f}")
        print(f"  Upper Bound Soc. Cost: ${total_cost:,.2f}  (${total_cost/1e9:.3f}B)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{OUTPUT_CSV_DIR}/breach_type_case_study_costs.csv', index=False)
    print(f'\nCase study results saved to {OUTPUT_CSV_DIR}/breach_type_case_study_costs.csv')


if __name__ == '__main__':
    main()
