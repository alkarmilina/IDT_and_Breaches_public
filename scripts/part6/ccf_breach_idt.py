import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import os

# To run: python scripts/part6/ccf_breach_idt.py

# --- CONFIGURATION (match wilcoxon_mega_breach_test.py) ---
RAW_THRESHOLD  = 10_000_000
MAX_LAG        = 12   # months
AR_MAX_ORDER   = 6    # max AR order for pre-whitening (selected by AIC)

MEGA_SHOCKS = {
    '2008-05': 12_500_000, '2009-01': 130_000_000, '2011-04': 77_000_000,
    '2013-12': 40_000_000, '2014-09': 56_000_000,  '2017-09': 143_000_000,
}


def standardize(series):
    """Z-score standardize; avoids AR overflow on large raw counts."""
    s = np.array(series, dtype=float)
    return (s - s.mean()) / s.std()


def select_ar_order(series, max_order=AR_MAX_ORDER):
    """Select AR order by AIC over 1..max_order on a standardized series."""
    best_aic, best_p = np.inf, 1
    for p in range(1, max_order + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = AutoReg(series, lags=p, old_names=False).fit()
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best_p = res.aic, p
        except Exception:
            pass
    return best_p


def prewhiten(x, y):
    """
    Standardize both series, fit AR(p) to x, apply same filter to y.
    Pre-whitening removes autocorrelation so the CCF of residuals reflects
    true cross-correlation, not artefacts from log-linear interpolation
    between survey waves.
    """
    x_std = standardize(x)
    y_std = standardize(y)

    p = select_ar_order(x_std)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ar_model = AutoReg(x_std, lags=p, old_names=False).fit()
    x_res = np.array(ar_model.resid)

    # Apply same AR filter to y (drop first p obs to align lengths)
    ar_params = ar_model.params[1:]  # exclude intercept
    y_filtered = np.array([
        y_std[i] - sum(ar_params[j] * y_std[i - j - 1] for j in range(p))
        for i in range(p, len(y_std))
    ])
    return x_res, y_filtered, p


def compute_ccf_with_ci(x_res, y_res, max_lag=MAX_LAG):
    """Return CCF values and 95% CI bounds at lags 0..max_lag."""
    n = len(x_res)
    ci = 1.96 / np.sqrt(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ccf_raw = ccf(x_res, y_res, nlags=max_lag, alpha=None)
    # statsmodels may return nlags or nlags+1 values depending on version
    ccf_vals = ccf_raw[:max_lag + 1]
    lags = np.arange(len(ccf_vals))
    return lags, ccf_vals, ci


def plot_ccf(lags, ccf_vals, ci, ar_order, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 20,
    })

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['tab:blue' if abs(v) > ci else 'lightsteelblue' for v in ccf_vals]
    ax.bar(lags, ccf_vals, color=colors, width=0.6)
    ax.axhline(ci,  color='red', linestyle='--', linewidth=2, label='95% CI')
    ax.axhline(-ci, color='red', linestyle='--', linewidth=2)
    ax.axhline(0,   color='black', linewidth=0.8)

    ax.set_xlabel('Lag (months): breach records → IDT victims')
    ax.set_ylabel('Cross-Correlation')
    ax.set_xticks(lags)
    ax.legend(loc='upper right')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out_dir, f'ccf_breach_idt.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CCF plot saved to {out_dir}")


def main():
    root = os.getcwd()
    sat_path = os.path.join(root, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    idt_path = os.path.join(root, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root, 'data/processed/part3/PRC_augmented.csv')

    idt_df = pd.read_csv(idt_path)
    raw_csv = pd.read_csv(raw_path)

    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'],
                                               format='mixed', errors='coerce')
    raw_csv = raw_csv.dropna(subset=['reported_date'])
    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)

    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly.apply(
        lambda r: max(r['total_affected'], MEGA_SHOCKS.get(r['Month_Str'], 0)), axis=1
    )

    master = pd.merge(
        idt_df[['Month_Str', 'Estimated_Victims']],
        raw_monthly[['Month_Str', 'total_affected_raw']],
        on='Month_Str', how='inner'
    ).sort_values('Month_Str').reset_index(drop=True)

    breach_series = master['total_affected_raw'].astype(float)
    idt_series    = master['Estimated_Victims'].astype(float)

    print(f"Series length: {len(master)} months "
          f"({master['Month_Str'].iloc[0]} – {master['Month_Str'].iloc[-1]})")

    x_res, y_res, ar_p = prewhiten(breach_series, idt_series)
    print(f"Pre-whitening AR order selected: {ar_p}")

    lags, ccf_vals, ci = compute_ccf_with_ci(x_res, y_res)

    print(f"\nCCF results (95% CI = ±{ci:.4f}):")
    print(f"{'Lag':>5} {'CCF':>10} {'Significant':>12}")
    for lag, val in zip(lags, ccf_vals):
        sig = '*' if abs(val) > ci else ''
        print(f"{lag:>5} {val:>10.4f} {sig:>12}")

    # --- Save results ---
    out_data = os.path.join(root, 'data/processed/part6')
    os.makedirs(out_data, exist_ok=True)
    results_df = pd.DataFrame({
        'lag':         lags,
        'ccf':         ccf_vals,
        'ci_95':       np.full(len(lags), ci),
        'significant': np.abs(ccf_vals) > ci,
    })
    results_df.to_csv(os.path.join(out_data, 'ccf_results.csv'), index=False)
    print(f"\nCCF results saved to {out_data}/ccf_results.csv")

    plot_ccf(lags, ccf_vals, ci, ar_p, out_dir='plots/part6')


if __name__ == '__main__':
    main()
