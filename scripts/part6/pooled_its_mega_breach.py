import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# To run: python scripts/part6/pooled_its_mega_breach.py

# --- CONFIGURATION (match wilcoxon_mega_breach_test.py) ---
RAW_THRESHOLD      = 10_000_000
OBSERVATION_WINDOW = 6    # months pre and post (same as Wilcoxon)
DISCOVERY_LAG      = 2    # months after breach before post-window starts (same as Wilcoxon lower bound)

MEGA_SHOCKS = {
    '2008-05': 12_500_000, '2009-01': 130_000_000, '2011-04': 77_000_000,
    '2013-12': 40_000_000, '2014-09': 56_000_000,  '2017-09': 143_000_000,
}


def identify_events(df, record_col, threshold):
    """
    Identify mega-breach event indices using the same logic as the Wilcoxon script:
    3-month cooldown between events; 2017-09 replaces 2017-06 if adjacent.
    """
    indices = df[df[record_col] >= threshold].index.tolist()
    event_indices = []
    if not indices:
        return event_indices

    event_indices.append(indices[0])
    for i in range(1, len(indices)):
        current  = df.loc[indices[i],          'Month_Str']
        last_sel = df.loc[event_indices[-1],   'Month_Str']
        if current == '2017-09' and last_sel == '2017-06':
            event_indices[-1] = indices[i]
            continue
        if indices[i] - event_indices[-1] > 3:
            event_indices.append(indices[i])

    return event_indices


def build_panel(df, event_indices, idt_col, window=OBSERVATION_WINDOW, lag=DISCOVERY_LAG):
    """
    For each event, extract (window) pre-breach and (window) post-breach months
    (offset by discovery lag). Normalize victims by the pre-breach mean so that
    the outcome is a fraction of baseline — this removes between-event level
    differences without event fixed effects, which become collinear with `post`
    when each event window is only 2*window observations.

    Returns long-format panel with columns:
      event_id, month_label, org, t, post, victims, victims_norm
    """
    rows = []
    usable = 0
    for event_id, idx in enumerate(event_indices):
        pre_start  = idx - window
        post_start = idx + lag
        post_end   = post_start + window - 1

        if pre_start < 0 or post_end >= len(df):
            continue

        pre_vals = df.loc[pre_start : idx - 1, idt_col].values
        pre_mean = pre_vals.mean()
        if pre_mean == 0:
            continue   # avoid division by zero

        usable += 1
        month_label = df.loc[idx, 'Month_Str']
        org = df.loc[idx, 'Top_Org'] if 'Top_Org' in df.columns else 'Unknown'

        for step, abs_idx in enumerate(range(pre_start, idx)):
            v = df.loc[abs_idx, idt_col]
            rows.append({
                'event_id':    event_id,
                'month_label': month_label,
                'org':         org,
                't':           step - window,      # -window to -1
                'post':        0,
                'victims':     v,
                'victims_norm': v / pre_mean,      # fraction of pre-breach baseline
            })

        for step, abs_idx in enumerate(range(post_start, post_end + 1)):
            v = df.loc[abs_idx, idt_col]
            rows.append({
                'event_id':    event_id,
                'month_label': month_label,
                'org':         org,
                't':           step,               # 0 to window-1
                'post':        1,
                'victims':     v,
                'victims_norm': v / pre_mean,
            })

    print(f"  Usable events (sufficient data): {usable} / {len(event_indices)}")
    return pd.DataFrame(rows)


def run_its_regression(panel):
    """
    Pooled interrupted time series on baseline-normalized outcome:
      victims_norm ~ t + post + t:post

    Coefficients:
      t        = pre-breach trend (fraction of baseline per month)
      post     = level shift at breach (fraction of baseline)
      t:post   = slope change post-breach
    Standard errors clustered by event_id.
    """
    model = smf.ols(
        'victims_norm ~ t + post + t:post',
        data=panel
    ).fit(cov_type='cluster', cov_kwds={'groups': panel['event_id']})

    return model


def plot_its(panel, out_dir):
    """
    Plot observed mean normalized pre/post victims across events.
    """
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

    obs = panel.groupby('t')['victims_norm'].mean()

    _, ax = plt.subplots(figsize=(14, 7))
    pre  = obs[obs.index < 0]
    post = obs[obs.index >= 0]
    ax.plot(pre.index,  pre.values,  'o-', color='tab:blue',
            linewidth=2.5, markersize=7, label='Pre-breach (mean across events)')
    ax.plot(post.index, post.values, 's-', color='tab:orange',
            linewidth=2.5, markersize=7, label='Post-breach (mean across events)')
    ax.axvline(x=0,  color='red',  linestyle=':', linewidth=2,
               label=f'Post-window start (lag={DISCOVERY_LAG}mo)')
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.5,
               label='Baseline (=1)')
    ax.set_xlabel(f'Months relative to breach event (window={OBSERVATION_WINDOW}mo, lag={DISCOVERY_LAG}mo)')
    ax.set_ylabel('IDT victims relative to pre-breach mean')
    ax.legend(loc='upper left')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out_dir, f'pooled_its.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pooled ITS plot saved to {out_dir}")


def main():
    root = os.getcwd()
    idt_path = os.path.join(root, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root, 'data/processed/part3/PRC_augmented.csv')

    idt_df  = pd.read_csv(idt_path)
    raw_csv = pd.read_csv(raw_path)

    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'],
                                               format='mixed', errors='coerce')
    raw_csv = raw_csv.dropna(subset=['reported_date'])
    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)

    # Top org per month (for labelling, same as Wilcoxon)
    def get_top_metadata(group):
        idx = group['total_affected'].idxmax()
        return pd.Series({'Top_Org': group.loc[idx, 'org_name']})

    meta_labels = raw_csv.groupby('Month_Str').apply(get_top_metadata).reset_index()

    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly.apply(
        lambda r: max(r['total_affected'], MEGA_SHOCKS.get(r['Month_Str'], 0)), axis=1
    )

    master = pd.merge(idt_df[['Month_Str', 'Estimated_Victims']],
                      raw_monthly, on='Month_Str', how='left')
    master = pd.merge(master, meta_labels, on='Month_Str', how='left')
    master = master.fillna(0).sort_values('Month_Str').reset_index(drop=True)

    print(f"Series: {master['Month_Str'].iloc[0]} – {master['Month_Str'].iloc[-1]} "
          f"({len(master)} months)")

    # --- Identify events ---
    event_indices = identify_events(master, 'total_affected_raw', RAW_THRESHOLD)
    print(f"Mega-breach events identified (threshold={RAW_THRESHOLD:,}): {len(event_indices)}")
    for idx in event_indices:
        print(f"  {master.loc[idx, 'Month_Str']}  {master.loc[idx, 'Top_Org']}")

    # --- Build panel ---
    panel = build_panel(master, event_indices, 'Estimated_Victims')
    print(f"\nPanel rows: {len(panel)}")

    # --- Run regression ---
    model = run_its_regression(panel)

    # --- Extract key coefficients ---
    params = model.params
    pvals  = model.pvalues
    ci     = model.conf_int()

    level_shift = params.get('post', np.nan)
    slope_change = params.get('t:post', np.nan)
    level_p  = pvals.get('post', np.nan)
    slope_p  = pvals.get('t:post', np.nan)
    level_ci = ci.loc['post'].values   if 'post'   in ci.index else [np.nan, np.nan]
    slope_ci = ci.loc['t:post'].values if 't:post' in ci.index else [np.nan, np.nan]

    print(f"\n{'='*60}")
    print("POOLED INTERRUPTED TIME SERIES RESULTS")
    print(f"{'='*60}")
    print(f"Level shift (β_post):    {level_shift:+.4f} × baseline  "
          f"(p={level_p:.4f}, 95% CI [{level_ci[0]:+.4f}, {level_ci[1]:+.4f}])")
    print(f"Slope change (β_t×post): {slope_change:+.4f} × baseline/month  "
          f"(p={slope_p:.4f}, 95% CI [{slope_ci[0]:+.4f}, {slope_ci[1]:+.4f}])")
    print(f"\nFull model summary:\n{model.summary()}")

    # --- Save results ---
    out_data = os.path.join(root, 'data/processed/part6')
    os.makedirs(out_data, exist_ok=True)

    summary_df = pd.DataFrame([{
        'coef':       'level_shift (post)',
        'estimate':   level_shift,
        'p_value':    level_p,
        'ci_lower':   level_ci[0],
        'ci_upper':   level_ci[1],
    }, {
        'coef':       'slope_change (t:post)',
        'estimate':   slope_change,
        'p_value':    slope_p,
        'ci_lower':   slope_ci[0],
        'ci_upper':   slope_ci[1],
    }])
    summary_df.to_csv(os.path.join(out_data, 'pooled_its_results.csv'), index=False)
    panel.to_csv(os.path.join(out_data, 'pooled_its_panel.csv'), index=False)
    print(f"\nResults saved to {out_data}/")

    plot_its(panel, out_dir='plots/part6')


if __name__ == '__main__':
    main()
