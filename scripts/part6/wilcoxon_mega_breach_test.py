import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part6/wilcoxon_mega_breach_test.py

"""
Part 6: Wilcoxon Test for a Post-Mega-Breach Increase in IDT

Tests whether the six months following a mega-breach (a breach
exposing at least 10 million records) show a statistically significant
increase in reported identity theft, compared to the six months
before. Breaches within 3 months of each other are consolidated into a
single event, using the earliest breach's month, except Equifax, whose
September 2017 public disclosure is used instead of an earlier June
2017 entry in the PRC data. Swept across discovery lags of 0 to 8
months, since IDT is rarely discovered the same month data is exposed.
Run twice: once on raw records exposed (PRC), and once on the
saturation model's estimated unique individuals compromised, at a
correspondingly lower threshold.

Setup:
Reads monthly raw records exposed from the augmented PRC dataset from
part3, the saturation model's monthly estimate from part4, and the
alpha-discounted monthly IDT victim data from part5.

Goal:
Produce the Wilcoxon signed-rank test results for both the raw and
saturation-model mega-breach event series, and the discovery-lag=2
case study deltas that feed the lower-bound mega-breach social cost
estimate.

Outputs:
- plots/part6/longitudinal_sweep_raw_data.png/.pdf: p-value and average
  victim change by discovery lag, raw data.
- plots/part6/longitudinal_sweep_saturation_model.png/.pdf: same, for
  the saturation model.
"""

RAW_THRESHOLD = 10_000_000  # Mega-breach threshold for raw records exposed.
MODEL_THRESHOLD = 5_000_000  # Mega-breach threshold for the saturation model's unique individuals estimate.
OBSERVATION_WINDOW = 6  # Months in each of the pre- and post-breach windows.
DELAY_RANGE = range(0, 9)  # Discovery lags i in {0, ..., 8}.


def analyze_longitudinal_sweep(df, record_col, idt_col, label, threshold):
    """
    Identifies mega-breach events in record_col, consolidates events
    within 3 months of each other, and runs the Wilcoxon signed-rank
    test comparing pre- and post-breach idt_col medians, at every
    discovery lag in DELAY_RANGE.

    Args:
        df (pd.DataFrame): monthly data with record_col, idt_col,
            Month_Str, and (if available) Top_Org and Breach_Type.
        record_col (str): column used to identify mega-breach months.
        idt_col (str): column used to measure victimization change.
        label (str): label for print output.
        threshold (float): minimum record_col value to count as a
            mega-breach.

    Returns:
        (pd.DataFrame, pd.DataFrame): the p-value/average-delta sweep
        results by discovery lag, and the per-event delta at lag 2.
    """
    indices = df[df[record_col] >= threshold].index.tolist()
    event_indices = []

    if indices:
        event_indices.append(indices[0])
        for i in range(1, len(indices)):
            current_month = df.loc[indices[i], 'Month_Str']
            last_sel_month = df.loc[event_indices[-1], 'Month_Str']
            # Equifax: the September 2017 disclosure supersedes an earlier June 2017 entry.
            if current_month == '2017-09' and last_sel_month == '2017-06':
                event_indices[-1] = indices[i]
                continue
            if indices[i] - event_indices[-1] > 3:
                event_indices.append(indices[i])

    print(f"\n{label} (threshold: {threshold:,.0f})")
    print(f"{len(event_indices)} events identified")

    sweep_results = []
    event_summary_data = []

    for d in DELAY_RANGE:
        pairs = []
        deltas = []

        # Case study deltas are only needed at the paper's lowest-p-value lag (i=2).
        collect_summary = (d == 2)

        for idx in event_indices:
            if idx < OBSERVATION_WINDOW or idx + d + OBSERVATION_WINDOW > len(df):
                continue

            pre_med = df.loc[idx - OBSERVATION_WINDOW: idx - 1, idt_col].median()
            post_start = idx + d
            post_med = df.loc[post_start: post_start + OBSERVATION_WINDOW - 1, idt_col].median()
            delta = post_med - pre_med

            pairs.append((pre_med, post_med))
            deltas.append(delta)

            if collect_summary:
                month = df.loc[idx, 'Month_Str']
                org = df.loc[idx, 'Top_Org'] if 'Top_Org' in df.columns else "Unknown"
                b_type = df.loc[idx, 'Breach_Type'] if 'Breach_Type' in df.columns else "Unknown"

                print(f"Event: {org[:25]:<25} | Type: {str(b_type)[:15]:<15} | Month: {month} | Delta: {delta:+,.0f}")

                event_summary_data.append({
                    'Organization': org,
                    'Breach_Type': b_type,
                    'Month': month,
                    'Delta': delta
                })

        if len(pairs) >= 5:
            _, p_wilcoxon = wilcoxon([p[1] for p in pairs], [p[0] for p in pairs], alternative='greater')
            sweep_results.append({'delay': d, 'p_wilcoxon': p_wilcoxon, 'avg_delta': np.mean(deltas)})

    return pd.DataFrame(sweep_results), pd.DataFrame(event_summary_data)


def print_delta_summary(df, label):
    """Prints every event's victim delta, split into positive (IDT increased) and negative (IDT decreased or flat) groups."""
    if df.empty:
        return

    df = df.sort_values('Delta', ascending=False)
    pos = df[df['Delta'] > 0]
    neg = df[df['Delta'] < 0]

    print(f"\n\n{'='*90}")
    print(f"{label}: positive vs negative impact")
    print(f"{'='*90}")

    print(f"\nPositive delta, IDT increased (n={len(pos)})")
    print(f"{'Organization':<35} | {'Breach Type':<25} | {'Delta':<15}")
    print("-" * 85)
    for _, row in pos.iterrows():
        b_type = str(row['Breach_Type']) if pd.notna(row['Breach_Type']) else "Unknown"
        print(f"{row['Organization'][:33]:<35} | {b_type[:23]:<25} | +{row['Delta']:,.0f}")

    print(f"\nNegative delta, IDT decreased/flat (n={len(neg)})")
    print(f"{'Organization':<35} | {'Breach Type':<25} | {'Delta':<15}")
    print("-" * 85)
    for _, row in neg.iterrows():
        b_type = str(row['Breach_Type']) if pd.notna(row['Breach_Type']) else "Unknown"
        print(f"{row['Organization'][:33]:<35} | {b_type[:23]:<25} | {row['Delta']:,.0f}")
    print()


def plot_sweep_results(res_df, label, plots_dir):
    """Plots p-value and average victim change by discovery lag, on a dual-axis chart."""
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

    _, ax1 = plt.subplots(figsize=(16, 12))

    ax1.plot(res_df['delay'], res_df['p_wilcoxon'], marker='o', linewidth=5, markersize=10, label='P-value', color='tab:blue')
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=3, label='p=0.05 Threshold')

    ax1.set_xlabel('i Months After Breach Event')
    ax1.set_ylabel('P-value', color='tab:blue')
    ax1.set_ylim(0, 0.4)

    ax2 = ax1.twinx()
    ax2.bar(res_df['delay'], res_df['avg_delta'], alpha=0.25, color='tab:green', label='Avg IDT Change')
    ax2.set_ylabel('Avg Change in Number of Victims', color='tab:green')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    lines, labels = ax1.get_legend_handles_labels()
    bars, b_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + b_labels, loc='upper right')

    base_filename = os.path.join(plots_dir, f"longitudinal_sweep_{label.lower().replace(' ', '_')}")
    plt.tight_layout()
    plt.savefig(f"{base_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_filename}.pdf", bbox_inches='tight')
    plt.close()


def get_top_metadata(group):
    """For one month's breaches, returns the organization and breach type of the single largest one."""
    idx = group['total_affected'].idxmax()
    row = group.loc[idx]
    return pd.Series({
        'Top_Org': row['org_name'],
        'Breach_Type': row['breach_type'] if 'breach_type' in row else "Unknown"
    })


def main():
    print("\n--- Part 6: Wilcoxon Test for a Post-Mega-Breach Increase in IDT ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    sat_path = os.path.join(root_dir, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    idt_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    plots_dir = os.path.join(root_dir, 'plots/part6')
    os.makedirs(plots_dir, exist_ok=True)

    sat_df = pd.read_csv(sat_path)
    print(f"Loaded data from {sat_path}")
    idt_df = pd.read_csv(idt_path)
    print(f"Loaded data from {idt_path}")
    raw_csv = pd.read_csv(raw_path)
    print(f"Loaded data from {raw_path}")

    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'], format='mixed', errors='coerce')
    if raw_csv['reported_date'].isna().sum() > 0:
        raw_csv = raw_csv.dropna(subset=['reported_date'])
    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)

    meta_labels = raw_csv.groupby('Month_Str').apply(get_top_metadata).reset_index()

    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly['total_affected']

    master = pd.merge(idt_df[['Month_Str', 'Estimated_Victims']], raw_monthly, on='Month_Str', how='left')
    master = pd.merge(master, sat_df[['Month_Str', 'Um_Unique_Individuals']], on='Month_Str', how='left')
    master = pd.merge(master, meta_labels, on='Month_Str', how='left').fillna(0)

    raw_sweep, raw_events = analyze_longitudinal_sweep(master, 'total_affected_raw', 'Estimated_Victims', "Raw Data", RAW_THRESHOLD)
    sat_sweep, sat_events = analyze_longitudinal_sweep(master, 'Um_Unique_Individuals', 'Estimated_Victims', "Saturation Model", MODEL_THRESHOLD)

    plot_sweep_results(raw_sweep, "Raw Data", plots_dir)
    plot_sweep_results(sat_sweep, "Saturation Model", plots_dir)
    print(f"\nData saved to {plots_dir}")

    print_delta_summary(raw_events, "Raw Data")
    print_delta_summary(sat_events, "Saturation Model")

    print(f"\n{'='*60}")
    print("Case study deltas (discovery lag i=2)")
    print(f"{'='*60}")
    case_study_months = ['2009-01', '2013-12', '2016-12', '2017-09']
    for target in case_study_months:
        match = raw_events[raw_events['Month'] == target]
        if not match.empty:
            delta_val = match.iloc[0]['Delta']
            print(f"Breach Month: {target} | Median Delta Victims: {delta_val:+,.0f}")
        else:
            print(f"Breach Month: {target} | Data not found in results.")


if __name__ == "__main__":
    main()
