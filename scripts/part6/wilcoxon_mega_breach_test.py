import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os

# To run: python scripts/part6/wilcoxon_mega_breach_test.py

# --- 1. CONFIGURATION ---
RAW_THRESHOLD = 10000000     # 10 Million for Raw Data
MODEL_THRESHOLD = 5000000    # 5 Million for Saturated (Unique) Data
OBSERVATION_WINDOW = 6 
DELAY_RANGE = range(0, 9)

MEGA_SHOCKS = {
    '2008-05': 12500000, '2009-01': 130000000, '2011-04': 77000000,
    '2013-12': 40000000, '2014-09': 56000000, '2017-09': 143000000,
}

def analyze_longitudinal_sweep(df, record_col, idt_col, label, threshold):
    # Get all indices that meet the threshold
    indices = df[df[record_col] >= threshold].index.tolist()
    event_indices = []
    
    if indices:
        event_indices.append(indices[0])
        for i in range(1, len(indices)):
            # Check if this index is for Equifax (2017-09)
            current_month = df.loc[indices[i], 'Month_Str']
            last_selected_month = df.loc[event_indices[-1], 'Month_Str']
            
            if current_month == '2017-09' and last_selected_month == '2017-06':
                event_indices[-1] = indices[i]
                continue

            # Standard 3-month cool down for everything else
            if indices[i] - event_indices[-1] > 3:
                event_indices.append(indices[i])

    print(f"\n{'-'*30}\nDETAILED EVENT BREAKDOWN: {label}\n(Threshold: {threshold:,.0f})\n{'-'*30}")
    print(f"Total Major Events Identified: {len(event_indices)}")
    
    sweep_results = []
    event_summary_data = [] # Collect data for the final report
    
    for d in DELAY_RANGE:
        pairs = []
        deltas = []
        
        # We only collect summary data during one pass (d=2 is used here for case study consistency)
        collect_summary = (d == 2)
        
        for idx in event_indices:
            if idx < OBSERVATION_WINDOW or idx + d + OBSERVATION_WINDOW > len(df):
                continue
            
            pre_med = df.loc[idx - OBSERVATION_WINDOW : idx - 1, idt_col].median()
            post_start = idx + d
            post_med = df.loc[post_start : post_start + OBSERVATION_WINDOW - 1, idt_col].median()
            delta = post_med - pre_med
            
            pairs.append((pre_med, post_med))
            deltas.append(delta)
            
            if collect_summary:
                month = df.loc[idx, 'Month_Str']
                org = df.loc[idx, 'Top_Org'] if 'Top_Org' in df.columns else "Unknown"
                b_type = df.loc[idx, 'Breach_Type'] if 'Breach_Type' in df.columns else "Unknown"
                
                # CLEANER PRINT FORMAT
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
    """Prints the separated Positive vs Negative lists for the advisor."""
    if df.empty:
        return

    # Sort by magnitude of impact
    df = df.sort_values('Delta', ascending=False)
    
    pos = df[df['Delta'] > 0]
    neg = df[df['Delta'] < 0]
    
    print(f"\n\n{'='*90}")
    print(f"FINAL SUMMARY REPORT: {label} (Positive vs Negative Impact Groups)")
    print(f"{'='*90}")
    
    print(f"\n>>> GROUP A: POSITIVE DELTA (Identity Theft INCREASED) [n={len(pos)}]")
    print(f"{'Organization':<35} | {'Breach Type':<25} | {'Delta':<15}")
    print("-" * 85)
    for _, row in pos.iterrows():
        b_type = str(row['Breach_Type']) if pd.notna(row['Breach_Type']) else "Unknown"
        print(f"{row['Organization'][:33]:<35} | {b_type[:23]:<25} | +{row['Delta']:,.0f}")

    print(f"\n>>> GROUP B: NEGATIVE DELTA (Identity Theft DECREASED/FLAT) [n={len(neg)}]")
    print(f"{'Organization':<35} | {'Breach Type':<25} | {'Delta':<15}")
    print("-" * 85)
    for _, row in neg.iterrows():
        b_type = str(row['Breach_Type']) if pd.notna(row['Breach_Type']) else "Unknown"
        print(f"{row['Organization'][:33]:<35} | {b_type[:23]:<25} | {row['Delta']:,.0f}")
    print("\n")

def plot_sweep_results(res_df, label):
    os.makedirs('plots/part6', exist_ok=True)
    
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
    
    fig, ax1 = plt.subplots(figsize=(16, 12)) 

    # P-value trends
    ax1.plot(res_df['delay'], res_df['p_wilcoxon'], marker='o', linewidth=5, markersize=10, label='P-value', color='tab:blue')
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=3, label='p=0.05 Threshold')
    
    ax1.set_xlabel('i Months After Breach Event')
    ax1.set_ylabel('P-value', color='tab:blue')
    ax1.set_ylim(0, 0.4)

    # Avg Change in IDT
    ax2 = ax1.twinx()
    ax2.bar(res_df['delay'], res_df['avg_delta'], alpha=0.25, color='tab:green', label='Avg IDT Change')
    ax2.set_ylabel('Avg Change in Number of Victims', color='tab:green')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    lines, labels = ax1.get_legend_handles_labels()
    bars, b_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + b_labels, loc='upper right')
    
    base_filename = f"plots/part6/longitudinal_sweep_{label.lower().replace(' ', '_')}"
    plt.tight_layout()
    plt.savefig(f"{base_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_filename}.pdf", bbox_inches='tight')
    plt.close()

def main():
    root_dir = os.getcwd()
    sat_path = os.path.join(root_dir, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    idt_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')

    sat_df, idt_df, raw_csv = pd.read_csv(sat_path), pd.read_csv(idt_path), pd.read_csv(raw_path)
    
    # --- FIX: Handle mixed date formats ---
    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'], format='mixed', errors='coerce')
    
    if raw_csv['reported_date'].isna().sum() > 0:
        raw_csv = raw_csv.dropna(subset=['reported_date'])

    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)
    
    # --- EXTRACT TOP ORG AND BREACH TYPE PER MONTH ---
    def get_top_metadata(group):
        idx = group['total_affected'].idxmax()
        row = group.loc[idx]
        return pd.Series({
            'Top_Org': row['org_name'],
            'Breach_Type': row['breach_type'] if 'breach_type' in row else "Unknown"
        })

    # Apply grouping and merge back
    meta_labels = raw_csv.groupby('Month_Str').apply(get_top_metadata).reset_index()
    
    # --- AGGREGATE MONTHLY TOTALS ---
    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly.apply(lambda r: max(r['total_affected'], MEGA_SHOCKS.get(r['Month_Str'], 0)), axis=1)

    # --- MERGE EVERYTHING ---
    master = pd.merge(idt_df[['Month_Str', 'Estimated_Victims']], raw_monthly, on='Month_Str', how='left')
    master = pd.merge(master, sat_df[['Month_Str', 'Um_Unique_Individuals']], on='Month_Str', how='left')
    master = pd.merge(master, meta_labels, on='Month_Str', how='left').fillna(0) # Merge Top_Org & Breach_Type

    # --- RUN ANALYSIS ---
    raw_sweep, raw_events = analyze_longitudinal_sweep(master, 'total_affected_raw', 'Estimated_Victims', "Raw Data", RAW_THRESHOLD)
    sat_sweep, sat_events = analyze_longitudinal_sweep(master, 'Um_Unique_Individuals', 'Estimated_Victims', "Saturation Model", MODEL_THRESHOLD)

    # --- PLOTS ---
    plot_sweep_results(raw_sweep, "Raw Data")
    plot_sweep_results(sat_sweep, "Saturation Model")

    # --- FINAL PRINTS FOR ADVISOR ---
    print_delta_summary(raw_events, "Raw Data")
    print_delta_summary(sat_events, "Saturation Model")

    # --- SPECIFIC DELTA EXTRACTIONS FOR LOWER BOUND ---
    print(f"\n{'='*60}")
    print("SPECIFIC CASE STUDY DELTAS (Discovery Lag i=2)")
    print(f"{'='*60}")
    case_study_months = ['2009-01', '2013-12', '2017-09']
    for target in case_study_months:
        match = raw_events[raw_events['Month'] == target]
        if not match.empty:
            delta_val = match.iloc[0]['Delta']
            print(f"Breach Month: {target} | Median Delta Victims: {delta_val:+,.0f}")
        else:
            print(f"Breach Month: {target} | Data not found in results.")

if __name__ == "__main__":
    main()