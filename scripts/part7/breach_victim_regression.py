import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
RAW_THRESHOLD = 10000000     
MODEL_THRESHOLD = 5000000    
OBSERVATION_WINDOW = 6       
CHOSEN_LAG = 1               
EXCLUDE_YAHOO = False  

MEGA_SHOCKS = {
    '2008-05': 12500000, '2009-01': 130000000, '2011-04': 77000000,
    '2013-12': 40000000, '2014-09': 56000000, '2017-09': 143000000,
}

def get_event_metrics(df, record_col, idt_col, threshold, lag):
    """
    Matches the Wilcoxon script's logic: picks the first threshold breach
    and applies a 3-month cool down, with a manual swap for Equifax.
    """
    indices = df[df[record_col] >= threshold].index.tolist()
    event_indices = []
    
    if indices:
        # Standard first-come-first-serve initialization
        event_indices.append(indices[0])
        for i in range(1, len(indices)):
            current_month = df.loc[indices[i], 'Month_Str']
            last_selected_month = df.loc[event_indices[-1], 'Month_Str']
            
            # MANUAL SWAP: Prioritize Equifax (2017-09) over Atlantic (2017-06)
            if current_month == '2017-09' and last_selected_month == '2017-06':
                event_indices[-1] = indices[i]
                continue

            # Standard 3-month cool down (Matches Wilcoxon script)
            if indices[i] - event_indices[-1] > 3:
                event_indices.append(indices[i])
    
    points = []
    for idx in event_indices:
        if idx < OBSERVATION_WINDOW or idx + lag + OBSERVATION_WINDOW > len(df):
            continue
            
        breach_size = df.loc[idx, record_col]
        org_name = df.loc[idx, 'Top_Org']
        date_str = df.loc[idx, 'Month_Str']

        pre_window = df.loc[idx - OBSERVATION_WINDOW : idx - 1, idt_col]
        post_start = idx + lag
        post_window = df.loc[post_start : post_start + OBSERVATION_WINDOW - 1, idt_col]
        
        delta = post_window.median() - pre_window.median()
        
        points.append({
            'Event': org_name,
            'Date': date_str,
            'Breach_Size_X': breach_size,
            'Victim_Delta_Y': delta
        })
        
    return pd.DataFrame(points)

def plot_clean_scatter(data, label, color_code):
    plot_data = data.copy()
    
    if EXCLUDE_YAHOO:
        plot_data = plot_data[~plot_data['Event'].str.contains('Yahoo', case=False, na=False)].copy()

    # Clean junk entries
    plot_data = plot_data[~plot_data['Event'].str.contains('trwert', case=False, na=False)].copy()

    if len(plot_data) < 3:
        print(f"Not enough points for {label}")
        return

    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=plot_data, x='Breach_Size_X', y='Victim_Delta_Y', s=150, color=color_code, alpha=0.8)
    
    for i, row in plot_data.iterrows():
        x = row['Breach_Size_X']
        y = row['Victim_Delta_Y']
        name = row['Event']
        
        if 'Target' in name or 'Equifax' in name:
            plt.annotate(name, (x, y), xytext=(10, 10), 
                         textcoords='offset points', 
                         fontsize=14, weight='bold', color='red',
                         arrowprops=dict(arrowstyle='->', color='red'))
        else:
            plt.annotate(name, (x, y), xytext=(5, 5), 
                         textcoords='offset points', 
                         fontsize=9, alpha=0.6)
    
    plt.title(f"{label}", fontsize=24)
    
    # --- UPDATED LABELS HERE ---
    if label == "Saturation Model":
        plt.xlabel("Number of Unique Records Breached", fontsize=18)
    elif label == "Raw Data":
        plt.xlabel("Number of Raw Records Breached", fontsize=18)
    else:
        plt.xlabel(f"Breach Size ({label})", fontsize=18)
    # ---------------------------

    plt.ylabel("Victim Delta (Change in Estimated Victims)", fontsize=18)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.grid(True, alpha=0.3)
    
    output_path = f"plots/part7/scatter_labeled_{label.lower().replace(' ', '_')}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved synchronized plot to: {output_path}")

def main():
    root_dir = os.getcwd()
    sat_path = os.path.join(root_dir, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    idt_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')

    sat_df, idt_df, raw_csv = pd.read_csv(sat_path), pd.read_csv(idt_path), pd.read_csv(raw_path)
    
    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'], format='mixed', errors='coerce')
    raw_csv = raw_csv.dropna(subset=['reported_date'])
    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)

    def get_top_org(group): 
        return group.loc[group['total_affected'].idxmax(), 'org_name']
    org_labels = raw_csv.groupby('Month_Str').apply(get_top_org, include_groups=False).reset_index(name='Top_Org')

    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly.apply(
        lambda r: max(r['total_affected'], MEGA_SHOCKS.get(r['Month_Str'], 0)), axis=1
    )

    master = pd.merge(idt_df[['Month_Str', 'Estimated_Victims']], raw_monthly, on='Month_Str', how='left')
    master = pd.merge(master, sat_df[['Month_Str', 'Um_Unique_Individuals']], on='Month_Str', how='left')
    master = pd.merge(master, org_labels, on='Month_Str', how='left').fillna(0)

    # --- RAW DATA SECTION ---
    print("\n" + "="*80)
    print("DETAILED BREACH LIST: RAW DATA (X-AXIS VALUES)")
    print("="*80)
    raw_points = get_event_metrics(master, 'total_affected_raw', 'Estimated_Victims', RAW_THRESHOLD, CHOSEN_LAG)
    
    # Formatting for easy reading: Org Name, Date, and X-value
    for _, row in raw_points.iterrows():
        print(f"Org: {row['Event'][:30]:<30} | Date: {row['Date']} | Records (X): {row['Breach_Size_X']:,.0f}")
    
    print(f"\nTotal Raw Events Plotted: {len(raw_points)}")
    plot_clean_scatter(raw_points, "Raw Data", "blue")

    # --- SATURATION MODEL SECTION ---
    print("\n" + "="*80)
    print("DETAILED BREACH LIST: SATURATION MODEL (X-AXIS VALUES)")
    print("="*80)
    sat_points = get_event_metrics(master, 'Um_Unique_Individuals', 'Estimated_Victims', MODEL_THRESHOLD, CHOSEN_LAG)
    
    for _, row in sat_points.iterrows():
        print(f"Org: {row['Event'][:30]:<30} | Date: {row['Date']} | Unique Indiv (X): {row['Breach_Size_X']:,.0f}")
    
    print(f"\nTotal Saturation Events Plotted: {len(sat_points)}")
    plot_clean_scatter(sat_points, "Saturation Model", "green")

if __name__ == "__main__":
    main()