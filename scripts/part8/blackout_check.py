import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
MODEL_THRESHOLD = 5000000    # Matches Wilcoxon Saturation Threshold
OBSERVATION_WINDOW = 6
CHOSEN_LAG = 1               # CHANGED TO 1 (Matches the Wilcoxon 'd=1' print output)
ITS_DATA_PATH = 'data/processed/part1/its_victims.parquet' 

MEGA_SHOCKS = {
    '2008-05': 12500000, '2009-01': 130000000, '2011-04': 77000000,
    '2013-12': 40000000, '2014-09': 56000000, '2017-09': 143000000,
}

def get_raw_its_data():
    """Reconstructs the RAW monthly data before interpolation."""
    df = pd.read_parquet(ITS_DATA_PATH)
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
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Raw_Victims'}, inplace=True)
    return monthly

def check_raw_blackouts(master_df, raw_df):
    # Merge Raw Data onto Master
    df = pd.merge(master_df, raw_df, left_on='Month_Str', right_on='Year_Month', how='left')
    
    # DEFINE BLACKOUT
    df['Is_Blackout'] = df['Raw_Victims'].isna() | (df['Raw_Victims'] == 0)
    
    print(f"\n--- BLACKOUT ANALYSIS (Matches Wilcoxon Saturation Output) ---")
    print(f"Filtering by Unique Records >= {MODEL_THRESHOLD:,.0f}")
    print(f"Using Delay (Lag) = {CHOSEN_LAG} months")
    
    total_months = len(df)
    blackout_months = df['Is_Blackout'].sum()
    print(f"True 'Data Holes' (NaN/Zero): {blackout_months} ({blackout_months/total_months:.1%})")

    # --- SELECTION LOGIC (Matches Wilcoxon EXACTLY) ---
    indices = df[df['Um_Unique_Individuals'] >= MODEL_THRESHOLD].index.tolist()
    event_indices = []
    
    if indices:
        event_indices.append(indices[0])
        for i in range(1, len(indices)):
            current_month = df.loc[indices[i], 'Month_Str']
            last_selected_month = df.loc[event_indices[-1], 'Month_Str']
            
            # Prioritize Equifax (Manual Override)
            if current_month == '2017-09' and last_selected_month == '2017-06':
                event_indices[-1] = indices[i]
                continue
            
            # 3-Month Cool Down
            if indices[i] - event_indices[-1] > 3:
                event_indices.append(indices[i])

    results = []
    for idx in event_indices:
        if idx < OBSERVATION_WINDOW or idx + CHOSEN_LAG + OBSERVATION_WINDOW > len(df):
            continue
            
        org = df.loc[idx, 'Top_Org']
        b_type = df.loc[idx, 'Breach_Type'] # Now included
        month = df.loc[idx, 'Month_Str']
        
        pre_med = df.loc[idx - OBSERVATION_WINDOW : idx - 1, 'Estimated_Victims'].median()
        post_start = idx + CHOSEN_LAG
        post_window_indices = range(post_start, post_start + OBSERVATION_WINDOW)
        post_med = df.loc[post_start : post_start + OBSERVATION_WINDOW - 1, 'Estimated_Victims'].median()
        delta = post_med - pre_med
        
        blackout_count = df.loc[post_window_indices, 'Is_Blackout'].sum()
        
        results.append({
            'Organization': org,
            'Breach_Type': b_type,
            'Month': month,
            'Delta': delta,
            'Group': 'Positive' if delta > 0 else 'Negative',
            'Raw_Gaps': blackout_count
        })

    results_df = pd.DataFrame(results)
    
    print("\nSummary: Average 'Raw Data Gaps' in 6-month post-window:")
    print(results_df.groupby('Group')['Raw_Gaps'].mean())
    
    # Negative Group Table
    print("\nDetailed Breakdown (Negative Group):")
    neg_group = results_df[results_df['Group'] == 'Negative'].sort_values('Delta')
    print(f"{'Organization':<30} | {'Type':<10} | {'Delta':<10} | {'Raw Gaps':<15}")
    print("-" * 75)
    for _, row in neg_group.iterrows():
        b_type = str(row['Breach_Type'])[:10]
        print(f"{row['Organization'][:28]:<30} | {b_type:<10} | {row['Delta']:<10,.0f} | {row['Raw_Gaps']:<15}")

    # Positive Group Table
    print("\nDetailed Breakdown (Positive Group):")
    pos_group = results_df[results_df['Group'] == 'Positive'].sort_values('Delta', ascending=False)
    print(f"{'Organization':<30} | {'Type':<10} | {'Delta':<10} | {'Raw Gaps':<15}")
    print("-" * 75)
    for _, row in pos_group.iterrows():
        b_type = str(row['Breach_Type'])[:10]
        print(f"{row['Organization'][:28]:<30} | {b_type:<10} | {row['Delta']:<10,.0f} | {row['Raw_Gaps']:<15}")

    # --- PLOTTING SECTION ---
    plt.figure(figsize=(16, 10))
    plt.plot(df['Date'], df['Estimated_Victims'], color='gray', alpha=0.4, linewidth=2, label='ITS Number of IDT Victims')
    
    gaps = df[df['Is_Blackout']]
    plt.scatter(gaps['Date'], [0] * len(gaps), color='red', marker='x', s=40, label='ITS Data Missing', zorder=2)

    y_max = df['Estimated_Victims'].max() * 1.05
    
    neg_indices = [df[df['Month_Str'] == r['Month']].index[0] for _, r in neg_group.iterrows()]
    neg_dates = df.loc[neg_indices, 'Date']
    plt.scatter(neg_dates, [y_max]*len(neg_dates), color='orange', marker='v', s=150, label='Negative Delta', zorder=4)

    pos_indices = [df[df['Month_Str'] == r['Month']].index[0] for _, r in pos_group.iterrows()]
    pos_dates = df.loc[pos_indices, 'Date']
    plt.scatter(pos_dates, [y_max]*len(pos_dates), color='green', marker='^', s=150, label='Positive Delta', zorder=4)

    for _, row in results_df.iterrows():
        match = df[df['Month_Str'] == row['Month']]
        if not match.empty:
            date_obj = match.iloc[0]['Date']
            label_text = row['Organization'][:15]
            plt.text(date_obj, y_max * 1.02, label_text, 
                     rotation=45, fontsize=9, ha='left', va='bottom', rotation_mode='anchor')

    plt.title(f"Mega-Breaches vs. Survey Blackouts", fontsize=20)
    plt.ylabel("Estimated Victims", fontsize=15)
    plt.ylim(0, y_max * 1.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_path = 'plots/part8/blackout_check.png'
    os.makedirs('plots/part8', exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved labeled plot to {output_path}")

def main():
    root_dir = os.getcwd()
    sat_path = os.path.join(root_dir, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    idt_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')

    sat_df, idt_df, raw_csv = pd.read_csv(sat_path), pd.read_csv(idt_path), pd.read_csv(raw_path)
    
    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'], format='mixed', errors='coerce')
    raw_csv = raw_csv.dropna(subset=['reported_date'])
    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)

    # --- UPDATED: Get Top Org AND Breach Type ---
    def get_top_metadata(group):
        idx = group['total_affected'].idxmax()
        row = group.loc[idx]
        return pd.Series({
            'Top_Org': row['org_name'],
            'Breach_Type': row['breach_type'] if 'breach_type' in row else "Unknown"
        })

    # Apply grouping and merge back
    meta_labels = raw_csv.groupby('Month_Str').apply(get_top_metadata).reset_index()

    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly.apply(lambda r: max(r['total_affected'], MEGA_SHOCKS.get(r['Month_Str'], 0)), axis=1)

    master = pd.merge(idt_df[['Month_Str', 'Estimated_Victims']], raw_monthly, on='Month_Str', how='left')
    master = pd.merge(master, meta_labels, on='Month_Str', how='left').fillna(0) # Merge Metadata
    master = pd.merge(master, sat_df[['Month_Str', 'Um_Unique_Individuals']], on='Month_Str', how='left').fillna(0)
    
    master['Date'] = pd.to_datetime(master['Month_Str'])
    master = master.sort_values('Date').reset_index(drop=True)

    raw_its = get_raw_its_data()
    check_raw_blackouts(master, raw_its)

if __name__ == "__main__":
    main()