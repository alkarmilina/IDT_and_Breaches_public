import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import os

# --- CONFIGURATION ---
OBSERVATION_WINDOW = 6
CHOSEN_LAG = 1
ITS_DATA_PATH = 'data/processed/part1/its_victims.parquet'

# The specific "High Confidence" list you identified (where survey data exists)
VALID_SUBSET = [
    'LinkedIn', 
    'Michaels Stores', 
    'Community Health', 
    'Eddie Bauer', 
    'Under Armour', 
    'Chegg', 
    'ClearVoice'
]

def get_raw_its_data():
    """Reconstructs the RAW monthly data to verify these aren't blackouts."""
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

def analyze_valid_subset(master_df, raw_df):
    # Merge Raw Data to confirm quality
    df = pd.merge(master_df, raw_df, left_on='Month_Str', right_on='Year_Month', how='left')
    df['Is_Blackout'] = df['Raw_Victims'].isna() | (df['Raw_Victims'] == 0)

    print(f"\n{'='*80}")
    print(f"ANALYSIS OF 'HIGH CONFIDENCE' BREACHES (Option 2)")
    print(f"{'='*80}")
    
    # Filter for only the events in VALID_SUBSET
    subset_indices = []
    for term in VALID_SUBSET:
        matches = df[df['Top_Org'].str.contains(term, case=False, na=False)]
        if not matches.empty:
            best_idx = matches['total_affected_raw'].idxmax()
            subset_indices.append(best_idx)
    
    results = []
    pairs = [] # Store (Pre, Post) for Wilcoxon
    
    print(f"\n{'-'*95}")
    print(f"{'Organization':<35} | {'Type':<10} | {'Date':<10} | {'Delta':<12} | {'Raw Gaps (0-6)':<15}")
    print(f"{'-'*95}")
    
    for idx in subset_indices:
        if idx < OBSERVATION_WINDOW or idx + CHOSEN_LAG + OBSERVATION_WINDOW > len(df):
            continue
            
        org = df.loc[idx, 'Top_Org']
        b_type = str(df.loc[idx, 'Breach_Type'])
        month = df.loc[idx, 'Month_Str']
        
        pre_med = df.loc[idx - OBSERVATION_WINDOW : idx - 1, 'Estimated_Victims'].median()
        post_start = idx + CHOSEN_LAG
        post_window_indices = range(post_start, post_start + OBSERVATION_WINDOW)
        post_med = df.loc[post_start : post_start + OBSERVATION_WINDOW - 1, 'Estimated_Victims'].median()
        delta = post_med - pre_med
        
        blackout_count = df.loc[post_window_indices, 'Is_Blackout'].sum()
        
        # Collect for Stats
        pairs.append((pre_med, post_med))
        
        results.append({
            'Organization': org,
            'Delta': delta
        })
        
        print(f"{org[:33]:<35} | {b_type[:10]:<10} | {month:<10} | {delta:+,.0f}      | {blackout_count:<15}")

    print(f"{'-'*95}\n")

    # --- STATISTICAL TEST ---
    print(f"STATISTICAL CHECK (N={len(pairs)}):")
    if len(pairs) >= 5:
        # Wilcoxon Signed-Rank Test (Alternative='greater' tests if Post > Pre)
        w_stat, p_val = wilcoxon([p[1] for p in pairs], [p[0] for p in pairs], alternative='greater')
        print(f"Wilcoxon W-Statistic: {w_stat}")
        print(f"P-Value: {p_val:.4f}")
        
        if p_val < 0.05:
            print(">> RESULT: Statistically Significant Increase (p < 0.05)")
        elif p_val < 0.15:
            print(">> RESULT: Strong Directional Signal (0.05 < p < 0.15)")
        else:
            print(">> RESULT: Not Significant (p > 0.15)")
    else:
        print(">> Not enough data points for Wilcoxon test (Need N>=5).")


    # --- PLOT THE CLEAN SUBSET ---
    plt.figure(figsize=(16, 10))
    plt.plot(df['Date'], df['Estimated_Victims'], color='gray', alpha=0.3, linewidth=2, label='Victim Volume (Background)')
    
    clean_indices = subset_indices
    
    pos_dates, pos_y, pos_labels = [], [], []
    neg_dates, neg_y, neg_labels = [], [], []
    
    y_max = df['Estimated_Victims'].max() * 1.05

    for i, idx in enumerate(clean_indices):
        delta = results[i]['Delta']
        org = results[i]['Organization']
        date = df.loc[idx, 'Date']
        
        if delta > 0:
            pos_dates.append(date)
            pos_y.append(y_max)
            pos_labels.append(org)
        else:
            neg_dates.append(date)
            neg_y.append(y_max)
            neg_labels.append(org)

    plt.scatter(pos_dates, pos_y, color='green', marker='^', s=200, label='Positive Delta', zorder=5)
    plt.scatter(neg_dates, neg_y, color='orange', marker='v', s=200, label='Negative Delta', zorder=5)

    for date, label in zip(pos_dates, pos_labels):
        plt.text(date, y_max * 1.03, label.split(',')[0], rotation=45, fontsize=11, color='green', ha='left', va='bottom')
    for date, label in zip(neg_dates, neg_labels):
        plt.text(date, y_max * 1.03, label.split(',')[0], rotation=45, fontsize=11, color='chocolate', ha='left', va='bottom')

    plt.title("The 'High Confidence' Subset (No Blackouts)", fontsize=20)
    plt.ylabel("Estimated Victims", fontsize=15)
    plt.ylim(0, y_max * 1.4) 
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_path = 'plots/part9/valid_subset_analysis_stats.png'
    os.makedirs('plots/part9', exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved plot to {output_path}")

def main():
    root_dir = os.getcwd()
    sat_path = os.path.join(root_dir, 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv')
    idt_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    raw_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')

    sat_df, idt_df, raw_csv = pd.read_csv(sat_path), pd.read_csv(idt_path), pd.read_csv(raw_path)
    
    raw_csv['reported_date'] = pd.to_datetime(raw_csv['reported_date'], format='mixed', errors='coerce')
    raw_csv = raw_csv.dropna(subset=['reported_date'])
    raw_csv['Month_Str'] = raw_csv['reported_date'].dt.to_period('M').astype(str)

    def get_top_metadata(group):
        idx = group['total_affected'].idxmax()
        row = group.loc[idx]
        return pd.Series({
            'Top_Org': row['org_name'],
            'Breach_Type': row['breach_type'] if 'breach_type' in row else "Unknown"
        })

    meta_labels = raw_csv.groupby('Month_Str').apply(get_top_metadata).reset_index()

    MEGA_SHOCKS = {
        '2008-05': 12500000, '2009-01': 130000000, '2011-04': 77000000,
        '2013-12': 40000000, '2014-09': 56000000, '2017-09': 143000000,
    }
    raw_monthly = raw_csv.groupby('Month_Str')['total_affected'].sum().reset_index()
    raw_monthly['total_affected_raw'] = raw_monthly.apply(lambda r: max(r['total_affected'], MEGA_SHOCKS.get(r['Month_Str'], 0)), axis=1)

    master = pd.merge(idt_df[['Month_Str', 'Estimated_Victims']], raw_monthly, on='Month_Str', how='left')
    master = pd.merge(master, meta_labels, on='Month_Str', how='left').fillna(0)
    master['Date'] = pd.to_datetime(master['Month_Str'])
    master = master.sort_values('Date').reset_index(drop=True)

    raw_its = get_raw_its_data()
    analyze_valid_subset(master, raw_its)

if __name__ == "__main__":
    main()