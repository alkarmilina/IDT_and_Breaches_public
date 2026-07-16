import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from thefuzz import fuzz, process

# To run: python scripts/part3/ME_NH_Cross_Validation.py

def clean_name(name):
    """Standardizes names for better fuzzy matching."""
    if pd.isna(name) or str(name).strip() == "": 
        return "unknown_org"
    cleaned = str(name).lower().strip().replace(".", "").replace(",", "")
    suffixes = [" inc", " corp", " llc", " ltd", " corporation", " lp", " group", " pa", " p.a."]
    for s in suffixes:
        if cleaned.endswith(s):
            cleaned = cleaned[:-len(s)].strip()
    return cleaned

def load_state_data(path):
    if not os.path.exists(path):
        print(f"Warning: Missing file {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
    
    # Map columns
    col_map = {
        'org': ['Organization', 'org_name', 'Entity Name', 'Company', 'DATA BREACH NOTICES', 'Company Whose Data Was Breached'],
        'affected': ['NH_Affected', 'maine_affected', 'Maine residents affected', 'Count', 'Number of Maine Residents Affected', 'Total number of Maine residents affected']
    }
    
    found_org = [c for c in col_map['org'] if c in df.columns]
    found_affected = [c for c in col_map['affected'] if c in df.columns]
    
    if found_org and found_affected:
        out = df[[found_org[0], found_affected[0]]].copy()
        out.columns = ['Organization', 'Affected']
        out['Affected'] = pd.to_numeric(out['Affected'], errors='coerce').fillna(0)
        out['clean_name'] = out['Organization'].apply(clean_name)
        return out[out['Affected'] > 0]
    return pd.DataFrame()

def main():
    root_dir = os.getcwd()
    nh_path = os.path.join(root_dir, 'data/raw/PRC_supplements/NH_Manual_Entry_Template.csv')
    me_path_1 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine1.xlsx')
    me_path_2 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine2.xlsx')
    
    output_plot = os.path.join(root_dir, "plots/part3/ME_NH_Validation_Fuzzy.png")
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    # 1. Load Data
    print("Loading State Data...")
    nh_df = load_state_data(nh_path)
    me_df = pd.concat([load_state_data(me_path_1), load_state_data(me_path_2)], ignore_index=True)
    
    # Deduplicate internally first (take max if same org appears twice in one state)
    nh_df = nh_df.sort_values('Affected', ascending=False).drop_duplicates('clean_name')
    me_df = me_df.sort_values('Affected', ascending=False).drop_duplicates('clean_name')

    print(f"Unique NH Records: {len(nh_df)}")
    print(f"Unique ME Records: {len(me_df)}")

    # 2. Fuzzy Matching Logic
    print("\nRunning Fuzzy Matching (Threshold > 85)...")
    matches = []
    
    # Iterate through NH (smaller set) and look for matches in ME (larger set)
    me_names = me_df['clean_name'].tolist()
    
    print(f"\n{'-'*85}")
    print(f"{'Organization (NH vs ME Match)':<50} | {'NH Count':<10} | {'ME Count':<10}")
    print(f"{'-'*85}")

    for idx, nh_row in nh_df.iterrows():
        # Skip very generic names to avoid false positives
        if len(nh_row['clean_name']) < 4: continue
            
        # Extract best match from Maine names
        best_match = process.extractOne(nh_row['clean_name'], me_names, scorer=fuzz.token_sort_ratio)
        
        if best_match and best_match[1] >= 85:
            # Get the corresponding row from Maine DF
            me_row = me_df[me_df['clean_name'] == best_match[0]].iloc[0]
            
            matches.append({
                'Organization_NH': nh_row['Organization'],
                'Organization_ME': me_row['Organization'],
                'Affected_NH': nh_row['Affected'],
                'Affected_ME': me_row['Affected']
            })
            
            print(f"{nh_row['Organization'][:20]} / {me_row['Organization'][:20]:<27} | {nh_row['Affected']:<10,.0f} | {me_row['Affected']:<10,.0f}")

    # 3. Create DataFrame from matches
    merged = pd.DataFrame(matches)
    
    if len(merged) < 2:
        print("\nNot enough matches found for regression.")
        return

    print(f"\nFound {len(merged)} overlapping incidents.")

    # 4. Regression & Stats
    x = merged['Affected_ME']
    y = merged['Affected_NH']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Expected Slope: NH Pop (1.40M) / ME Pop (1.37M) ~= 1.02
    expected_slope = 1.02

    print("\n--- CROSS-STATE VALIDATION RESULTS ---")
    print(f"Correlation (R): {r_value:.4f}")
    print(f"R-Squared:       {r_value**2:.4f}")
    print(f"Slope (NH/ME):   {slope:.4f} (Expected ~{expected_slope})")
    print(f"P-Value:         {p_value:.4e}")

    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x, y=y, s=100, alpha=0.7, color='purple')
    
    # Plot Fit Line
    line_x = np.array([0, x.max()])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', linestyle='--', linewidth=2, label=f'Fit: y={slope:.2f}x + {intercept:.0f}')
    
    # Plot Expected Identity Line
    plt.plot(line_x, line_x * expected_slope, color='gray', linestyle=':', label='Expected Pop Ratio (1.02x)')
    
    plt.title(f"Cross-State Validation (Fuzzy Match)\n(n={len(merged)} overlapping breaches)", fontsize=16)
    plt.xlabel("Maine Victims", fontsize=14)
    plt.ylabel("New Hampshire Victims", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale if range is huge
    if x.max() > 100000 or y.max() > 100000:
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Cross-State Validation (Log Scale)\n(n={len(merged)} overlapping breaches)", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Saved validation plot to {output_plot}")

if __name__ == "__main__":
    main()