import pandas as pd
import os

# python scripts/part5/calculate_megabreach_cost.py

# --- 1. CONFIGURATION ---
SOCIAL_COST_PATH = 'data/processed/part2/social_cost/social_cost_analysis.csv'
# Path to the CSV exported from the monthly_data_breach_conversion.py script
MERGED_CONV_PATH = 'data/processed/part5/conversion_data.csv'

OUTPUT_DIR = 'data/processed/part5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mega-breach Facts used in the paper
BREACH_DATA = {
    "Target": {
        "month": "2013-12",
        "records": 40_000_000,
        "settlement": 18_500_000,
        "cost_year": 2012 # Uses 2012 survey period as proxy per Section 7.1
    },
    "Equifax": {
        "month": "2017-09",
        "records": 143_000_000,
        "settlement": 700_000_000,
        "cost_year": 2016 # Uses 2016 survey period as proxy per Section 7.1
    }
}

def main():
    print("Integrating Social Cost and Conversion data...")
    
    # 1. Load Social Cost (Part 2 output)
    if not os.path.exists(SOCIAL_COST_PATH):
        print(f"ERROR: {SOCIAL_COST_PATH} not found. Run social_cost scripts first.")
        return
    cost_df = pd.read_csv(SOCIAL_COST_PATH)
    
    # 2. Load Conversion Data (Exported from Part 5 monthly script)
    if not os.path.exists(MERGED_CONV_PATH):
        print(f"ERROR: {MERGED_CONV_PATH} not found. Run monthly_data_breach_conversion.py first.")
        return
    conv_df = pd.read_csv(MERGED_CONV_PATH)

    results = []

    for name, info in BREACH_DATA.items():
        # A. Extract Conversion Rate (Smoothed 6-month average)
        match = conv_df[conv_df['Month_Str'] == info['month']]
        if match.empty:
            print(f"Warning: Could not find conversion rate for {info['month']}")
            continue
            
        rate_per_100k = match.iloc[0]['Rate_Smoothed']
        prob_per_record = rate_per_100k / 100000
        
        # B. Extract Social Cost per Victim for the representative year
        cost_match = cost_df[cost_df['Year'] == info['cost_year']]
        cost_per_victim = cost_match.iloc[0]['Total Social Cost per Victim ($)']
        
        # C. Perform Calculations 
        # Social Cost per Record = Social Cost per Victim * Prob per Record
        social_cost_per_record = cost_per_victim * prob_per_record
        total_social_harm = social_cost_per_record * info['records']
        
        results.append({
            "Breach": name,
            "Month": info['month'],
            "Records Exposed": info['records'],
            "Conv Rate (per 100k)": rate_per_100k,
            "Prob per Record": prob_per_record,
            "Social Cost/Victim ($)": cost_per_victim,
            "Social Cost/Record ($)": social_cost_per_record,
            "Total Social Harm ($B)": total_social_harm / 1e9,
            "Corporate Settlement ($M)": info['settlement'] / 1e6,
            "Harm-to-Settlement Ratio": total_social_harm / info['settlement']
        })

    # Output Results to CSV
    final_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'megabreach_social_cost_comparison.csv')
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("MEGA-BREACH SOCIAL COST ANALYSIS")
    for r in results:
        print(f"\n{r['Breach']} Analysis ({r['Month']}):")
        # Added requested intermediate values
        print(f"  1. 6-Mo Smoothed Conv Rate: {r['Conv Rate (per 100k)']:,.2f} victims per 100k records")
        print(f"  2. Prob. of IDT per record: {r['Prob per Record']:.4f}")
        print(f"  3. Social Cost per Victim:  ${r['Social Cost/Victim ($)']:,.2f}")
        
        print(f"  --- Resulting Multipliers ---")
        print(f"  - Social Cost per Record:   ${r['Social Cost/Record ($)']:.2f}")
        print(f"  - Total National Harm:      ${r['Total Social Harm ($B)']:.2f} Billion")
        print(f"  - Reported Settlement:      ${r['Corporate Settlement ($M)']:.2f} Million")
        print(f"  - Burden Ratio:             {r['Harm-to-Settlement Ratio']:,.0f}x higher than settlement")
    print("="*60)

if __name__ == "__main__":
    main()