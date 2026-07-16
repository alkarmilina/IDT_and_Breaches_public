import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# To run this script:
# python scripts/part2/hypothesis.py

"""
Statistical Inference: Core Project Hypothesis Testing (Section 5)
This script validates H1 (Digital Vulnerability by Age) and H2 (Victimization Odds).
H2 specifically targets the reproduction of Table 6: Logistic Regression.
"""

def main():
    # --- 1. Path Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    
    victim_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    all_respondents_path = os.path.join(root_dir, 'data/processed/part1/its_full.parquet')
    
    WEIGHT_COL = 'FINAL_ITS_WEIGHT'

    # =============================================================================
    # PART 1: DIGITAL VULNERABILITY (Section 5.2 - Table 5)
    # =============================================================================
    print("\n" + "="*60)
    print(" PART 1: TESTING H1 (AGE & THEFT METHOD)")
    print("="*60)

    try:
        df_victims = pd.read_parquet(victim_data_path)
        
        # Define Age Groups
        df_victims['AGE_GROUP'] = pd.cut(
            pd.to_numeric(df_victims['PERSON_AGE'], errors='coerce'),
            bins=[15, 29, 49, 64, np.inf],
            labels=['16-29', '30-49', '50-64', '65+'],
            right=True
        )

        # Define Digital vs Physical
        df_victims['THEFT_CAT'] = 'Other'
        df_victims.loc[df_victims['THEFT_METHOD'].isin([3, 4, 5]), 'THEFT_CAT'] = 'Digital'
        df_victims.loc[df_victims['THEFT_METHOD'] == 1, 'THEFT_CAT'] = 'Physical'

        # Filter for unambiguous cases
        df_filtered = df_victims[df_victims['THEFT_CAT'] != 'Other'].copy()
        df_filtered[WEIGHT_COL] = pd.to_numeric(df_filtered[WEIGHT_COL], errors='coerce')
        
        weighted_counts = df_filtered.groupby(['AGE_GROUP', 'THEFT_CAT'], observed=True)[WEIGHT_COL].sum().unstack()
        weighted_proportions = weighted_counts.div(weighted_counts.sum(axis=1), axis=0)

        print("\nTable 5: Weighted Proportions of Theft Type by Age Group:")
        print(weighted_proportions.round(3))
        
    except Exception as e:
        print(f"Error in Part 1: {e}")

    # =============================================================================
    # PART 2: VICTIMIZATION MODELS (REPRODUCING TABLE 6)
    # =============================================================================
    print("\n" + "="*60)
    print(" PART 2: TESTING H2 (TABLE 6 LOGISTIC REGRESSION)")
    print("="*60)

    try:
        df_all = pd.read_parquet(all_respondents_path)
        df_all['IS_VICTIM'] = df_all['IS_SUCCESSFUL_VICTIM'].astype(int)
        
        # --- Categorical Setup for Table 6 ---
        
        # 1. Age (Ref: 65+)
        df_all['AGE_REG'] = pd.cut(pd.to_numeric(df_all['PERSON_AGE'], errors='coerce'),
                                   bins=[15, 29, 49, 64, np.inf], labels=['16-29', '30-49', '50-64', '65+'])
        df_all['AGE_REG'] = pd.Categorical(df_all['AGE_REG'], categories=['65+', '16-29', '30-49', '50-64'])
        
        # 2. Education (Ref: < HS)
        # Mapping based on your harmonization dictionary (Table 28)
        edu_map = {1: '< HS', 2: '< HS', 3: '< HS', 4: 'HS Grad', 5: 'Some Coll', 6: 'Bach', 7: 'Grad'}
        df_all['EDU_REG'] = df_all['PERSON_EDUCATION'].map(edu_map)
        df_all['EDU_REG'] = pd.Categorical(df_all['EDU_REG'], categories=['< HS', 'HS Grad', 'Some Coll', 'Bach', 'Grad'])

        # 3. Income (Ref: Under $15,000)
        # Mapping based on your harmonization dictionary (Table 27)
        # Grouping 1-5 (<15k), 6-8 (15k-25k), 9-12 (25k-50k), 13 (50k-75k), 14 (75k+)
        inc_bins = [0, 5, 8, 12, 13, 14]
        inc_labels = ['< $15k', '$15k-$25k', '$25k-$50k', '$50k-$75k', '$75k+']
        df_all['INC_REG'] = pd.cut(df_all['HOUSEHOLD_INCOME'], bins=inc_bins, labels=inc_labels)
        df_all['INC_REG'] = pd.Categorical(df_all['INC_REG'], categories=['< $15k', '$15k-$25k', '$25k-$50k', '$50k-$75k', '$75k+'])

        # 4. Race (Ref: Other)
        race_map = {1: 'White', 2: 'Black', 4: 'Asian'}
        df_all['RACE_REG'] = df_all['PERSON_RACE'].map(race_map).fillna('Other')
        df_all['RACE_REG'] = pd.Categorical(df_all['RACE_REG'], categories=['Other', 'White', 'Black', 'Asian'])

        # Filter for complete cases including income
        df_reg = df_all[['IS_VICTIM', 'AGE_REG', 'EDU_REG', 'INC_REG', 'RACE_REG', WEIGHT_COL]].dropna()
        
        Y = df_reg['IS_VICTIM']
        W = pd.to_numeric(df_reg[WEIGHT_COL], errors='coerce')
        
        # Creating dummies (drop_first=True uses the first category in 'categories' list as the reference)
        X = pd.get_dummies(df_reg[['AGE_REG', 'EDU_REG', 'INC_REG', 'RACE_REG']], drop_first=True, dtype=int)
        X = sm.add_constant(X)

        print(f"Fitting Table 6 Model with N={len(df_reg):,} respondents...")
        model = sm.Logit(Y, X).fit(freq_weights=W, disp=0)
        
        print("\nTable 6: Logistic Regression Summary (Odds Ratios):")
        params = model.params
        results = pd.DataFrame({'OR': np.exp(params), 'P>|z|': model.pvalues})
        
        # Rounding and formatting to match paper's display
        print(results.round(3))

    except Exception as e:
        print(f"Error in Part 2: {e}")

if __name__ == '__main__':
    main()