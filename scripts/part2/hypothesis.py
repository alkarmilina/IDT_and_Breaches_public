import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/hypothesis.py

"""
Part 2: Hypothesis Testing on Age, Education, Income, and Race

Tests two hypotheses about why some groups are overrepresented among
identity theft victims. H1 tests whether the youngest age group,
"digital natives," is less vulnerable to digital theft methods and more
vulnerable to physical theft methods than every older age group. H2
tests whether age, education, income, and race are independently
associated with the odds of being a victim at all, using a weighted
logistic regression over all survey respondents, not just victims.

Setup:
H1 reads the victims-only dataset from part1. H2 reads the all-respondents
dataset from part1, since it needs non-victims to estimate victimization
odds. Age and education are grouped the same way as the paper's other
demographic tables. Race here is grouped into White, Black, Asian, and
Other only, with no separate Hispanic category, matching how the paper's
regression models treat race.

Goal:
Produce the paper's weighted proportion test of theft method by age
group (H1), and its logistic regression of victimization odds on age,
education, income, and race (H2).

Outputs:
Both results are printed to the terminal. Neither is saved to a file.
"""


def main():
    print("\n--- Part 2: Hypothesis Testing on Age, Education, Income, and Race ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    victim_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    all_respondents_path = os.path.join(root_dir, 'data/processed/part1/its_full.parquet')

    weight_col = 'FINAL_ITS_WEIGHT'

    print("\nH1: age and theft method")

    try:
        df_victims = pd.read_parquet(victim_data_path)
        print(f"Loaded data from {victim_data_path}")

        df_victims['AGE_GROUP'] = pd.cut(
            pd.to_numeric(df_victims['PERSON_AGE'], errors='coerce'),
            bins=[15, 29, 49, 64, np.inf],
            labels=['16-29', '30-49', '50-64', '65+'],
            right=True
        )

        # Digital: hacking, scam/phishing, or a data breach. Physical: a lost or stolen physical
        # item. "Stolen during a transaction" and "Other" are ambiguous and excluded.
        df_victims['THEFT_CAT'] = 'Other'
        df_victims.loc[df_victims['THEFT_METHOD'].isin([3, 4, 5]), 'THEFT_CAT'] = 'Digital'
        df_victims.loc[df_victims['THEFT_METHOD'] == 1, 'THEFT_CAT'] = 'Physical'

        df_filtered = df_victims[df_victims['THEFT_CAT'] != 'Other'].copy()
        df_filtered[weight_col] = pd.to_numeric(df_filtered[weight_col], errors='coerce')

        weighted_counts = df_filtered.groupby(['AGE_GROUP', 'THEFT_CAT'], observed=True)[weight_col].sum().unstack()
        weighted_proportions = weighted_counts.div(weighted_counts.sum(axis=1), axis=0)

        print("\nWeighted proportion of digital vs. physical theft by age group:")
        print(weighted_proportions.round(3))

    except Exception as e:
        print(f"Error in H1: {e}")

    print("\nH2: logistic regression of victimization odds")

    try:
        df_all = pd.read_parquet(all_respondents_path)
        print(f"Loaded data from {all_respondents_path}")
        df_all['IS_VICTIM'] = df_all['IS_SUCCESSFUL_VICTIM'].astype(int)

        # Age (Ref: 65+)
        df_all['AGE_REG'] = pd.cut(pd.to_numeric(df_all['PERSON_AGE'], errors='coerce'),
                                   bins=[15, 29, 49, 64, np.inf], labels=['16-29', '30-49', '50-64', '65+'])
        df_all['AGE_REG'] = pd.Categorical(df_all['AGE_REG'], categories=['65+', '16-29', '30-49', '50-64'])

        # Education (Ref: < HS)
        edu_map = {1: '< HS', 2: '< HS', 3: '< HS', 4: 'HS Grad', 5: 'Some Coll', 6: 'Bach', 7: 'Grad'}
        df_all['EDU_REG'] = df_all['PERSON_EDUCATION'].map(edu_map)
        df_all['EDU_REG'] = pd.Categorical(df_all['EDU_REG'], categories=['< HS', 'HS Grad', 'Some Coll', 'Bach', 'Grad'])

        # Income (Ref: Under $15,000). Codes 1-5 are under $15k, 6-8 are $15k-$25k,
        # 9-12 are $25k-$50k, 13 is $50k-$75k, 14 is $75k and over.
        inc_bins = [0, 5, 8, 12, 13, 14]
        inc_labels = ['< $15k', '$15k-$25k', '$25k-$50k', '$50k-$75k', '$75k+']
        df_all['INC_REG'] = pd.cut(df_all['HOUSEHOLD_INCOME'], bins=inc_bins, labels=inc_labels)
        df_all['INC_REG'] = pd.Categorical(df_all['INC_REG'], categories=inc_labels)

        # Race (Ref: Other)
        race_map = {1: 'White', 2: 'Black', 4: 'Asian'}
        df_all['RACE_REG'] = df_all['PERSON_RACE'].map(race_map).fillna('Other')
        df_all['RACE_REG'] = pd.Categorical(df_all['RACE_REG'], categories=['Other', 'White', 'Black', 'Asian'])

        df_reg = df_all[['IS_VICTIM', 'AGE_REG', 'EDU_REG', 'INC_REG', 'RACE_REG', weight_col]].dropna()

        Y = df_reg['IS_VICTIM']
        W = pd.to_numeric(df_reg[weight_col], errors='coerce')

        # drop_first=True uses the first category in each variable's categories list as the reference.
        X = pd.get_dummies(df_reg[['AGE_REG', 'EDU_REG', 'INC_REG', 'RACE_REG']], drop_first=True, dtype=int)
        X = sm.add_constant(X)

        print(f"Fitting logistic regression with N={len(df_reg):,} respondents...")
        model = sm.Logit(Y, X).fit(freq_weights=W, disp=0)

        print("\nLogistic regression summary (odds ratios):")
        params = model.params
        results = pd.DataFrame({'OR': np.exp(params), 'P>|z|': model.pvalues})
        print(results.round(3))

    except Exception as e:
        print(f"Error in H2: {e}")


if __name__ == '__main__':
    main()
