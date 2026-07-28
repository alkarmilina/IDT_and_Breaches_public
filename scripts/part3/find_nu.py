import pandas as pd
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part3/find_nu.py

"""
Part 3: Undisclosed Record Estimation (Finding nu)

Estimates the annual baseline for undisclosed PRC breach records (nu).
Each PRC breach is categorized by breach type, and a typical size
(median, excluding known megabreaches) is calculated for each type from
breaches with a known record count. The annual baseline is the sum,
across breach types, of the number of undisclosed incidents in that
type times its typical size, divided by the number of years in the
study period.

Setup:
Reads the PRC-HHS integrated dataset from PRC_fill_in_health.py.

Goal:
Produce the paper's derivation of nu. The result isn't written to a
file, it's meant to be read from the console and used as the
annual_nu_prc_only constant in PRC_Maine_NH_regression.py's final
imputation step.

Outputs:
None saved. Prints the derived annual nu baseline to the terminal.
"""


def main():
    print("\n--- Part 3: Undisclosed Record Estimation (Finding nu) ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    prc_data_path = os.path.join(root_dir, 'data/processed/part3/prc_hhs_integrated.csv')

    df_prc = pd.read_csv(prc_data_path)
    print(f"Loaded data from {prc_data_path}")

    # Typical (median) size per breach type, from known non-megabreach record counts.
    megabreaches = [12500000, 130000000, 77000000, 40000000, 56000000, 143000000]
    known_counts = df_prc[(df_prc['total_affected'] > 0) & (~df_prc['total_affected'].isin(megabreaches))]
    sector_medians = known_counts.groupby('breach_type')['total_affected'].median().to_dict()
    overall_median = known_counts['total_affected'].median()

    # PRC incidents with an undisclosed (zero) record count.
    pure_unknowns = df_prc[df_prc['total_affected'] == 0].copy()

    total_nu = 0
    for b_type in ['CARD', 'DISC', 'HACK', 'INSD', 'PHYS', 'PORT', 'STAT', 'UNKN']:
        count = len(pure_unknowns[pure_unknowns['breach_type'] == b_type])
        median_weight = sector_medians.get(b_type, overall_median)
        total_nu += count * median_weight

    total_years = 14
    annual_nu = total_nu / total_years

    print("\n" + "="*45)
    print(f"Total undisclosed volume ({total_years} years): {total_nu:,.0f}")
    print("-" * 45)
    print(f"Annual nu baseline:                    {annual_nu:,.0f}")
    print("="*45)


if __name__ == "__main__":
    main()
