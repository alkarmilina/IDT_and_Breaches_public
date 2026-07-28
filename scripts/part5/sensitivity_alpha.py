import pandas as pd
import numpy as np
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part5/sensitivity_alpha.py

"""
Part 5: Sensitivity to the Discount Factor Alpha

Checks whether the choice of monthly discount factor alpha materially
affects the conversion model's results, by repeating the full pipeline
(computing D_t, fitting the log-quadratic conversion rate, and
evaluating the upper-bound social cost for the four mega-breach case
studies) at alpha = 0.70, 0.80, and 0.90.

Setup:
Reads monthly victim counts from the ITS victims-only dataset from
part1, monthly raw records exposed from the augmented PRC dataset from
part3, and social cost per victim by year from part2. Heartland's 2009
breach is added manually, matching monthly_data_breach_conversion_beta.py,
since it predates the PRC/HHS/state augmentation sources. Settlement
amounts for the ratio columns are the same as in the paper's mega-breach
case study comparison: $107M (Heartland), $18.5M (Target), $117.5M
(Yahoo), $700M (Equifax).

Goal:
Produce the paper's alpha sensitivity table, fit coefficients and
upper-bound social cost estimates for each case study breach, at each
alpha value.

Outputs:
- data/processed/part5/sensitivity_alpha.csv: fit coefficients by alpha.
- data/processed/part5/sensitivity_alpha_case_studies.csv: upper-bound
  social cost and settlement ratio for each case study breach, by alpha.
"""

SMOOTH_WINDOW = 6
CUTOFF_DATE = '2022-01-01'
START_DATE = '2008-01-01'
ALPHA_VALUES = [0.70, 0.80, 0.90]
DISC_LAG = 2  # Discovery lag in months, matches calculate_megabreach_cost_extended.py.

# Case study breaches, matching the paper's mega-breach comparison. Settlement amounts are
# Heartland $107M, Target $18.5M, Yahoo $117.5M, Equifax $700M.
CASE_STUDY_BREACHES = {
    'Heartland': {'org_name': 'Heartland', 'date': '2009-01-20', 'settlement': 107_000_000},
    'Target': {'org_name': 'Target Corp.', 'date': '2013-12-19', 'settlement': 18_500_000},
    'Yahoo': {'org_name': 'Yahoo! Inc.', 'date': '2016-12-14', 'settlement': 117_500_000},
    'Equifax': {'org_name': 'Equifax', 'date': '2017-09-07', 'settlement': 700_000_000},
}


def load_its_monthly(its_data_path):
    """Monthly weighted victim counts from the ITS victims-only dataset, by discovery month."""
    df = pd.read_parquet(its_data_path)
    q2m = {1.0: 2, 2.0: 5, 3.0: 8, 4.0: 11}
    df['DISCOVERY_MONTH'] = pd.to_numeric(df['DISCOVERY_MONTH'], errors='coerce')
    df['INTERVIEW_QUARTER'] = pd.to_numeric(df['INTERVIEW_QUARTER'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    mask = df['DISCOVERY_MONTH'].isnull()
    df.loc[mask, 'DISCOVERY_MONTH'] = df.loc[mask, 'INTERVIEW_QUARTER'].map(q2m)
    df = df.dropna(subset=['DISCOVERY_MONTH', 'year', 'FINAL_ITS_WEIGHT']).copy()
    df['Year_Month'] = (df['year'].astype(int).astype(str) + '-' +
                        df['DISCOVERY_MONTH'].astype(int).astype(str).str.zfill(2))
    monthly = df.groupby('Year_Month')['FINAL_ITS_WEIGHT'].sum().reset_index()
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Victims'}, inplace=True)
    return monthly


def load_breach_data(raw_breach_path):
    """Loads the augmented PRC dataset with Heartland's 2009 breach added manually."""
    df = pd.read_csv(raw_breach_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])

    heartland = pd.DataFrame({
        'org_name': ['Heartland'],
        'reported_date': [pd.Timestamp('2009-01-20')],
        'total_affected': [130_000_000]
    })
    return pd.concat([df, heartland], ignore_index=True)


def get_breach_monthly(df):
    """Aggregates raw records exposed by month, with zero-count months interpolated."""
    df = df.copy()
    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Year_Month')['total_affected'].sum().reset_index()
    monthly.rename(columns={'total_affected': 'Raw_Records_Exposed'}, inplace=True)
    monthly = monthly.sort_values('Year_Month').reset_index(drop=True)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].replace(0, np.nan)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].interpolate(method='linear')
    return monthly


def get_case_study_records(df_with_heartland):
    """Looks up each case study breach's exact record count and report month from the augmented PRC data."""
    records = {}
    for name, info in CASE_STUDY_BREACHES.items():
        date = pd.Timestamp(info['date'])
        match = df_with_heartland[
            (df_with_heartland['org_name'] == info['org_name']) &
            (df_with_heartland['reported_date'] == date)
        ]
        records[name] = {
            'records': match['total_affected'].iloc[0],
            'year_month': date.strftime('%Y-%m'),
        }
    return records


def make_social_cost_fn(cost_df):
    """Builds a month index -> social cost per victim function, via piecewise linear interpolation between survey wave anchors."""
    survey_wave_t = {2008: 0, 2012: 48, 2014: 72, 2016: 96, 2018: 120, 2021: 156}
    known_t = [survey_wave_t[y] for y in cost_df['Year'] if y in survey_wave_t]
    known_cost = [cost_df.loc[cost_df['Year'] == y, 'Total Social Cost per Victim ($)'].values[0]
                  for y in cost_df['Year'] if y in survey_wave_t]
    return lambda t: float(np.interp(t, known_t, known_cost))


def upper_bound_cost(breach_year_month, records, base_df, poly_log, social_cost_fn, alpha):
    """Projected long-term victims and social cost for one breach at a given alpha, summed from a discovery lag of two months through the end of the study period."""
    match = base_df[base_df['Year_Month'] == breach_year_month]
    if match.empty:
        return None, None
    T_idx = match.index[0]
    start_idx = T_idx + DISC_LAG

    total_victims = 0.0
    total_cost = 0.0
    for k in range(start_idx, len(base_df)):
        months_since = k - T_idx
        fresh_records = records * (alpha ** months_since)
        conv_rate = np.exp(poly_log(base_df.loc[k, 't'])) / 100_000
        victims_k = fresh_records * conv_rate
        cost_k = social_cost_fn(base_df.loc[k, 't'])
        total_victims += victims_k
        total_cost += victims_k * cost_k
    return total_victims, total_cost


def main():
    print("\n--- Part 5: Sensitivity to the Discount Factor Alpha ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    its_data_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    raw_breach_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    social_cost_path = os.path.join(root_dir, 'data/processed/part2/social_cost/social_cost_analysis.csv')
    output_dir = os.path.join(root_dir, 'data/processed/part5')
    os.makedirs(output_dir, exist_ok=True)

    its_df = load_its_monthly(its_data_path)
    print(f"Loaded data from {its_data_path}")
    breach_df = load_breach_data(raw_breach_path)
    print(f"Loaded data from {raw_breach_path}")
    raw_df = get_breach_monthly(breach_df)
    case_studies = get_case_study_records(breach_df)
    cost_df = pd.read_csv(social_cost_path)
    print(f"Loaded data from {social_cost_path}")
    social_cost_fn = make_social_cost_fn(cost_df)

    base = pd.merge(its_df, raw_df, on='Year_Month', how='outer')
    base['Date'] = pd.to_datetime(base['Year_Month'])
    base = (base[(base['Date'] < CUTOFF_DATE) & (base['Date'] >= START_DATE)]
            .sort_values('Date').reset_index(drop=True))
    base['Estimated_Victims'] = np.exp(
        np.log(base['Estimated_Victims'].astype(float)).interpolate(method='linear'))
    base['Estimated_Victims'] = base['Estimated_Victims'].ffill().bfill()
    base['t'] = np.arange(len(base))  # Months since Jan 2008.

    coeff_results = []
    case_study_results = []

    print("=" * 60)
    print(f"{'Alpha':>6} | {'Coeffs (a, b, c)':>38}")
    print("-" * 60)

    for alpha in ALPHA_VALUES:
        mdf = base.copy()

        disc = []
        s = 0
        for r in mdf['Raw_Records_Exposed']:
            s = r + alpha * s
            disc.append(s)
        mdf['D_t'] = disc

        mdf['C_t_raw'] = (mdf['Estimated_Victims'] / mdf['D_t']) * 100_000
        mdf['C_t_smooth'] = mdf['C_t_raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

        fit_data = mdf.dropna(subset=['C_t_smooth'])
        x = fit_data['t'].values
        y = np.log(fit_data['C_t_smooth'].values)
        coeffs = np.polyfit(x, y, 2)
        poly_log = np.poly1d(coeffs)

        coeff_results.append({
            'alpha': alpha,
            'coeff_a': coeffs[0],
            'coeff_b': coeffs[1],
            'coeff_c': coeffs[2],
        })

        print(f"  {alpha:.2f} | a={coeffs[0]:.2e}, b={coeffs[1]:.2e}, c={coeffs[2]:.2f}")

        for name, info in case_studies.items():
            victims_ub, cost_ub = upper_bound_cost(
                info['year_month'], info['records'], mdf, poly_log, social_cost_fn, alpha
            )
            settlement = CASE_STUDY_BREACHES[name]['settlement']
            case_study_results.append({
                'alpha': alpha,
                'breach': name,
                'ub_social_cost_B': round(cost_ub / 1e9, 3) if cost_ub else None,
                'ratio_to_settlement': round(cost_ub / settlement, 1) if cost_ub else None,
            })

    coeff_df = pd.DataFrame(coeff_results)
    coeff_path = os.path.join(output_dir, 'sensitivity_alpha.csv')
    coeff_df.to_csv(coeff_path, index=False)
    print(f"\nData saved to {coeff_path}")

    case_study_df = pd.DataFrame(case_study_results)
    case_study_path = os.path.join(output_dir, 'sensitivity_alpha_case_studies.csv')
    case_study_df.to_csv(case_study_path, index=False)
    print(f"Data saved to {case_study_path}")

    print("\nFit coefficients by alpha")
    print(coeff_df.to_string(index=False))

    print("\nUpper-bound social cost ($B) and settlement ratio by alpha")
    print(case_study_df.pivot(index='breach', columns='alpha', values=['ub_social_cost_B', 'ratio_to_settlement']).to_string())


if __name__ == '__main__':
    main()
