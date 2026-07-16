import pandas as pd
import numpy as np
import os

# python scripts/part5/sensitivity_alpha.py
# Sensitivity analysis on the discount factor alpha (β in script notation).
# Refits the log-quadratic conversion model for each alpha value and recomputes
# upper-bound social cost estimates for the three case study breaches.

ITS_DATA_PATH   = 'data/processed/part1/its_victims.parquet'
RAW_BREACH_PATH = 'data/processed/part3/PRC_augmented.csv'
OUTPUT_DIR      = 'data/processed/part5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOOTH_WINDOW = 6
CUTOFF_DATE   = '2022-01-01'
START_DATE    = '2008-01-01'
ALPHA_VALUES  = [0.70, 0.80, 0.90]

# Social cost per victim by survey wave (t = months since Jan 2008)
KNOWN_T     = [0, 48, 72, 96, 120, 156]
KNOWN_COSTS = [1110.31, 921.38, 853.41, 245.27, 244.40, 295.73]

CASE_STUDIES = [
    {'name': 'Heartland', 'date': '2009-01-01', 'records': 130_000_000,   'settlement': 107_000_000},
    {'name': 'Target',    'date': '2013-12-01', 'records':  40_000_000,   'settlement':  18_500_000},
    {'name': 'Yahoo',     'date': '2016-12-01', 'records': 1_000_000_000, 'settlement': 117_500_000},
    {'name': 'Equifax',   'date': '2017-09-01', 'records': 147_000_000,   'settlement': 700_000_000},
]


def load_its_monthly():
    df = pd.read_parquet(ITS_DATA_PATH)
    q2m = {1.0: 2, 2.0: 5, 3.0: 8, 4.0: 11}
    df['DISCOVERY_MONTH']   = pd.to_numeric(df['DISCOVERY_MONTH'],   errors='coerce')
    df['INTERVIEW_QUARTER'] = pd.to_numeric(df['INTERVIEW_QUARTER'], errors='coerce')
    df['year']              = pd.to_numeric(df['year'],              errors='coerce')
    mask = df['DISCOVERY_MONTH'].isnull()
    df.loc[mask, 'DISCOVERY_MONTH'] = df.loc[mask, 'INTERVIEW_QUARTER'].map(q2m)
    df = df.dropna(subset=['DISCOVERY_MONTH', 'year', 'FINAL_ITS_WEIGHT']).copy()
    df['Year_Month'] = (df['year'].astype(int).astype(str) + '-' +
                        df['DISCOVERY_MONTH'].astype(int).astype(str).str.zfill(2))
    monthly = df.groupby('Year_Month')['FINAL_ITS_WEIGHT'].sum().reset_index()
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Victims'}, inplace=True)
    return monthly


def load_breach_monthly():
    df = pd.read_csv(RAW_BREACH_PATH)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])
    heartland = pd.DataFrame({'reported_date': [pd.Timestamp('2009-01-20')],
                               'total_affected': [130_000_000]})
    df = pd.concat([df, heartland], ignore_index=True)
    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Year_Month')['total_affected'].sum().reset_index()
    monthly.rename(columns={'total_affected': 'Raw_Records_Exposed'}, inplace=True)
    monthly = monthly.sort_values('Year_Month').reset_index(drop=True)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].replace(0, np.nan)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].interpolate(method='linear')
    return monthly


its_df = load_its_monthly()
raw_df = load_breach_monthly()

base = pd.merge(its_df, raw_df, on='Year_Month', how='outer')
base['Date'] = pd.to_datetime(base['Year_Month'])
base = (base[(base['Date'] < CUTOFF_DATE) & (base['Date'] >= START_DATE)]
        .sort_values('Date').reset_index(drop=True))
base['Estimated_Victims'] = np.exp(
    np.log(base['Estimated_Victims'].astype(float)).interpolate(method='linear'))
base['Estimated_Victims'] = base['Estimated_Victims'].ffill().bfill()
base['Months_Since_Start'] = np.arange(len(base))

results = []

print("=" * 70)
print(f"{'Alpha':>6} | {'Breach':>10} | {'Coeffs (a, b, c)':>38} | {'UB ($B)':>10} | {'Ratio':>6}")
print("-" * 70)

for alpha in ALPHA_VALUES:
    mdf = base.copy()

    # Compute discounted denominator with this alpha
    disc = []
    s = 0
    for r in mdf['Raw_Records_Exposed']:
        s = r + alpha * s
        disc.append(s)
    mdf['Discounted_Denominator'] = disc

    mdf['Rate_Raw']      = (mdf['Estimated_Victims'] / mdf['Discounted_Denominator']) * 100_000
    mdf['Rate_Smoothed'] = mdf['Rate_Raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

    fit_data  = mdf.dropna(subset=['Rate_Smoothed'])
    x         = fit_data['Months_Since_Start'].values
    y         = np.log(fit_data['Rate_Smoothed'].values)
    coeffs    = np.polyfit(x, y, 2)
    poly_log  = np.poly1d(coeffs)

    for study in CASE_STUDIES:
        T_date   = pd.to_datetime(study['date'])
        T_idx    = mdf[mdf['Date'] == T_date].index[0]
        lag_idx  = T_idx + 2

        total_victims   = 0
        total_cost      = 0

        for k in range(lag_idx, len(mdf)):
            months_since = k - T_idx
            fresh        = study['records'] * (alpha ** months_since)
            t            = mdf.iloc[k]['Months_Since_Start']
            conv         = np.exp(poly_log(t)) / 100_000
            victims      = fresh * conv
            sc           = np.interp(t, KNOWN_T, KNOWN_COSTS)
            total_victims += victims
            total_cost    += victims * sc

        ub_billions = total_cost / 1e9
        ratio       = total_cost / study['settlement']

        results.append({
            'alpha':        alpha,
            'breach':       study['name'],
            'coeff_a':      coeffs[0],
            'coeff_b':      coeffs[1],
            'coeff_c':      coeffs[2],
            'ub_billions':  ub_billions,
            'settlement_m': study['settlement'] / 1e6,
            'ratio':        ratio,
        })

        print(f"  {alpha:.2f} | {study['name']:>10} | "
              f"a={coeffs[0]:.2e}, b={coeffs[1]:.2e}, c={coeffs[2]:.2f} | "
              f"${ub_billions:>8.2f}B | {ratio:>5.0f}x")

out_df = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, 'sensitivity_alpha.csv')
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Print pivot table for easy LaTeX entry
print("\n--- Upper Bound ($B) by alpha and breach ---")
pivot = out_df.pivot(index='alpha', columns='breach', values='ub_billions')
print(pivot[['Heartland', 'Target', 'Yahoo', 'Equifax']].round(2).to_string())

print("\n--- Fit coefficients by alpha ---")
coeff_df = out_df[['alpha','breach','coeff_a','coeff_b','coeff_c']].drop_duplicates(subset='alpha')
print(coeff_df.to_string(index=False))
