"""
Extended mega-breach social cost analysis.

Computes upper-bound social cost estimates for all qualifying mega-breaches
(>= 10M records, within study period 2008-2021) using the same lifecycle
model as in monthly_data_breach_conversion_beta.py.

dark_web_verified:
  True  — data credibly appeared on dark web / criminal markets (pre-populated
           from public reporting; user should manually verify before publishing)
  False — strong attribution to state-sponsored actors or data reportedly not
           circulated; unlikely to drive consumer IDT
  None  — insufficient public evidence; leave for manual review

To run: python scripts/part5/calculate_megabreach_cost_extended.py
"""

import numpy as np
import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────
SOCIAL_COST_PATH  = 'data/processed/part2/social_cost/social_cost_analysis.csv'
CONV_PATH         = 'data/processed/part5/conversion_data.csv'
PRC_PATH          = 'data/processed/part3/PRC_augmented.csv'
OUTPUT_DIR        = 'data/processed/part5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model parameters (must match monthly_data_breach_conversion_beta.py) ──
ALPHA        = 0.80   # monthly discount factor
DISC_LAG     = 2      # discovery lag (months)
THETA        = 1.75   # under-reporting scaler (baked into C_t_smooth already)
SMOOTH_WIN   = 6      # smoothing window used when producing C_t_smooth

# ── Mega-breach catalogue ───────────────────────────────────────────────────
# month       : YYYY-MM when data entered criminal ecosystem (discovery/disclosure)
# records     : number of records exposed (best public estimate)
# settlement  : total known corporate payout in USD (None if unknown/ongoing)
# dark_web_verified:
#   True  = data credibly appeared on dark web/criminal markets
#   False = state-sponsored attribution; data unlikely to reach consumer markets
#   None  = unclear; needs manual verification
BREACH_DATA = [
    {
        "name": "Heartland Payment Systems",
        "month": "2009-01",
        "records": 130_000_000,
        "settlement": 107_000_000,
        "dark_web_verified": True,
        "notes": "Payment card data sold on criminal forums; well-documented"
    },
    {
        "name": "Target",
        "month": "2013-12",
        "records": 40_000_000,
        "settlement": 18_500_000,
        "dark_web_verified": True,
        "notes": "Card data sold on rescator.su and other carder markets"
    },
    {
        "name": "Anthem",
        "month": "2015-02",
        "records": 80_000_000,
        "settlement": 115_000_000,
        "dark_web_verified": False,
        "notes": "Attributed to Chinese state actors (APT10/Deep Panda); "
                 "health records not believed to have appeared on dark web"
    },
    {
        "name": "T-Mobile (2015)",
        "month": "2015-10",
        "records": 15_100_000,
        "settlement": None,
        "dark_web_verified": None,
        "notes": "Experian breach affecting T-Mobile applicants; "
                 "dark web circulation unclear"
    },
    {
        "name": "Yahoo (2013 breach, disclosed 2016)",
        "month": "2016-12",
        "records": 1_000_000_000,
        "settlement": 117_500_000,
        "dark_web_verified": True,
        "notes": "Data sold on dark web by 'Peace' in 2016; "
                 "originally 3B accounts, 1B used here per PRC entry"
    },
    {
        "name": "Equifax",
        "month": "2017-09",
        "records": 147_000_000,
        "settlement": 700_000_000,
        "dark_web_verified": False,
        "notes": "Attributed to Chinese military (PLA); data not observed on "
                 "commercial dark web markets — reviewer flagged this"
    },
    {
        "name": "Uber",
        "month": "2017-11",
        "records": 57_000_000,
        "settlement": 148_000_000,
        "dark_web_verified": None,
        "notes": "Hackers paid $100K ransom to delete data; "
                 "unclear if records circulated on dark web"
    },
    {
        "name": "Marriott / Starwood",
        "month": "2018-11",
        "records": 500_000_000,
        "settlement": 23_800_000,  # UK ICO fine (£18.4M ≈ $23.8M); class action ongoing
        "dark_web_verified": False,
        "notes": "Attributed to Chinese state actors; no credible dark web sales "
                 "reported — similar state-actor concern as Equifax"
    },
    {
        "name": "Under Armour / MyFitnessPal",
        "month": "2018-03",
        "records": 150_000_000,
        "settlement": 3_500_000,
        "dark_web_verified": True,
        "notes": "Username/password data appeared on dark web markets"
    },
    {
        "name": "Capital One",
        "month": "2019-07",
        "records": 100_000_000,
        "settlement": 190_000_000,
        "dark_web_verified": False,
        "notes": "Insider/cloud-misconfiguration; perpetrator arrested before "
                 "distributing data — no evidence of wide dark web circulation"
    },
    {
        "name": "Zynga",
        "month": "2019-09",
        "records": 200_000_000,
        "settlement": None,
        "dark_web_verified": True,
        "notes": "Game credential data sold by 'GnosticPlayers' on dark web"
    },
    {
        "name": "T-Mobile (2021)",
        "month": "2021-08",
        "records": 53_700_000,
        "settlement": 350_000_000,
        "dark_web_verified": True,
        "notes": "SSN/driver's licence data sold on RaidForums cybercrime forum"
    },
]


def fit_log_quadratic(conv_df):
    """Re-fit log-quadratic to C_t_smooth (Eq. 3), returning a poly1d."""
    df = conv_df.dropna(subset=['C_t_smooth']).copy()
    df['t'] = np.arange(len(conv_df))[conv_df['C_t_smooth'].notna()]
    coeffs = np.polyfit(df['t'].values, np.log(df['C_t_smooth'].values), 2)
    return np.poly1d(coeffs)


def upper_bound_cost(breach_month_str, records, conv_df, poly_log, social_cost_fn):
    """
    Sum lifecycle victims × time-varying social cost over the remainder of the
    study period, starting DISC_LAG months after the breach.
    """
    conv_df = conv_df.copy().reset_index(drop=True)
    conv_df['t'] = np.arange(len(conv_df))

    # Find index of breach month
    match = conv_df[conv_df['Month_Str'] == breach_month_str]
    if match.empty:
        return None, None
    T_idx = match.index[0]
    start_idx = T_idx + DISC_LAG

    total_victims = 0.0
    total_cost    = 0.0
    for k in range(start_idx, len(conv_df)):
        months_since = k - T_idx
        fresh_records = records * (ALPHA ** months_since)
        conv_rate     = np.exp(poly_log(conv_df.loc[k, 't'])) / 100_000
        victims_k     = fresh_records * conv_rate
        cost_k        = social_cost_fn(conv_df.loc[k, 't'])
        total_victims += victims_k
        total_cost    += victims_k * cost_k
    return total_victims, total_cost


def make_social_cost_fn(cost_df, conv_df):
    """Return a function t -> $/victim via linear interpolation of survey years."""
    # Map survey years to month index t
    conv_df = conv_df.copy().reset_index(drop=True)
    conv_df['t'] = np.arange(len(conv_df))
    conv_df['year'] = pd.to_datetime(conv_df['Date']).dt.year

    year_to_t = (conv_df.groupby('year')['t'].mean().to_dict())
    known_t    = [year_to_t[y] for y in cost_df['Year'] if y in year_to_t]
    known_cost = [cost_df.loc[cost_df['Year'] == y, 'Total Social Cost per Victim ($)'].values[0]
                  for y in cost_df['Year'] if y in year_to_t]
    return lambda t: float(np.interp(t, known_t, known_cost))


def main():
    cost_df = pd.read_csv(SOCIAL_COST_PATH)
    conv_df = pd.read_csv(CONV_PATH)
    conv_df['Date'] = pd.to_datetime(conv_df['Date'])

    poly_log       = fit_log_quadratic(conv_df)
    social_cost_fn = make_social_cost_fn(cost_df, conv_df)

    rows = []
    for b in BREACH_DATA:
        victims_ub, cost_ub = upper_bound_cost(
            b['month'], b['records'], conv_df, poly_log, social_cost_fn
        )

        # Snapshot conversion rate at breach month for reference
        match = conv_df[conv_df['Month_Str'] == b['month']]
        rate_snap = float(match['C_t_smooth'].values[0]) if not match.empty else None

        harm_ratio = (cost_ub / b['settlement']) if (cost_ub and b['settlement']) else None

        rows.append({
            'breach':                   b['name'],
            'month':                    b['month'],
            'records_exposed':          b['records'],
            'settlement_usd':           b['settlement'],
            'conv_rate_per_100k':       rate_snap,
            'ub_projected_victims':     round(victims_ub) if victims_ub else None,
            'ub_social_cost_usd':       round(cost_ub, 2) if cost_ub else None,
            'ub_social_cost_B':         round(cost_ub / 1e9, 3) if cost_ub else None,
            'harm_to_settlement_ratio': round(harm_ratio, 1) if harm_ratio else None,
            'dark_web_verified':        b['dark_web_verified'],
            'notes':                    b['notes'],
        })

        print(f"\n{b['name']} ({b['month']})")
        print(f"  Records:          {b['records']:>15,.0f}")
        print(f"  UB Victims:       {victims_ub:>15,.0f}" if victims_ub else "  UB Victims:       N/A")
        print(f"  UB Social Cost:   ${cost_ub/1e9:>10.3f}B" if cost_ub else "  UB Social Cost:   N/A")
        if harm_ratio:
            print(f"  Harm/Settlement:  {harm_ratio:>10.1f}x")
        print(f"  Dark Web:         {b['dark_web_verified']}")

    out = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'megabreach_social_cost_comparison_extended.csv')
    out.to_csv(path, index=False)
    print(f"\nSaved → {path}")


if __name__ == '__main__':
    main()
