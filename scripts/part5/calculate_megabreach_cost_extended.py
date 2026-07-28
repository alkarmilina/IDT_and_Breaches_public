import numpy as np
import pandas as pd
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part5/calculate_megabreach_cost_extended.py

"""
Part 5: Mega-Breach Social Cost (Upper Bound)

Computes the upper-bound social cost estimate for every breach in the
augmented PRC dataset above a minimum size, using the same
alpha-discounted lifecycle model as monthly_data_breach_conversion_beta.py.
This is the projected long-term impact estimate. The paper's lower
bound (the empirically measured short-term impact from the Wilcoxon
signed-rank test) is computed separately, in part6.

Setup:
For a breach of size B occurring at time t, the remaining "fresh"
records from that breach in a later month k are estimated as
B * alpha^(k-t), starting from a discovery lag of two months
(k = t + 2) through the end of the study period. Multiplying by the
conversion rate at month k gives the estimated victims that month, and
weighting by the interpolated social cost per victim at month k gives
the estimated cost. Summing across the study period gives the total
projected victims and social cost for that breach.

Goal:
Produce the upper-bound social cost estimate for every qualifying
breach in the augmented PRC dataset.

Outputs:
- data/processed/part5/megabreach_social_cost_comparison_extended.csv:
  one row per qualifying breach, with projected victims and social
  cost.
"""

ALPHA = 0.80  # Monthly discount factor, matches monthly_data_breach_conversion_beta.py.
DISC_LAG = 2  # Months after a breach before its records start converting into attributed IDT victims.
MIN_RECORDS = 1_000_000  # Minimum breach size to include.


def load_qualifying_breaches(prc_path, min_records):
    """Breaches from the augmented PRC dataset at or above min_records, with their report month."""
    df = pd.read_csv(prc_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])
    df = df[df['total_affected'] >= min_records].copy()
    df['month'] = df['reported_date'].dt.to_period('M').astype(str)
    df = df.sort_values('reported_date').reset_index(drop=True)
    return df[['org_name', 'month', 'total_affected']].rename(
        columns={'org_name': 'name', 'total_affected': 'records'})


def fit_log_quadratic(conv_df):
    """Re-fits ln(C_t_smooth) = a*t^2 + b*t + c from the alpha-discounted conversion data. Returns a poly1d."""
    df = conv_df.dropna(subset=['C_t_smooth']).copy()
    df['t'] = np.arange(len(conv_df))[conv_df['C_t_smooth'].notna()]
    coeffs = np.polyfit(df['t'].values, np.log(df['C_t_smooth'].values), 2)
    return np.poly1d(coeffs)


def upper_bound_cost(breach_month_str, records, conv_df, poly_log, social_cost_fn):
    """
    Projected long-term victims and social cost for one breach, summed
    from a discovery lag of two months through the end of the study
    period.

    Args:
        breach_month_str (str): the breach's report month, as it
            appears in conv_df's Month_Str column.
        records (float): total records exposed in the breach.
        conv_df (pd.DataFrame): monthly conversion rate data, with a
            Month_Str column.
        poly_log (np.poly1d): the log-quadratic conversion rate fit.
        social_cost_fn (callable): month index -> social cost per victim.

    Returns:
        (float, float) or (None, None): total projected victims and
        total projected social cost, or (None, None) if the breach
        month isn't found in conv_df.
    """
    conv_df = conv_df.copy().reset_index(drop=True)
    conv_df['t'] = np.arange(len(conv_df))

    match = conv_df[conv_df['Month_Str'] == breach_month_str]
    if match.empty:
        return None, None
    T_idx = match.index[0]
    start_idx = T_idx + DISC_LAG

    total_victims = 0.0
    total_cost = 0.0
    for k in range(start_idx, len(conv_df)):
        months_since = k - T_idx
        fresh_records = records * (ALPHA ** months_since)
        conv_rate = np.exp(poly_log(conv_df.loc[k, 't'])) / 100_000
        victims_k = fresh_records * conv_rate
        cost_k = social_cost_fn(conv_df.loc[k, 't'])
        total_victims += victims_k
        total_cost += victims_k * cost_k
    return total_victims, total_cost


def make_social_cost_fn(cost_df):
    """Builds a month index -> social cost per victim function, via piecewise linear interpolation between survey wave anchors."""
    survey_wave_t = {2008: 0, 2012: 48, 2014: 72, 2016: 96, 2018: 120, 2021: 156}
    known_t = [survey_wave_t[y] for y in cost_df['Year'] if y in survey_wave_t]
    known_cost = [cost_df.loc[cost_df['Year'] == y, 'Total Social Cost per Victim ($)'].values[0]
                  for y in cost_df['Year'] if y in survey_wave_t]
    return lambda t: float(np.interp(t, known_t, known_cost))


def main():
    print("\n--- Part 5: Mega-Breach Social Cost (Upper Bound) ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    social_cost_path = os.path.join(root_dir, 'data/processed/part2/social_cost/social_cost_analysis.csv')
    conv_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    prc_path = os.path.join(root_dir, 'data/processed/part3/PRC_augmented.csv')
    output_dir = os.path.join(root_dir, 'data/processed/part5')
    os.makedirs(output_dir, exist_ok=True)

    cost_df = pd.read_csv(social_cost_path)
    print(f"Loaded data from {social_cost_path}")
    conv_df = pd.read_csv(conv_path)
    print(f"Loaded data from {conv_path}")
    conv_df['Date'] = pd.to_datetime(conv_df['Date'])

    poly_log = fit_log_quadratic(conv_df)
    social_cost_fn = make_social_cost_fn(cost_df)

    breaches = load_qualifying_breaches(prc_path, MIN_RECORDS)
    print(f"Loaded data from {prc_path}")

    rows = []
    for _, b in breaches.iterrows():
        victims_ub, cost_ub = upper_bound_cost(
            b['month'], b['records'], conv_df, poly_log, social_cost_fn
        )
        rows.append({
            'breach': b['name'],
            'month': b['month'],
            'records_exposed': b['records'],
            'ub_projected_victims': round(victims_ub) if victims_ub else None,
            'ub_social_cost_usd': round(cost_ub, 2) if cost_ub else None,
            'ub_social_cost_B': round(cost_ub / 1e9, 3) if cost_ub else None,
        })
        if victims_ub and cost_ub:
            print(f"{b['name']} ({b['month']}): {victims_ub:,.0f} victims, ${cost_ub/1e9:.3f}B")

    out = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'megabreach_social_cost_comparison_extended.csv')
    out.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == '__main__':
    main()
