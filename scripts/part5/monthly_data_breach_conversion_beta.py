import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
# python scripts/part5/monthly_data_breach_conversion_beta.py

# Paths and Config
ITS_DATA_PATH = 'data/processed/part1/its_victims.parquet'
RAW_BREACH_PATH = 'data/processed/part3/PRC_augmented.csv'
CSV_OUTPUT_DIR = 'data/processed/part5/'
OUTPUT_DIR = os.path.join('plots', 'part5')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

SMOOTH_WINDOW = 6  
CUTOFF_DATE = '2022-01-01'
START_DATE = '2008-01-01'
BETA = 0.80  

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 35,
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 28,  
    'ytick.labelsize': 28,  
    'legend.fontsize': 25,
    'figure.titlesize': 40
})

def get_its_monthly_data():
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
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Victims'}, inplace=True)
    return monthly

def get_raw_breach_monthly():
    df = pd.read_csv(RAW_BREACH_PATH)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])
    
    heartland = pd.DataFrame({
        'reported_date': [pd.Timestamp('2009-01-20')],
        'total_affected': [130000000]
    })
    df = pd.concat([df, heartland], ignore_index=True)
    
    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Year_Month')['total_affected'].sum().reset_index()
    monthly.rename(columns={'total_affected': 'Raw_Records_Exposed'}, inplace=True)
    
    monthly = monthly.sort_values('Year_Month').reset_index(drop=True)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].replace(0, np.nan)
    monthly['Raw_Records_Exposed'] = monthly['Raw_Records_Exposed'].interpolate(method='linear')
    return monthly

its_df = get_its_monthly_data()
raw_df = get_raw_breach_monthly()

merged_df = pd.merge(its_df, raw_df, on='Year_Month', how='outer')
merged_df['Date'] = pd.to_datetime(merged_df['Year_Month'])
merged_df = merged_df[(merged_df['Date'] < CUTOFF_DATE) & (merged_df['Date'] >= START_DATE)].sort_values('Date').reset_index(drop=True)

merged_df['Estimated_Victims'] = np.exp(np.log(merged_df['Estimated_Victims'].astype(float)).interpolate(method='linear'))
merged_df['Estimated_Victims'] = merged_df['Estimated_Victims'].ffill().bfill()

discounted_records = []
current_sum = 0
for r in merged_df['Raw_Records_Exposed']:
    current_sum = r + (BETA * current_sum)
    discounted_records.append(current_sum)

merged_df['Discounted_Denominator'] = discounted_records

merged_df['Rate_Raw'] = (merged_df['Estimated_Victims'] / merged_df['Discounted_Denominator']) * 100000
merged_df['Rate_Smoothed'] = merged_df['Rate_Raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

merged_df['Months_Since_Start'] = np.arange(len(merged_df))
fit_data = merged_df.dropna(subset=['Rate_Smoothed']).copy()
x_time = fit_data['Months_Since_Start'].values
y_log_rate = np.log(fit_data['Rate_Smoothed'].values)

coeffs = np.polyfit(x_time, y_log_rate, 2)
poly_log = np.poly1d(coeffs)
merged_df['Smooth_Trend'] = np.exp(poly_log(merged_df['Months_Since_Start']))

def format_sci(val):
    s = "{:.4e}".format(val)
    base, exp = s.split('e')
    return f"{base} \\times 10^{{{int(exp)}}}"

def save_plot(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), bbox_inches='tight')

# --- Plotting ---
fig1, ax1 = plt.subplots(figsize=(16, 12))

# Plot lines with specific labels for the legend
ax1.plot(merged_df['Date'], merged_df['Rate_Raw'], color='mediumpurple', linewidth=4, alpha=0.7, label='Monthly rate')
ax1.plot(merged_df['Date'], merged_df['Rate_Smoothed'], color='indigo', linewidth=6, label='6-Mo moving avg')
ax1.plot(merged_df['Date'], merged_df['Smooth_Trend'], color='cyan', linewidth=8, label='Fit')

ax1.set_yscale('log')
ax1.set_title(f"Estimated Conversion Rate of Cumulative Records to IDT", pad=20)
ax1.set_ylabel('Victims per 100k Discounted Records')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')

# Call the legend
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

events = [('Equifax', '2017-09-01', (40, 40)), ('Yahoo', '2016-12-01', (-40, 40)), ('Target', '2013-12-01', (-40, 40)), ('Heartland', '2009-01-01', (40, 40))]
for label, date_str, offset in events:
    event_date = pd.to_datetime(date_str)
    y_val = merged_df.loc[merged_df['Date'] == event_date, 'Rate_Smoothed'].values
    if len(y_val) > 0:
        ax1.annotate(label, xy=(event_date, y_val[0]), xytext=offset, textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='black', lw=2), fontsize=25, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

eqn_text = rf"$\ln(Rate) = ({format_sci(coeffs[0])})t^2 + ({format_sci(coeffs[1])})t + {coeffs[2]:.2f}$"
ax1.text(0.05, 0.05, f"Fit Equation:\n{eqn_text}", 
         transform=ax1.transAxes, fontsize=24, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_plot(fig1, 'conversion_rate_beta')

merged_df.to_csv(os.path.join(CSV_OUTPUT_DIR, 'conversion_data.csv'), index=False)

# --- Console Output ---
print(f"\n--- Conversion Rates (Victims per 100k Records) with β={BETA} ---")
target_dates = {'Heartland': '2009-01-01', 'Target': '2013-12-01', 'Yahoo': '2016-12-01', 'Equifax': '2017-09-01'}
for name, d_str in target_dates.items():
    row = merged_df[merged_df['Date'] == pd.to_datetime(d_str)]
    if not row.empty:
        rate = row['Rate_Smoothed'].values[0]
        print(f"{name} ({d_str}): {rate:.4f}")

print(f"\n--- Upper Bound Social Cost Projections (Sum of Harm Lifecycle) ---")
known_t = [0, 48, 72, 96, 120, 156]
known_costs = [1110.31, 921.38, 853.41, 245.27, 244.40, 295.73]

case_studies = [
    {'name': 'Heartland', 'date': '2009-01-01', 'size': 130000000},
    {'name': 'Target', 'date': '2013-12-01', 'size': 40000000},
    {'name': 'Equifax', 'date': '2017-09-01', 'size': 147000000}
]

for study in case_studies:
    T_date = pd.to_datetime(study['date'])
    T_idx = merged_df[merged_df['Date'] == T_date].index[0]
    lag_idx = T_idx + 2 
    
    total_projected_victims = 0
    total_social_cost_sum = 0
    
    for k in range(lag_idx, len(merged_df)):
        months_since_breach = k - T_idx
        fresh_records = study['size'] * (BETA ** months_since_breach)
        t = merged_df.iloc[k]['Months_Since_Start']
        conv_rate = np.exp(poly_log(t)) / 100000
        monthly_victims = fresh_records * conv_rate
        current_social_cost_per_victim = np.interp(t, known_t, known_costs)
        total_projected_victims += monthly_victims
        total_social_cost_sum += (monthly_victims * current_social_cost_per_victim)
        
    avg_implied_cost = total_social_cost_sum / total_projected_victims if total_projected_victims > 0 else 0
    print(f"\n{study['name']} (Exposed Records: {study['size']:,})")
    print(f"Total Projected Victims: {total_projected_victims:,.0f}")
    print(f"FINAL UPPER BOUND SOCIAL COST: ${total_social_cost_sum:,.2f}")

print(f"\nSuccess! Visuals and interpolated calculations completed.")