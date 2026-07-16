import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# To run: python scripts/part5/monthly_data_breach_conversion_lag.py

# --- 1. CONFIGURATION ---
ITS_DATA_PATH = 'data/processed/part1/its_victims.parquet'
RAW_BREACH_PATH = 'data/processed/part3/PRC_augmented.csv' 
CSV_OUTPUT_DIR = 'data/processed/part5/'
OUTPUT_DIR = os.path.join('plots', 'part5', 'lag')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

SMOOTH_WINDOW = 6  
LAG_MONTHS = 2  
CUTOFF_DATE = '2022-01-01'
START_DATE = '2008-01-01'

# --- STYLING ---
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
    
    # Manually inject Heartland (Jan 2009)
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

# --- 2. LOAD, MERGE, AND INTERPOLATE ---
its_df = get_its_monthly_data()
raw_df = get_raw_breach_monthly()

merged_df = pd.merge(its_df, raw_df, on='Year_Month', how='outer')
merged_df['Date'] = pd.to_datetime(merged_df['Year_Month'])
merged_df = merged_df[(merged_df['Date'] < CUTOFF_DATE) & (merged_df['Date'] >= START_DATE)].sort_values('Date').reset_index(drop=True)

# Log-linear interpolation for Victim counts
merged_df['Estimated_Victims'] = np.exp(np.log(merged_df['Estimated_Victims'].astype(float)).interpolate(method='linear'))
merged_df['Estimated_Victims'] = merged_df['Estimated_Victims'].ffill().bfill()
merged_df['Um_Final'] = merged_df['Raw_Records_Exposed']

# --- 3. APPLY LAG ---
# Shift Estimated_Victims backwards (negative shift) so that Victims at t+LAG are aligned with Records at t
merged_df['Estimated_Victims_Lagged'] = merged_df['Estimated_Victims'].shift(-LAG_MONTHS)

# --- 4. CONVERSION RATES ---
# Using the lagged victims for the rate calculation
merged_df['Rate_Raw'] = (merged_df['Estimated_Victims_Lagged'] / merged_df['Um_Final']) * 100000
merged_df['Rate_Smoothed'] = merged_df['Rate_Raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

# --- 5. SMOOTH BLUE LINE FIT ---
merged_df['Months_Since_Start'] = np.arange(len(merged_df))
fit_data = merged_df.dropna(subset=['Rate_Smoothed']).copy()
x_time = fit_data['Months_Since_Start'].values
y_log_rate = np.log(fit_data['Rate_Smoothed'].values)

coeffs = np.polyfit(x_time, y_log_rate, 2)
poly_log = np.poly1d(coeffs)
merged_df['Smooth_Blue_Trend'] = np.exp(poly_log(merged_df['Months_Since_Start']))

# --- 6. VISUALIZATION ---

def format_sci(val):
    s = "{:.4e}".format(val)
    base, exp = s.split('e')
    return f"{base} \\times 10^{{{int(exp)}}}"

def log_formatter(x, pos):
    if x >= 1e9: return f'{x/1e9:.1f}B'
    if x >= 1e6: return f'{x/1e6:.1f}M'
    if x >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.0f}'

# PLOT 1: Conversion Rate with Lag
fig1, ax1 = plt.subplots(figsize=(16, 12))
ax1.plot(merged_df['Date'], merged_df['Rate_Raw'], color='purple', alpha=0.1, linewidth=2, label=f'Monthly rate ({LAG_MONTHS}-mo lag)')
ax1.plot(merged_df['Date'], merged_df['Rate_Smoothed'], color='purple', linewidth=6, label='6-mo moving avg')
ax1.plot(merged_df['Date'], merged_df['Smooth_Blue_Trend'], color='cyan', linewidth=8, label='Log-quadratic regression (Lagged)')
ax1.set_yscale('log')
ax1.set_title(f"Lagged Conversion Rate ({LAG_MONTHS} Months)", pad=20)
ax1.set_ylabel('Victims per 100k Records')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')
ax1.legend(loc='upper right', frameon=True)

eqn_text = rf"$\ln(Rate) = ({format_sci(coeffs[0])})t^2 + ({format_sci(coeffs[1])})t + {coeffs[2]:.2f}$"
ax1.text(0.05, 0.05, f"Fit Equation:\n{eqn_text}\n(t = months since start)", 
         transform=ax1.transAxes, fontsize=24, verticalalignment='bottom', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'conversion_rate_lag.png'), dpi=300)

# PLOT 2: Overlay (Lagged)
fig2, ax2 = plt.subplots(figsize=(16, 12))
ax2.plot(merged_df['Date'], merged_df['Um_Final'], color='tomato', linewidth=3, label='Records Exposed (t)', alpha=0.8)
ax2.plot(merged_df['Date'], merged_df['Estimated_Victims_Lagged'], color='teal', linewidth=4, linestyle='--', label=f'IDT Victims (t + {LAG_MONTHS})')
ax2.set_yscale('log')
ax2.set_title(f"Comparison: Records vs. Lagged Victims", pad=20)
ax2.set_ylabel("Count (Log Scale)")
ax2.yaxis.set_major_formatter(FuncFormatter(log_formatter))
ax2.grid(True, alpha=0.3, which='both', linestyle='--')
ax2.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overlay_lag.png'), dpi=300)

# --- 7. EXPORT ---
merged_df.to_csv(os.path.join(CSV_OUTPUT_DIR, 'conversion_data_lag.csv'), index=False)
print(f"Success! Lagged Fit Coefficients: a={coeffs[0]:.4e}, b={coeffs[1]:.4e}, c={coeffs[2]:.4f}")