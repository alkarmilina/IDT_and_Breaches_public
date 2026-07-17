import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# python scripts/part5/monthly_data_breach_conversion.py
# --- 1. CONFIGURATION ---
ITS_DATA_PATH = 'data/processed/part1/its_victims.parquet'
ITS_ALL_PATH = 'data/processed/part1/its_full.parquet' 
FINAL_SUPPLY_PATH = 'data/processed/part4/PRC_Data_After_Saturation_Model_Exp.csv' 
CSV_OUTPUT_DIR = 'data/processed/part5/'

OUTPUT_DIR = os.path.join('plots', 'part5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESIDUAL_BUFFER = 1
SMOOTH_WINDOW = 6  
CUTOFF_DATE = '2022-01-01'

# --- STYLING UPDATE ---
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

def get_gold_data():
    df = pd.read_parquet(ITS_ALL_PATH)
    quarter_to_month = {1.0: 2, 2.0: 5, 3.0: 8, 4.0: 11}
    df['DISCOVERY_MONTH'] = pd.to_numeric(df['DISCOVERY_MONTH'], errors='coerce')
    df['INTERVIEW_QUARTER'] = pd.to_numeric(df['INTERVIEW_QUARTER'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    missing_mask = df['DISCOVERY_MONTH'].isnull()
    df.loc[missing_mask, 'DISCOVERY_MONTH'] = df.loc[missing_mask, 'INTERVIEW_QUARTER'].map(quarter_to_month)
    
    df = df.dropna(subset=['year', 'FINAL_ITS_WEIGHT']).copy()
    df['DISCOVERY_MONTH'] = df['DISCOVERY_MONTH'].fillna(1)
    
    df['Year_Month'] = (
        df['year'].astype(int).astype(str) + '-' + 
        df['DISCOVERY_MONTH'].astype(int).astype(str).str.zfill(2)
    )
    notified = df[df['NOTIFIED_OF_DATA_BREACH'] == 1].groupby(['Year_Month'])['FINAL_ITS_WEIGHT'].sum().reset_index()
    notified.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Notified'}, inplace=True)
    return notified

# --- 2. LOAD & MERGE ---
its_df = get_its_monthly_data()
gold_df = get_gold_data()
um_df = pd.read_csv(FINAL_SUPPLY_PATH)

its_combined = pd.merge(its_df, gold_df, on='Year_Month', how='outer')
merged_df = pd.merge(its_combined, um_df, left_on='Year_Month', right_on='Month_Str', how='outer')
merged_df['Date'] = pd.to_datetime(merged_df['Month_Str'])
merged_df.sort_values('Date', inplace=True)
merged_df = merged_df[merged_df['Date'] < CUTOFF_DATE].copy()
merged_df.reset_index(drop=True, inplace=True)

# --- 3. LOG-LINEAR INTERPOLATION ---
for col in ['Estimated_Victims', 'Estimated_Notified']:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').replace(0, np.nan)
    merged_df[col] = np.exp(np.log(merged_df[col].astype(float)).interpolate(method='linear'))
    merged_df[col] = merged_df[col].ffill().bfill()

def calculate_section7_supply(row):
    # UPDATED: Constant Theta = 1.75
    theta = 1.75
        
    # Calculate Supply purely from the Saturation Model (no floor)
    modeled_supply = row['Um_Unique_Individuals'] * theta
    
    return modeled_supply

merged_df['Um_Final'] = merged_df.apply(calculate_section7_supply, axis=1)

# --- 4. CONVERSION RATE ---
merged_df['Rate_Raw'] = (merged_df['Estimated_Victims'] / merged_df['Um_Final']) * 100000
merged_df['Rate_Smoothed'] = merged_df['Rate_Raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

# --- 5. VISUALIZATION ---

# Plot 1: Supply/Demand Overlay
fig, ax = plt.subplots(figsize=(16, 12)) 
ax.plot(merged_df['Date'], merged_df['Um_Final'], color='tomato', 
        label='Records Exposed', alpha=0.8, linestyle='-', linewidth=4)
ax.plot(merged_df['Date'], merged_df['Estimated_Victims'], color='teal', 
        label='IDT Victims', linestyle='--', linewidth=5)

ax.set_title(f"Month-to-Month Breach Records and Victim IDT Reports", pad=20)
ax.set_yscale('log')
ax.set_ylabel('Count (Log Scale)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))

ax.tick_params(axis='both', which='major', labelsize=28)
ax.legend(loc='upper left', frameon=True)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overlay.pdf'), dpi=300)

# Plot 2: Conversion Rate
fig, ax = plt.subplots(figsize=(16, 12)) 
ax.plot(merged_df['Date'], merged_df['Rate_Raw'], color='purple', alpha=0.25, 
         linewidth=2, label='Monthly Rate')
ax.plot(merged_df['Date'], merged_df['Rate_Smoothed'], color='purple', 
         linewidth=6, label=f'{SMOOTH_WINDOW}-Mo Moving Avg')

ax.set_title("Month-to-Month Breach Victim to IDT Conversion Rate", pad=20)
ax.set_yscale('log')
ax.set_ylabel('Victims per 100k Records')

ax.tick_params(axis='both', which='major', labelsize=28)
ax.grid(True, alpha=0.5, which='both', linestyle='--')
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'conversion_rate.pdf'), dpi=300)

print(f"Plots saved to {OUTPUT_DIR}")

# --- 6. EXPORT ---
merged_df.to_csv(os.path.join(CSV_OUTPUT_DIR, 'conversion_data_saturation.csv'), index=False)
print(f"Data saved to {CSV_OUTPUT_DIR}/conversion_data_saturation.csv")