import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# python scripts/part5/peak_analysis.py
# --- PATHS AND CONFIGURATION ---
ITS_DATA_PATH = 'data/processed/part1/its_victims.parquet'
RAW_BREACH_PATH = 'data/processed/part3/PRC_augmented.csv' 
CUTOFF_DATE = '2022-01-01'
START_DATE = '2008-01-01'
BETA = 0.80  
SMOOTH_WINDOW = 6  
SURVEY_YEARS = [2008, 2012, 2014, 2016, 2018, 2021]

# --- DATA LOADING FUNCTIONS ---
def get_its_monthly_data():
    # Replicating your harmonization and weighting logic [cite: 131, 137, 853]
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
    # Summing survey weights to get national estimates [cite: 295, 857]
    monthly = analysis_df.groupby(['Year_Month'])['FINAL_ITS_WEIGHT'].sum().reset_index()
    monthly.rename(columns={'FINAL_ITS_WEIGHT': 'Estimated_Victims'}, inplace=True)
    return monthly

def get_raw_breach_monthly():
    # Processing breach chronology data [cite: 144, 175]
    df = pd.read_csv(RAW_BREACH_PATH)
    df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    df = df.dropna(subset=['reported_date'])
    
    # Adding Heartland manually as per your workflow [cite: 386, 512]
    heartland = pd.DataFrame({
        'reported_date': [pd.Timestamp('2009-01-20')],
        'total_affected': [130000000]
    })
    df = pd.concat([df, heartland], ignore_index=True)
    
    df['Year_Month'] = df['reported_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Year_Month')['total_affected'].sum().reset_index()
    monthly.rename(columns={'total_affected': 'Raw_Records_Exposed'}, inplace=True)
    return monthly

# --- MAIN PROCESSING ---
its_df = get_its_monthly_data()
raw_df = get_raw_breach_monthly()

merged_df = pd.merge(its_df, raw_df, on='Year_Month', how='outer')
merged_df['Date'] = pd.to_datetime(merged_df['Year_Month'])
merged_df = merged_df[(merged_df['Date'] < CUTOFF_DATE) & (merged_df['Date'] >= START_DATE)].sort_values('Date').reset_index(drop=True)

# Interpolation logic for ITS gaps 
merged_df['Estimated_Victims'] = np.exp(np.log(merged_df['Estimated_Victims'].astype(float)).interpolate(method='linear'))
merged_df['Estimated_Victims'] = merged_df['Estimated_Victims'].ffill().bfill()

# Cumulative discounted sum logic [cite: 489, 493]
discounted_records = []
current_sum = 0
for r in merged_df['Raw_Records_Exposed'].fillna(0):
    current_sum = r + (BETA * current_sum)
    discounted_records.append(current_sum)

merged_df['Discounted_Denominator'] = discounted_records
merged_df['Rate_Raw'] = (merged_df['Estimated_Victims'] / merged_df['Discounted_Denominator']) * 100000
merged_df['Rate_Smoothed'] = merged_df['Rate_Raw'].rolling(window=SMOOTH_WINDOW, center=True).mean()

# --- PEAK ANALYSIS ---
# Prominence is the 'height' required for a spike to be counted.
peak_indices, properties = find_peaks(merged_df['Rate_Smoothed'].fillna(0), prominence=0.01)
spikes = merged_df.iloc[peak_indices].copy()

print(f"{'Month':<10} | {'Rate':<8} | {'Status':<15}")
print("-" * 40)
for idx, row in spikes.iterrows():
    date = row['Date']
    rate = row['Rate_Smoothed']
    dist_to_survey = min([abs(date.year - sy) for sy in SURVEY_YEARS])
    status = "SURVEY MATCH" if dist_to_survey == 0 else f"{dist_to_survey}yr from survey"
    print(f"{date.strftime('%Y-%m'):<10} | {rate:<8.2f} | {status:<15}")

# --- VISUALIZATION ---
plt.figure(figsize=(14, 7))
plt.plot(merged_df['Date'], merged_df['Rate_Smoothed'], color='indigo', label='6-Mo Moving Avg', linewidth=2)
plt.scatter(spikes['Date'], spikes['Rate_Smoothed'], color='red', marker='x', s=100, label='Detected Peaks', zorder=5)

for year in SURVEY_YEARS:
    plt.axvspan(pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-12-31"), 
                color='gray', alpha=0.15, label='ITS Survey Year' if year == 2008 else "")

plt.yscale('log')
plt.title("Conversion Rate Spikes vs. ITS Survey Waves")
plt.ylabel("Victims per 100k Discounted Records")
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()