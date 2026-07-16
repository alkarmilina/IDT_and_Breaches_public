import requests
import pandas as pd
import os

"""
================================================================================
SCRIPT: race_table.py
PURPOSE: Extracts raw U.S. Census data for demographic comparison to ITS data.
LOCATION: scripts/part2/census_api/race_table.py
================================================================================
"""

def get_census_race_data():
    """
    Connects to the US Census Bureau API to pull population data for specific racial groups and years.
    """
    # --- 1. Path Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigating from scripts/part2/census_api/ up to IDT_and_Breaches/
    root_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
    
    # Updated output location: data/processed/part2/census/
    output_dir = os.path.join(root_dir, "data/processed/part2/census")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "census_race_data_2008-2021.csv")

    # --- 2. Census Configuration ---
    API_KEY = "66c29d488e88fac1bbc276d63a30c99fdd24ece0"
    YEARS = ["2021", "2018", "2016", "2014", "2012", "2008"]
    
    # Variables for ACS 2017 and later
    RACE_VARIABLES_POST_2016 = {
        "White alone": "DP05_0037E",
        "Black or African American alone": "DP05_0038E",
        "American Indian/Alaska Native alone": "DP05_0039E",
        "Asian alone": "DP05_0044E",
        "Native Hawaiian/Pacific Islander alone": "DP05_0052E",
        "Two or More Races": "DP05_0058E"
    }
    
    # Variables for ACS 2016 and earlier
    RACE_VARIABLES_PRE_2017 = {
        "White alone": "DP05_0032E",
        "Black or African American alone": "DP05_0033E",
        "American Indian/Alaska Native alone": "DP05_0034E",
        "Asian alone": "DP05_0039E",
        "Native Hawaiian/Pacific Islander alone": "DP05_0047E",
        "Two or More Races": "DP05_0053E"
    }
    
    results = []
    print("Starting data retrieval from Census API for race data...")

    for year in YEARS:
        print(f"Fetching data for {year}...")
        
        if int(year) >= 2017:
            current_race_vars = RACE_VARIABLES_POST_2016
        else:
            current_race_vars = RACE_VARIABLES_PRE_2017
            
        all_vars = list(current_race_vars.values())
        url = f"https://api.census.gov/data/{year}/acs/acs1/profile"
        
        params = {
            "get": ",".join(all_vars),
            "for": "us:1",
            "key": API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            header = data[0]
            values = data[1]
            row_data = dict(zip(header, values))
            
            year_summary = {"Year": year}
            for race_name, var_code in current_race_vars.items():
                population = int(float(row_data.get(var_code, 0)))
                year_summary[race_name] = population
            
            results.append(year_summary)
            print(f"Successfully processed data for {year}.")

        except Exception as e:
            print(f"Error for year {year}: {e}")

    if not results:
        print("Could not retrieve any data. Exiting.")
        return

    # --- 3. Processing and Saving ---
    df = pd.DataFrame(results)
    df.set_index('Year', inplace=True)
    df_transposed = df.transpose()

    final_df = pd.DataFrame(index=df_transposed.index)

    for year in sorted(df_transposed.columns):
        final_df[year] = df_transposed[year]
        total_population = df_transposed[year].sum()
        percentage_col_name = f"{year} (%)"
        final_df[percentage_col_name] = (df_transposed[year] / total_population * 100).round(1)

    # Save to the specific part2/census location
    final_df.to_csv(output_filename)
    
    print("\n" + "="*40)
    print("      Data Retrieval Complete!")
    print("="*40)
    print(f"\nData saved to: {output_filename}")

if __name__ == "__main__":
    get_census_race_data()