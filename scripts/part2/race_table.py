import requests
import pandas as pd
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/race_table.py
#
# Requires a Census API key set as an environment variable:
# export CENSUS_API_KEY=your_key_here
# Free key available at https://api.census.gov/data/key_signup.html

"""
Part 2: Census Race Population Data (ACS 1-Year Estimates)

Pulls total U.S. population by race from the Census Bureau's American
Community Survey Data Profile (DP05), for each ITS survey year, to serve
as the general-population comparison point for victim race demographics.

Setup:
The DP05 variable codes for each race category changed between the 2016
and 2017 vintages of the ACS, so two separate variable maps are used
depending on year.

Goal:
Produce a table of U.S. population by race, in raw counts and as a
percentage of total population, for each ITS survey year.

Outputs:
- data/processed/part2/census/census_race_data_2008-2021.csv: one row
  per race category, one pair of columns (count, percentage) per year.
"""

YEARS = ["2021", "2018", "2016", "2014", "2012", "2008"]

# DP05 variable codes for ACS 2017 and later.
RACE_VARS_POST_2016 = {
    "White alone": "DP05_0037E",
    "Black or African American alone": "DP05_0038E",
    "American Indian/Alaska Native alone": "DP05_0039E",
    "Asian alone": "DP05_0044E",
    "Native Hawaiian/Pacific Islander alone": "DP05_0052E",
    "Two or More Races": "DP05_0058E"
}

# DP05 variable codes for ACS 2016 and earlier.
RACE_VARS_PRE_2017 = {
    "White alone": "DP05_0032E",
    "Black or African American alone": "DP05_0033E",
    "American Indian/Alaska Native alone": "DP05_0034E",
    "Asian alone": "DP05_0039E",
    "Native Hawaiian/Pacific Islander alone": "DP05_0047E",
    "Two or More Races": "DP05_0053E"
}


def get_census_race_data():
    print("\n--- Part 2: Census Race Population Data ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    output_dir = os.path.join(root_dir, "data/processed/part2/census")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "census_race_data_2008-2021.csv")

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("Error: CENSUS_API_KEY environment variable not set.")
        return

    results = []
    print("Fetching Census race data...")

    for year in YEARS:
        print(f"Fetching data for {year}...")

        current_race_vars = RACE_VARS_POST_2016 if int(year) >= 2017 else RACE_VARS_PRE_2017
        all_vars = list(current_race_vars.values())
        url = f"https://api.census.gov/data/{year}/acs/acs1/profile"

        params = {
            "get": ",".join(all_vars),
            "for": "us:1",
            "key": api_key
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

        except Exception as e:
            print(f"  Error for year {year}: {e}")

    if not results:
        print("Could not retrieve any data. Exiting.")
        return

    df = pd.DataFrame(results)
    df.set_index('Year', inplace=True)
    df_transposed = df.transpose()

    final_df = pd.DataFrame(index=df_transposed.index)

    for year in sorted(df_transposed.columns):
        final_df[year] = df_transposed[year]
        total_population = df_transposed[year].sum()
        percentage_col_name = f"{year} (%)"
        final_df[percentage_col_name] = (df_transposed[year] / total_population * 100).round(1)

    final_df.to_csv(output_filename)
    print(f"\nData saved to {output_filename}")


if __name__ == "__main__":
    get_census_race_data()
