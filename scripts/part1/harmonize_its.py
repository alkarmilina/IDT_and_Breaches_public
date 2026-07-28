import pandas as pd
import json
import numpy as np
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part1/harmonize_its.py

"""
Part 1: ITS Survey Harmonization (2008-2021)

The NCVS Identity Theft Supplement changed its variable names, value
codes, and screening questions across its six waves, so the raw survey
files cannot be pooled directly. This script applies a single
harmonization dictionary to recode every wave onto a common set of
variables, then builds a longitudinal victim dataset spanning 2008 to
2021.

Setup:
Harmonization rules are read from config/harmonization_dictionary.json,
keyed by harmonized variable name, with per year source variable names
and recoding instructions. Raw survey files are read from
data/raw/ITS/<year>/DS0001/.

Goal:
Produce a longitudinal dataset of successful identity theft victims for
the rest of the analysis, plus a second version that also keeps
non-victim respondents, needed by analyses that require the full
respondent population rather than victims alone.

Outputs:
- data/processed/part1/its_victims.parquet: one row per victim across all
  six waves, filtered to successful (non-attempted) identity theft.
- data/processed/part1/its_full.parquet: one row per respondent across
  all six waves, victims and non-victims, with an IS_SUCCESSFUL_VICTIM flag.
"""

QUARTER_MID_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}


def get_victim_mask(df, year):
    """Flags respondents who reported any form of identity theft, attempted or successful. The screening questions differ by year."""
    if year == 2021:
        return (df['VS012'] == '1') | (df['VS017'] == '1') | (df['VS019'] == '1') | \
               (df['VS017E'] == '1') | (df['VS041'] == '1') | (df['VS063'] == '1')
    elif year in [2018, 2016, 2014]:
        return (df['VS012'] == '1') | (df['VS017'] == '1') | (df['VS019'] == '1') | \
               (df['VS041'] == '1') | (df['VS063'] == '1')
    elif year == 2012:
        return (df['VS012'] == '1') | (df['VS016'] == '1') | (df['VS018'] == '1') | \
               (df['VS041'] == '1') | (df['VS067'] == '1')
    elif year == 2008:
        return (df['VS011'] == '1') | (df['VS013'] == '1') | (df['VS015'] == '1') | \
               (df['VS017'] == '1') | (df['VS019'] == '1') | (df['VS021'] == '1')


def get_attempt_mask(df, year):
    """Flags respondents whose identity theft was attempted only, never successful. 2021 has no attempt-only category, since that wave was redesigned to screen for successful victims directly."""
    if year == 2008:
        return (df['VS012'] == '2') & (df['VS014'] == '2') & (df['VS016'] == '2') & \
               (df['VS018'] == '2') & (df['VS020'] == '2')
    elif year == 2012:
        return df['VS098'] == '009'
    elif year == 2014:
        return df['VS093'] == '009'
    elif year in [2016, 2018]:
        return df['VS093'] == '9'
    else:
        return pd.Series(False, index=df.index)


def apply_custom_imputation(df, harmonized_df, year):
    """
    Fills in DISCOVERY_YEAR. For 2012, 2014, and 2016, the survey asks
    for the discovery month but not the discovery year, so a 12-month
    lag correction is applied whenever the discovery month falls after
    the interview quarter, since that means discovery happened in the
    prior year.

    Args:
        df (pd.DataFrame): raw survey data for one year.
        harmonized_df (pd.DataFrame): harmonized columns built so far.
        year (int): survey year.

    Returns:
        pd.DataFrame: harmonized_df with DISCOVERY_YEAR filled in.
    """
    if year in [2012, 2014, 2016] and all(v in harmonized_df.columns for v in ['DISCOVERY_MONTH', 'INTERVIEW_QUARTER']):
        print(f"  Applying 12-month lag imputation for DISCOVERY_YEAR in {year}...")
        harmonized_df['DISCOVERY_MONTH'] = pd.to_numeric(harmonized_df['DISCOVERY_MONTH'], errors='coerce')
        harmonized_df['INTERVIEW_QUARTER'] = pd.to_numeric(harmonized_df['INTERVIEW_QUARTER'], errors='coerce')
        interview_month = harmonized_df['INTERVIEW_QUARTER'].map(QUARTER_MID_MONTH)
        lag_mask = harmonized_df['DISCOVERY_MONTH'] > interview_month
        harmonized_df.loc[lag_mask, 'DISCOVERY_YEAR'] = year - 1
        harmonized_df.loc[~lag_mask, 'DISCOVERY_YEAR'] = year
        harmonized_df['DISCOVERY_YEAR'] = pd.to_numeric(harmonized_df['DISCOVERY_YEAR'], errors='coerce')
    elif 'DISCOVERY_YEAR' in harmonized_df.columns:
        harmonized_df['DISCOVERY_YEAR'] = pd.to_numeric(harmonized_df['DISCOVERY_YEAR'], errors='coerce')
    return harmonized_df


def process_yearly_data(df, year, variable_harmonization_rules, is_victim_only=True):
    """
    Recodes every variable in variable_harmonization_rules for one
    survey year onto the harmonized variable names and codes.

    Args:
        df (pd.DataFrame): raw survey data for one year, read as strings.
        year (int): survey year.
        variable_harmonization_rules (dict): the VARIABLE_HARMONIZATION
            section of the harmonization dictionary.
        is_victim_only (bool): if False, also keeps IS_SUCCESSFUL_VICTIM
            so non-victims can be told apart from victims downstream.

    Returns:
        pd.DataFrame: one row per respondent, one column per harmonized
        variable.
    """
    df['year'] = year
    harmonized_df = pd.DataFrame({'year': df['year']})
    if not is_victim_only:
        harmonized_df['IS_SUCCESSFUL_VICTIM'] = df['IS_SUCCESSFUL_VICTIM']

    for group_name, instructions in variable_harmonization_rules.items():
        var_name = instructions.get('name')
        source_info = instructions.get('source_variables', {}).get(str(year))
        if not source_info: continue
        source_vars = source_info if isinstance(source_info, list) else [source_info]
        main_source_var = next((s_var for s_var in source_vars if s_var in df.columns), None)
        if not main_source_var: continue

        recoding_instructions = instructions.get('recoding_instructions')
        harmonized_map = instructions.get('map', {})
        new_series = pd.Series(np.nan, index=df.index)

        if recoding_instructions:
            for new_category, mappings in recoding_instructions.items():
                harmonized_code = harmonized_map.get(new_category)
                if harmonized_code is not None:
                    for mapping in mappings:
                        if str(mapping.get("source_year")) == str(year):
                            source_var_for_map = mapping.get("source_variable", main_source_var)
                            if source_var_for_map not in df.columns: continue
                            for original_code in mapping.get("original_codes", []):
                                mask = df[source_var_for_map] == str(original_code)
                                new_series[mask] = harmonized_code
        else:
            if main_source_var in df.columns:
                new_series = pd.to_numeric(df[main_source_var], errors='coerce')
            else:
                new_series = pd.Series(np.nan, index=df.index)
        harmonized_df[var_name] = new_series

    harmonized_df = apply_custom_imputation(df, harmonized_df, year)

    # Cap outlier values (e.g. hours spent resolving the incident) at each variable's max_value.
    for _, rules in variable_harmonization_rules.items():
        var_name = rules.get('name')
        cleaning_rules = rules.get('cleaning_rules', {})
        if var_name in harmonized_df.columns and 'max_value' in cleaning_rules:
            max_val = cleaning_rules['max_value']
            harmonized_df[var_name] = pd.to_numeric(harmonized_df[var_name], errors='coerce')
            harmonized_df.loc[harmonized_df[var_name] > max_val, var_name] = np.nan

    print(f"  Applied harmonization. Shape of harmonized data: {harmonized_df.shape}")
    return harmonized_df


def main():
    print("\n--- Part 1: ITS Survey Harmonization ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    harmonization_plan_path = os.path.join(root_dir, "config/harmonization_dictionary.json")
    raw_data_paths = {
        "2021": os.path.join(root_dir, "data/raw/ITS/2021/DS0001/38501-0001-Data.tsv"),
        "2018": os.path.join(root_dir, "data/raw/ITS/2018/DS0001/37923-0001-Data.tsv"),
        "2016": os.path.join(root_dir, "data/raw/ITS/2016/DS0001/36829-0001-Data.tsv"),
        "2014": os.path.join(root_dir, "data/raw/ITS/2014/DS0001/36044-0001-Data.tsv"),
        "2012": os.path.join(root_dir, "data/raw/ITS/2012/DS0001/34735-0001-Data.tsv"),
        "2008": os.path.join(root_dir, "data/raw/ITS/2008/DS0001/26362-0001-Data.tsv")
        }

    output_dir = os.path.join(root_dir, "data/processed/part1")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, "its_victims.parquet")
    output_all_respondents_filename = os.path.join(output_dir, "its_full.parquet")

    try:
        with open(harmonization_plan_path, 'r') as f:
            harmonization_plan = json.load(f)
        print(f"Loaded plan from {harmonization_plan_path}")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    variable_harmonization_rules = harmonization_plan.get("VARIABLE_HARMONIZATION", {})

    # Pass 1: identity theft victims only, the dataset used throughout the rest of the analysis.
    processed_yearly_data = []
    for year, path in raw_data_paths.items():
        year = int(year)
        print(f"\nProcessing {year} (victims only)")
        try:
            df = pd.read_csv(path, sep='\t', low_memory=False, dtype=str)
        except Exception:
            print(f"  File not found at: {path}")
            continue

        df = df[get_victim_mask(df, year)].copy()
        df = df[~get_attempt_mask(df, year)]

        harmonized_df = process_yearly_data(df, year, variable_harmonization_rules, is_victim_only=True)
        processed_yearly_data.append(harmonized_df)

    if processed_yearly_data:
        final_df = pd.concat(processed_yearly_data, ignore_index=True)
        final_df.to_parquet(output_filename, index=False)
        print(f"\nVictim data saved to {output_filename}")

    # Pass 2: all respondents, victims and non-victims. Needed by analyses that require the
    # full population, such as the victimization odds regression and breach notification counts.
    processed_all_data = []
    for year, path in raw_data_paths.items():
        year = int(year)
        print(f"\nProcessing {year} (all respondents)")
        try:
            df = pd.read_csv(path, sep='\t', low_memory=False, dtype=str)
        except Exception:
            continue

        df['IS_SUCCESSFUL_VICTIM'] = get_victim_mask(df, year) & ~get_attempt_mask(df, year)
        harmonized_df = process_yearly_data(df, year, variable_harmonization_rules, is_victim_only=False)
        processed_all_data.append(harmonized_df)

    if processed_all_data:
        all_respondents_df = pd.concat(processed_all_data, ignore_index=True)
        all_respondents_df.to_parquet(output_all_respondents_filename, index=False)
        print(f"Full data saved to {output_all_respondents_filename}")


if __name__ == '__main__':
    main()
