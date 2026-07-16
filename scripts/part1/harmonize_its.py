import pandas as pd
import json
import numpy as np
import sys
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part1/harmonize_its.py

"""
Data Harmonization Pipeline: Part 1 - ITS Survey Integration (2008-2021)
"""

QUARTER_MID_MONTH = {1: 2, 2: 5, 3: 8, 4: 11} 
vars_for_imputation = ['DISCOVERY_MONTH', 'INTERVIEW_QUARTER', 'DISCOVERY_YEAR']


def apply_custom_imputation(df, harmonized_df, year):
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
    df['year'] = year
    harmonized_df = pd.DataFrame({'year': df['year']})
    if not is_victim_only:
        harmonized_df['IS_SUCCESSFUL_VICTIM'] = df['IS_SUCCESSFUL_VICTIM']
    
    for group_name, instructions in variable_harmonization_rules.items():
        proposed_name = instructions.get('proposed_harmonized_name')
        source_info = instructions.get('source_variables', {}).get(str(year))
        if not source_info: continue
        source_vars = source_info if isinstance(source_info, list) else [source_info]
        main_source_var = next((s_var for s_var in source_vars if s_var in df.columns), None)
        if not main_source_var: continue 

        recoding_instructions = instructions.get('recoding_instructions')
        harmonized_map = instructions.get('proposed_harmonized_map', {})
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
        harmonized_df[proposed_name] = new_series
    
    harmonized_df = apply_custom_imputation(df, harmonized_df, year)
    
    for var_name, rules in variable_harmonization_rules.items():
        clean_var_name = rules.get('proposed_harmonized_name')
        cleaning_rules = rules.get('cleaning_rules', {})
        if clean_var_name in harmonized_df.columns and 'max_value' in cleaning_rules:
            max_val = cleaning_rules['max_value']
            harmonized_df[clean_var_name] = pd.to_numeric(harmonized_df[clean_var_name], errors='coerce')
            harmonized_df.loc[harmonized_df[clean_var_name] > max_val, clean_var_name] = np.nan
            
    print(f"  Applied harmonization. Shape of harmonized data: {harmonized_df.shape}")
    return harmonized_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Updated path to the config folder
    harmonization_plan_path = os.path.join(root_dir, "config/harmonization_dictionary.json")
        
    # Updated paths based on your ls -R output
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

    print("--- Starting Data Harmonization Process ---")

    try:
        with open(harmonization_plan_path, 'r') as f:
            harmonization_plan = json.load(f)
        print(f"Successfully loaded plan from '{harmonization_plan_path}'.")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    variable_harmonization_rules = harmonization_plan.get("VARIABLE_HARMONIZATION", {})

    # --- PART 1: VICTIMS-ONLY ---
    processed_yearly_data = []
    for year, path in raw_data_paths.items():
        year = int(year)
        print(f"\n--- Processing Year: {year} (Victims Only) ---")
        try:
            df = pd.read_csv(path, sep='\t', low_memory=False, dtype=str)
        except Exception:
            print(f"  File not found at: {path}")
            continue

        if year == 2021:
            victim_mask = (df['VS012'] == '1') | (df['VS017'] == '1') | (df['VS019'] == '1') | \
                          (df['VS017E'] == '1') | (df['VS041'] == '1') | (df['VS063'] == '1')
        elif year in [2018, 2016, 2014]:
            victim_mask = (df['VS012'] == '1') | (df['VS017'] == '1') | (df['VS019'] == '1') | \
                          (df['VS041'] == '1') | (df['VS063'] == '1')
        elif year == 2012:
            victim_mask = (df['VS012'] == '1') | (df['VS016'] == '1') | (df['VS018'] == '1') | \
                          (df['VS041'] == '1') | (df['VS067'] == '1')
        elif year == 2008:
            victim_mask = (df['VS011'] == '1') | (df['VS013'] == '1') | (df['VS015'] == '1') | \
                          (df['VS017'] == '1') | (df['VS019'] == '1') | (df['VS021'] == '1')
        
        df = df[victim_mask].copy()

        if year == 2008:
            attempt_mask = (df['VS012'] == '2') & (df['VS014'] == '2') & (df['VS016'] == '2') & \
                           (df['VS018'] == '2') & (df['VS020'] == '2')
            df = df[~attempt_mask]
        elif year == 2012:
            df = df[df['VS098'] != '009']
        elif year == 2014:
            df = df[df['VS093'] != '009']
        elif year in [2016, 2018]:
            df = df[df['VS093'] != '9']

        harmonized_df = process_yearly_data(df, year, variable_harmonization_rules, is_victim_only=True)
        processed_yearly_data.append(harmonized_df)

    if processed_yearly_data:
        final_df = pd.concat(processed_yearly_data, ignore_index=True)
        final_df.to_parquet(output_filename, index=False)
        print(f"\nVictim data saved to '{output_filename}'.")

    # --- PART 2: ALL RESPONDENTS ---
    processed_all_data = []
    for year, path in raw_data_paths.items():
        year = int(year)
        print(f"\n--- Processing Year: {year} (All Respondents) ---")
        try:
            df = pd.read_csv(path, sep='\t', low_memory=False, dtype=str)
        except Exception: continue

        if year == 2021:
            victim_mask = (df['VS012'] == '1') | (df['VS017'] == '1') | (df['VS019'] == '1') | \
                          (df['VS017E'] == '1') | (df['VS041'] == '1') | (df['VS063'] == '1')
        elif year in [2018, 2016, 2014]:
            victim_mask = (df['VS012'] == '1') | (df['VS017'] == '1') | (df['VS019'] == '1') | \
                          (df['VS041'] == '1') | (df['VS063'] == '1')
        elif year == 2012:
            victim_mask = (df['VS012'] == '1') | (df['VS016'] == '1') | (df['VS018'] == '1') | \
                          (df['VS041'] == '1') | (df['VS067'] == '1')
        elif year == 2008:
            victim_mask = (df['VS011'] == '1') | (df['VS013'] == '1') | (df['VS015'] == '1') | \
                          (df['VS017'] == '1') | (df['VS019'] == '1') | (df['VS021'] == '1')
        
        if year == 2008:
            attempt_mask = (df['VS012'] == '2') & (df['VS014'] == '2') & (df['VS016'] == '2') & \
                           (df['VS018'] == '2') & (df['VS020'] == '2')
        elif year == 2012:
            attempt_mask = (df['VS098'] == '009')
        elif year == 2014:
            attempt_mask = (df['VS093'] == '009')
        elif year in [2016, 2018]:
            attempt_mask = (df['VS093'] == '9')
        else:
            attempt_mask = pd.Series(False, index=df.index)

        df['IS_SUCCESSFUL_VICTIM'] = victim_mask & ~attempt_mask
        harmonized_df = process_yearly_data(df, year, variable_harmonization_rules, is_victim_only=False)
        processed_all_data.append(harmonized_df)

    if processed_all_data:
        all_respondents_df = pd.concat(processed_all_data, ignore_index=True)
        all_respondents_df.to_parquet(output_all_respondents_filename, index=False)
        print(f"Full data saved to '{output_all_respondents_filename}'.")


if __name__ == '__main__':
    main()