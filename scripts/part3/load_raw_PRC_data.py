import sqlite3
import pandas as pd
import os
# To run: python scripts/part3/load_raw_PRC_data.py
# Define paths based on your repository structure
root_dir = os.getcwd() 
db_path = os.path.join(root_dir, 'data/raw/PRC/PRC_DataBreachChronology_v2_202512/Data_Breach_Chronology.db')
output_path = os.path.join(root_dir, 'data/raw/PRC/loaded/clean_prc_base.csv')

def load_and_verify_prc():
    """
    Loads raw PRC data from SQLite, discovers the correct table, 
    deduplicates based on group_uuid, and prints a sanity check report.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file NOT found at: {db_path}")
        return

    with sqlite3.connect(db_path) as conn:
        # Step 1: Discover the actual table names in the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print(f"Error: The database at {db_path} is EMPTY (no tables found).")
            return
        
        # Determine the target table based on your previous run
        target_table = 'Data_Breach_Chronology_by_PrivacyRightsClearinghouse'
        if target_table not in tables:
            target_table = tables[0]
            
        print(f"Reading from table: {target_table}...")

        # Step 2: Load the data into a DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {target_table}", conn)

    # Initial counts and data prep
    total_raw_reports = len(df)
    
    # Standardizing 'total_affected' as numeric for comparison [cite: 1040, 1201]
    df['total_affected'] = pd.to_numeric(df['total_affected'], errors='coerce').fillna(0)
    zero_exposed_count = (df['total_affected'] == 0).sum()

    # Step 3: Deduplication logic using group_uuid [cite: 1100, 1220]
    # We sort by total_affected (descending) to ensure we keep the report 
    # with the highest impact number for each unique group.
    group_col = 'group_uuid' if 'group_uuid' in df.columns else 'id'
    df_sorted = df.sort_values(by=[group_col, 'total_affected'], ascending=[True, False])
    
    # Identify groups with more than one report before dropping them
    duplicate_groups = df_sorted[df_sorted.duplicated(subset=[group_col], keep=False)]
    
    # Perform the drop
    clean_df = df_sorted.drop_duplicates(subset=[group_col], keep='first')

    # Calculate zero records for UNIQUE events only
    unique_zero_exposed = (clean_df['total_affected'] == 0).sum()
    unique_total = len(clean_df)
    zero_percent = (unique_zero_exposed / unique_total) * 100

    # Print updated Sanity Check
    print("\n" + "="*40)
    print("--- PRC DATA LOAD SANITY CHECK ---")
    print(f"Total Raw Reports loaded:      {total_raw_reports}")
    print(f"Duplicate Reports dropped:     {total_raw_reports - unique_total}")
    print(f"Total Unique Breach Events:    {unique_total}")
    print(f"Unique Events with 0 records:  {unique_zero_exposed} ({zero_percent:.1f}%)")
    print("="*40)

    # Step 5: Print examples of dropped duplicates
    if not duplicate_groups.empty:
        print("\n--- EXAMPLES OF DROPPED DUPLICATES (Same Event, Multiple Reports) ---")
        # Show up to 3 groups that had duplicates
        example_ids = duplicate_groups[group_col].unique()[:3]
        
        for g_id in example_ids:
            group_data = duplicate_groups[duplicate_groups[group_col] == g_id]
            kept_record = group_data.iloc[0] 
            dropped_records = group_data.iloc[1:]
            
            print(f"\nGroup ID: {g_id}")
            print(f"  [KEPT]    {kept_record['org_name']} | Affected: {kept_record['total_affected']} | Source: {kept_record['source']}")
            for _, dropped in dropped_records.iterrows():
                print(f"  [DROPPED] {dropped['org_name']} | Affected: {dropped['total_affected']} | Source: {dropped['source']}")
        print("\n" + "-"*60)

    # Step 6: Save to the clean base directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"Saved clean base to: {output_path}")

if __name__ == "__main__":
    load_and_verify_prc()