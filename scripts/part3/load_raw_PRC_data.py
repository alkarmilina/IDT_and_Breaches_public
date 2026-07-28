import sqlite3
import pandas as pd
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part3/load_raw_PRC_data.py

"""
Part 3: PRC Breach Chronology Cleaning

Loads the raw PRC breach chronology database and deduplicates it. PRC
assigns the same group_uuid to multiple state-level reports of the same
underlying breach event, so each group is collapsed to a single record,
keeping whichever report has the highest affected count.

Setup:
Reads the raw PRC SQLite database.

Goal:
Produce a single cleaned dataset of unique breach events, the starting
point for the rest of the PRC augmentation pipeline.

Outputs:
- data/raw/PRC/loaded/clean_prc_base.csv: one row per unique breach
  event.
"""


def load_and_verify_prc():
    print("\n--- Part 3: PRC Breach Chronology Cleaning ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    db_path = os.path.join(root_dir, 'data/raw/PRC/PRC_DataBreachChronology_v2_202512/Data_Breach_Chronology.db')
    output_path = os.path.join(root_dir, 'data/raw/PRC/loaded/clean_prc_base.csv')

    if not os.path.exists(db_path):
        print(f"Error: database not found at {db_path}")
        return

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            print(f"Error: database at {db_path} has no tables.")
            return
        target_table = 'Data_Breach_Chronology_by_PrivacyRightsClearinghouse'
        if target_table not in tables:
            target_table = tables[0]
        print(f"Reading from table: {target_table}...")
        df = pd.read_sql_query(f"SELECT * FROM {target_table}", conn)
        print(f"Loaded {len(df):,} rows from {target_table}")

    total_raw_reports = len(df)
    df['total_affected'] = pd.to_numeric(df['total_affected'], errors='coerce').fillna(0)

    # For each group_uuid, keep the record with the highest affected count.
    group_col = 'group_uuid' if 'group_uuid' in df.columns else 'id'
    df_sorted = df.sort_values(by=[group_col, 'total_affected'], ascending=[True, False])
    duplicate_groups = df_sorted[df_sorted.duplicated(subset=[group_col], keep=False)]
    clean_df = df_sorted.drop_duplicates(subset=[group_col], keep='first')

    unique_zero_exposed = (clean_df['total_affected'] == 0).sum()
    unique_total = len(clean_df)
    zero_percent = (unique_zero_exposed / unique_total) * 100

    print(f"\nTotal raw reports:      {total_raw_reports}")
    print(f"Duplicates dropped:     {total_raw_reports - unique_total}")
    print(f"Unique breach events:   {unique_total}")
    print(f"Events with 0 records:  {unique_zero_exposed} ({zero_percent:.1f}%)")

    if not duplicate_groups.empty:
        print("\nExamples of dropped duplicates:")
        for g_id in duplicate_groups[group_col].unique()[:3]:
            group_data = duplicate_groups[duplicate_groups[group_col] == g_id]
            kept = group_data.iloc[0]
            print(f"\n  Group {g_id}")
            print(f"  [kept]    {kept['org_name']} | {kept['total_affected']} affected | {kept['source']}")
            for _, dropped in group_data.iloc[1:].iterrows():
                print(f"  [dropped] {dropped['org_name']} | {dropped['total_affected']} affected | {dropped['source']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    load_and_verify_prc()
