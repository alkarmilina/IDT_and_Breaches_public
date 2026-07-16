import pandas as pd
import numpy as np
import os
from thefuzz import fuzz, process

# To run: python scripts/part3/find_nu.py

"""
Part 3: Undisclosed Record Estimation (Finding n_u)
Calculates the annual baseline for undisclosed records (n_u) by applying 
Sector Medians to missing state incidents to avoid record inflation.
"""

# --- REUSING CLASSIFICATION LOGIC ---
def classify_nh_entity(org_name):
    """Strict Allowlist for NH (Small Dataset)"""
    org_lower = str(org_name).lower()
    national_allowlist = [
        "hewlett packard", "pepsi", "toyota", "orbitz", "merrill lynch", "aetna",
        "warby parker", "oxo", "ameriprise", "quest diagnostics", "regal entertainment",
        "alere", "benchmade", "wilton brands", "topps", "djo", "sourcefire",
        "lokai", "la jolla", "baylor", "six red marbles", "workers united", 
        "gloria jean", "dutch bros"
    ]
    local_blocklist = [
        "upper connecticut valley", "enterprise bank", "greater boston catholic",
        "women infants hospital", "fiondella", "geocomp", "clearview realty",
        "fonville morisey", "florida digestive", "mt diablo", "south carolina revenue",
        "upper skagit", "barlow respiratory", "buffalo lodging", "brandywine pediatrics"
    ]
    for n in national_allowlist:
        if n in org_lower: return True 
    for l in local_blocklist:
        if l in org_lower: return False 
    return False

def classify_maine_entity(org_name):
    """Blocklist for Maine (Large Dataset)"""
    org_lower = str(org_name).lower()
    local_keywords = [
        "school", "elementary", "high school", "district", "academy", "college", "university",
        "town of", "city of", "village of", "county", "police", "sheriff", "municipality",
        "dental", "dentist", "orthodontic", "family practice", "pediatrics", "clinic",
        "eye care", "optometry", "chiropractic", "veterinary", "animal hospital",
        "surgery center", "medical center", "hospital", 
        "credit union", "fcu", "savings bank", "community bank",
        "realty", "real estate", "properties", "condo", "apartments",
        "auto", "motors", "dealership", "nissan of", "toyota of", "ford of", 
        "restaurant", "grill", "pizza", "cafe", "bakery", "brewing",
        "plumbing", "heating", "electric", "landscaping", "construction"
    ]
    for keyword in local_keywords:
        if keyword in org_lower: return False
    return True

def main():
    root_dir = os.getcwd()
    prc_data_path = os.path.join(root_dir, 'data/processed/part3/prc_hhs_integrated.csv')
    
    # State-level raw files from PRC_supplements
    nh_path = os.path.join(root_dir, 'data/raw/PRC_supplements/NH_Manual_Entry_Template.csv')
    me_path_1 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine1.xlsx')
    me_path_2 = os.path.join(root_dir, 'data/raw/PRC_supplements/Maine2.xlsx')

    # --- 1. ROBUST STATE LOADING ---
    def load_state(path, state_type):
        if not os.path.exists(path): return pd.DataFrame()
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        
        col_map = {
            'org': ['Organization', 'org_name', '02_03_01_Entity Name', 'Entity Name', 'Company', 'DATA BREACH NOTICES'],
            'affected': ['NH_Affected', 'maine_affected', '03_01_02_Total number of Maine residents affected', 
                         'Maine residents affected', 'Count', 'Unnamed: 6']
        }
        
        found_org = [c for c in col_map['org'] if c in df.columns]
        found_aff = [c for c in col_map['affected'] if c in df.columns]
        
        if found_org and found_aff:
            df = df[[found_org[0], found_aff[0]]].copy()
            df.columns = ['Organization', 'X_State']
            df['X_State'] = pd.to_numeric(df['X_State'], errors='coerce').fillna(0)
            df = df.dropna(subset=['Organization'])
            
            # APPLY CLASSIFICATION IMMEDIATELY
            if state_type == 'NH':
                df['is_national'] = df['Organization'].apply(classify_nh_entity)
            else:
                df['is_national'] = df['Organization'].apply(classify_maine_entity)
            
            return df
        return pd.DataFrame()

    print("Loading state and national data...")
    # Load and tag data
    nh_df = load_state(nh_path, 'NH')
    me_df = pd.concat([load_state(me_path_1, 'ME'), load_state(me_path_2, 'ME')], ignore_index=True)
    
    state_full = pd.concat([nh_df, me_df], ignore_index=True)
    state_full = state_full.sort_values('X_State', ascending=False).drop_duplicates('Organization')
    state_full['clean_org'] = state_full['Organization'].astype(str).str.lower().str.strip()

    # FILTER: Only keep National-Scale entities for gap analysis
    # If a local entity is missing from PRC, we don't care (it doesn't add to national magnitude)
    state_filtered = state_full[state_full['is_national'] == True].copy()

    print(f"Total State Records: {len(state_full)}")
    print(f"Records Kept for Gap Analysis (National Scale): {len(state_filtered)}")

    df_prc = pd.read_csv(prc_data_path)
    national_names = df_prc['org_name'].astype(str).str.lower().str.strip().unique().tolist()

    # --- 2. SECTOR MEDIANS ---
    # Megabreaches to exclude from 'typical' weight calculation
    MEGABREACHES = [12500000, 130000000, 77000000, 40000000, 56000000, 143000000]
    known_counts = df_prc[(df_prc['total_affected'] > 0) & (~df_prc['total_affected'].isin(MEGABREACHES))]
    sector_medians = known_counts.groupby('breach_type')['total_affected'].median().to_dict()
    overall_median = known_counts['total_affected'].median()

    # --- 3. FUZZY GAP ANALYSIS ---
    print(f"Performing Fuzzy Match on {len(state_filtered)} national-scale state orgs...")
    missing_count = 0
    nu_from_missing_orgs = 0
    
    # Iterate ONLY through the filtered national-scale entities
    for idx, row in state_filtered.iterrows():
        query = row['clean_org']
        if not query or len(query) < 2: continue
            
        # Check for matches
        match = process.extractOne(query, national_names, scorer=fuzz.token_sort_ratio)
        
        if not match or match[1] < 90:
            # Paper Logic: Use the typical (median) size for unknown events
            # Citation: "calculated a typical weight for each category using the known, non-outlier data" 
            nu_from_missing_orgs += overall_median
            missing_count += 1

    # --- 4. PRC INTERNAL UNKNOWNS ---
    # Records that are in the PRC but have a 0 count
    # Citation: "We distributed that volume across the zero-count incidents proportionally by year" [cite: 151]
    
    # Note: We do NOT filter PRC data by the blocklist. 
    # If it's in the PRC database, it's considered a relevant breach event by definition of the source.
    pure_unknowns = df_prc[df_prc['total_affected'] == 0].copy()

    nu_from_sector_weights = 0
    for b_type in ['CARD', 'DISC', 'HACK', 'INSD', 'PHYS', 'PORT', 'STAT', 'UNKN']:
        count = len(pure_unknowns[pure_unknowns['breach_type'] == b_type])
        median_weight = sector_medians.get(b_type, overall_median)
        nu_from_sector_weights += (count * median_weight)

    # --- 5. FINALIZE ---
    TOTAL_YEARS = 14
    total_nu = nu_from_missing_orgs + nu_from_sector_weights
    annual_nu = total_nu / TOTAL_YEARS

    print("\n" + "="*45)
    print(f"Missing Orgs (National Scale Only):  {missing_count}")
    print(f"Records from Missing Orgs:           {nu_from_missing_orgs:,.0f}")
    print(f"Records from PRC Unknowns:           {nu_from_sector_weights:,.0f}")
    print(f"Total Undisclosed (14 Years):        {total_nu:,.0f}")
    print("-" * 45)
    print(f"FINAL ANNUAL n_u BASELINE:           {annual_nu:,.0f}")
    print("="*45)

if __name__ == "__main__":
    main()