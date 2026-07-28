import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/incident_trends_plotter.py

"""
Part 2: Incident Trend Plots

Plots year-by-year trends in how identity theft occurred: theft method,
theft type, discovery method, and social cost, plus average social cost
broken down by demographic group.

Setup:
Reads the harmonized victim dataset from part1, and the per-demographic
social cost tables written by calculate_social_cost.py. The theft
method and discovery method percentages each include a residual "Don't
Know" category, computed as whatever weighted mass isn't accounted for
by the named categories. This captures both the survey's explicit
"Don't Know" code and respondents who gave no answer at all, matching
how the paper reports this category.

Goal:
Produce the paper's plots of theft method, theft type, discovery
method, and social cost trends over time, plus social cost by
demographic group.

Outputs:
- plots/part2/plot1_theft_methods_percentage / _count (.png/.pdf)
- plots/part2/plot2_theft_types_percentage / _count (.png/.pdf)
- plots/part2/plot3_discovery_percentage / _count (.png/.pdf)
- plots/part2/plot4_social_cost_evolution (.png/.pdf)
- plots/part2/plot5_oop_loss_by_type (.png/.pdf)
- plots/part2/plot6-9_social_cost_by_<demographic> (.png/.pdf)
"""

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 28,
    'figure.titlesize': 30,
    'lines.linewidth': 6,
    'lines.markersize': 12
})

RENAME_MAP = {
    "no schooling/kindergarten": "< High School",
    "elementary school (grades 1-8)": "< High School",
    "some high school (grades 9-12, no diploma)": "< High School",
    "high school graduate (or equivalent)": "High School Grad",
    "some college/associate degree": "High School Grad",
    "bachelor's degree": "Bachelor's",
    "graduate/professional degree": "Grad / Prof Degree",
    "white only": "White", "black only": "Black", "hispanic": "Hispanic",
    "asian only": "Asian", "american indian/alaska native only": "Native American",
    "hawaiian/pacific islander only": "Pacific Islander",
    "less than $5,000": "< 15k", "$5,000 to $7,499": "< $15k",
    "$7,500 to $9,999": "< $15k", "$10,000 to $12,499": "< $15k",
    "$12,500 to $14,999": "< $15k", "$15,000 to $17,499": "$15k – $25k",
    "$17,500 to $19,999": "$15k – $25k", "$20,000 to $24,999": "$15k – $25k",
    "$25,000 to $29,999": "$25k – $35k", "$30,000 to $34,999": "$25k – $35k",
    "$35,000 to $39,999": "$35k – $50k", "$40,000 to $49,999": "$35k – $50k",
    "$50,000 to $74,999": "$50k – $75k", "$75,000 and over": "> 75k"
}

THEFT_METHOD_MAP = {
    1: "Lost or Stolen Physical Item", 2: "Stolen During a Transaction",
    3: "Stolen from Computer/Device", 4: "Deceived by Scam/Phishing",
    5: "Stolen from Company (Data Breach)", 7: "Other"
}

DISCOVERY_MAP = {
    1: "Noticed Problem", 2: "Notified by Bank", 3: "Notified by Other",
    4: "Received Bill", 5: "Credit Report", 7: "Other"
}

THEFT_TYPE_COLS = {
    'EXISTING_BANK_ACCT_MISUSE_12MO': 'Existing Bank Account Misuse',
    'EXISTING_CC_MISUSE_12MO': 'Existing Credit Card Misuse',
    'EXISTING_OTHER_ACCT_MISUSE_12MO': 'Other Existing Acct Misuse',
    'NEW_ACCT_OPENED_12MO': 'New Account Fraud',
    'OTHER_FRAUDULENT_PURPOSE_12MO': 'Other Misuse of PI'
}

COLOR_MAP_TYPES = {
    'Existing Bank Account Misuse': 'tomato',
    'Existing Credit Card Misuse': 'teal',
    'Other Existing Acct Misuse': 'purple',
    'New Account Fraud': 'darkorange',
    'Other Misuse of PI': 'forestgreen'
}


def save_dual_formats(output_folder, base_name):
    """Saves the current matplotlib figure as both PNG and PDF under the same base file name."""
    plt.savefig(os.path.join(output_folder, f"{base_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, f"{base_name}.pdf"), bbox_inches='tight')


def weighted_percentage(df, col, val, weight_col, year, total_dict, is_categorical=True):
    """Weighted percentage of one year's victims where col == val (or col == 1 if is_categorical is False), out of that year's total weighted victim count."""
    group = df[df['year'] == year]
    total = total_dict.get(year, 0)
    if total == 0: return 0
    if is_categorical:
        subset = group[group[col] == val]
    else:
        subset = group[group[col] == 1]
    return (subset[weight_col].sum() / total) * 100


def weighted_count(df, col, val, weight_col, year, is_categorical=True):
    """Weighted count of one year's victims where col == val (or col == 1 if is_categorical is False)."""
    group = df[df['year'] == year]
    if is_categorical:
        subset = group[group[col] == val]
    else:
        subset = group[group[col] == 1]
    return subset[weight_col].sum() if not subset.empty else 0


def weighted_mean_loss(df, condition_col, loss_col, weight_col, year):
    """Weighted average of loss_col among one year's victims where condition_col == 1."""
    mask = (df['year'] == year) & (df[condition_col] == 1) & (df[loss_col].notna())
    group = df[mask]
    if group.empty or group[weight_col].sum() == 0: return 0
    return (group[loss_col] * group[weight_col]).sum() / group[weight_col].sum()


def plot_demographic_costs(folder, output_folder, file_name, group_col, fig_num):
    """Plots average social cost per victim by demographic group, reading one of calculate_social_cost.py's per-demographic CSV outputs."""
    path = os.path.join(folder, file_name)
    if not os.path.exists(path): return  # This demographic's CSV hasn't been generated yet.
    df = pd.read_csv(path).set_index(group_col)
    print(f"Loaded data from {path}")

    def mapper(x):
        x_str = str(x).lower().strip()
        # A hyphen in a race combination (e.g. "White-Black") means multiracial, but age
        # ranges ("50-64") and income ranges also contain hyphens, so numeric-looking
        # labels are excluded from that check.
        is_age_or_income = any(char.isdigit() for char in x_str) or '$' in x_str
        if '-' in x_str and not is_age_or_income: return "Two or more races"
        if 'two or three races' in x_str or 'four or five races' in x_str: return "Two or more races"
        return RENAME_MAP.get(x_str, str(x))

    df.index = df.index.map(mapper)
    df = df.groupby(df.index).mean()

    if 'age' in file_name.lower(): order = ['16-29', '30-49', '50-64', '65+']
    elif 'income' in file_name.lower(): order = ['< 15k', '$15k – $25k', '$25k – $35k', '$35k – $50k', '$50k – $75k', '> 75k']
    elif 'education' in file_name.lower(): order = ['< High School', 'High School Grad', "Bachelor's", 'Grad / Prof Degree']
    elif 'race' in file_name.lower(): order = ['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Pacific Islander', 'Two or more races']
    else: order = df.index.tolist()

    df = df.reindex([o for o in order if o in df.index])

    plt.figure(figsize=(14, 12))
    ax = df.plot(kind='bar', stacked=True, color=['#5b84b1', '#a62c2b', '#2F4F4F', '#A9A9A9'], ax=plt.gca(), legend=False)
    plt.title(f'Social Cost by {group_col}', pad=20)
    plt.ylabel('Average Cost per Victim (2021$)', labelpad=15)
    plt.xlabel(group_col, labelpad=15)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%1.0f'))
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_name = f'plot{fig_num}_social_cost_by_{group_col.lower().replace(" ", "_").replace("/", "_")}'
    save_dual_formats(output_folder, save_name)
    plt.close('all')


def main():
    print("\n--- Part 2: Incident Trend Plots ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    input_path = os.path.join(root_dir, 'data/processed/part1/its_victims.parquet')
    sc_folder = os.path.join(root_dir, 'data/processed/part2/social_cost')
    plots_folder = os.path.join(root_dir, 'plots/part2')
    os.makedirs(plots_folder, exist_ok=True)

    df = pd.read_parquet(input_path)
    print(f"Loaded data from {input_path}")
    calc_cols = ['year', 'FINAL_ITS_WEIGHT', 'THEFT_METHOD', 'HOW_DISCOVERED_MISUSE', 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT'] + list(THEFT_TYPE_COLS.keys())
    for col in [c for c in calc_cols if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT' in df.columns:
        df.loc[df['OUT_OF_POCKET_LOSS_RECENT_INCIDENT'] >= 999996, 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT'] = np.nan

    years = sorted(df['year'].unique())
    total_victims = df.groupby('year')['FINAL_ITS_WEIGHT'].sum().to_dict()

    print("Plotting theft methods...")
    for m in ['percentage', 'count']:
        plt.figure(figsize=(14, 12))
        plot_data = {'Year': years}

        for code, label in THEFT_METHOD_MAP.items():
            if m == 'percentage':
                plot_data[label] = [weighted_percentage(df, 'THEFT_METHOD', code, 'FINAL_ITS_WEIGHT', y, total_victims) for y in years]
            else:
                plot_data[label] = [weighted_count(df, 'THEFT_METHOD', code, 'FINAL_ITS_WEIGHT', y) for y in years]

        dk_vals = []
        for i, y in enumerate(years):
            known_sum = sum(plot_data[label][i] for label in THEFT_METHOD_MAP.values())
            total = 100 if m == 'percentage' else total_victims[y]
            dk_vals.append(max(0.01, total - known_sum))
        plot_data["Don't Know"] = dk_vals

        df_plot = pd.DataFrame(plot_data).melt('Year', var_name='Theft Method', value_name='Val')
        palette = {label: sns.color_palette("tab10")[i] for i, label in enumerate(THEFT_METHOD_MAP.values())}
        palette["Don't Know"] = "grey"

        sns.lineplot(data=df_plot, x='Year', y='Val', hue='Theft Method', marker='o', dashes=False, palette=palette, legend=(m == 'count'))

        plt.title(f"How Victim's Info was Obtained ({'%' if m == 'percentage' else 'Victim Count'})", pad=20)

        if m == 'percentage':
            plt.yscale('log')
            plt.ylabel('% of Victims (Log Scale)')
            plt.yticks([0.1, 1, 10, 50, 100], ['0.1%', '1%', '10%', '50%', '100%'])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, facecolor='white')
        else:
            plt.yscale('log'); plt.ylabel('Total Victims (Log Scale)')
            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))
            plt.legend(loc='lower right', frameon=True, facecolor='white')

        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_dual_formats(plots_folder, f'plot1_theft_methods_{m}')
        plt.close()

    print("Plotting theft types...")
    for m in ['percentage', 'count']:
        plt.figure(figsize=(14, 12))
        for col, label in THEFT_TYPE_COLS.items():
            color = COLOR_MAP_TYPES[label]
            vals = [weighted_percentage(df, col, 1, 'FINAL_ITS_WEIGHT', y, total_victims, False) if m == 'percentage'
                    else weighted_count(df, col, 1, 'FINAL_ITS_WEIGHT', y, False) for y in years]
            plt.plot(years, vals, marker='o', color=color, label=label)

        plt.title(f"Types of Identity Theft ({'%' if m == 'percentage' else 'Victim Count'})", pad=20)

        if m == 'percentage':
            plt.yscale('log')
            plt.ylabel('% of Victims (Log Scale)')
            plt.yticks([0.1, 1, 10, 50, 100], ['0.1%', '1%', '10%', '50%', '100%'])
            plt.legend(loc='upper right', frameon=True, facecolor='white')
        else:
            plt.yscale('log'); plt.ylabel('Total Victims (Log Scale)')
            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))
            plt.legend(loc='upper right', frameon=True, facecolor='white')

        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_dual_formats(plots_folder, f'plot2_theft_types_{m}')
        plt.close()

    print("Plotting discovery methods...")
    for m in ['percentage', 'count']:
        plt.figure(figsize=(14, 12))
        plot_data = {'Year': years}

        for code, label in DISCOVERY_MAP.items():
            if m == 'percentage':
                plot_data[label] = [weighted_percentage(df, 'HOW_DISCOVERED_MISUSE', code, 'FINAL_ITS_WEIGHT', y, total_victims) for y in years]
            else:
                plot_data[label] = [weighted_count(df, 'HOW_DISCOVERED_MISUSE', code, 'FINAL_ITS_WEIGHT', y) for y in years]

        dk_vals = []
        for i, y in enumerate(years):
            known_sum = sum(plot_data[label][i] for label in DISCOVERY_MAP.values())
            total = 100 if m == 'percentage' else total_victims[y]
            dk_vals.append(max(0.01, total - known_sum))
        plot_data["Don't Know"] = dk_vals

        df_plot = pd.DataFrame(plot_data).melt('Year', var_name='Method', value_name='Val')
        palette = {label: sns.color_palette("tab10")[i] for i, label in enumerate(DISCOVERY_MAP.values())}
        palette["Don't Know"] = "grey"

        sns.lineplot(data=df_plot, x='Year', y='Val', hue='Method', marker='o', dashes=False, palette=palette, legend=(m == 'count'))

        plt.title(f"How Victims Discovered Misuse ({'%' if m == 'percentage' else 'Victim Count'})", pad=20)

        if m == 'percentage':
            plt.yscale('log')
            plt.ylabel('% of Victims (Log Scale)')
            plt.yticks([0.1, 1, 10, 50, 100], ['0.1%', '1%', '10%', '50%', '100%'])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, facecolor='white')
        else:
            plt.yscale('log'); plt.ylabel('Total Victims (Log Scale)')
            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))
            plt.legend(loc='upper right', frameon=True, facecolor='white')

        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_dual_formats(plots_folder, f'plot3_discovery_{m}')
        plt.close()

    sc_path = os.path.join(sc_folder, 'social_cost_analysis.csv')
    if os.path.exists(sc_path):
        print("Plotting social cost evolution...")
        sc_df = pd.read_csv(sc_path)
        print(f"Loaded data from {sc_path}")
        _, ax1 = plt.subplots(figsize=(14, 12))
        sns.barplot(data=sc_df, x='Year', y=sc_df['Total National Social Cost ($)']/1e9, color='#4682B4', ax=ax1)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Total National Social Cost ($ Billions)', color='#4671a1')
        ax1.tick_params(axis='y', labelcolor='#4671a1')
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.1fB'))

        ax2 = ax1.twinx()
        sns.lineplot(x=range(len(sc_df)), y=sc_df['Total Social Cost per Victim ($)'], color='#a62c2b', marker='o', ax=ax2)
        ax2.set_ylabel('Total Social Cost per Victim ($)', color='#a62c2b')
        ax2.tick_params(axis='y', labelcolor='#a62c2b')
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))

        plt.title('Evolution of Social Cost (2008-2021)', pad=20)
        plt.tight_layout()
        save_dual_formats(plots_folder, 'plot4_social_cost_evolution')
        plt.close()

    print("Plotting average OOP loss by theft type...")
    plt.figure(figsize=(14, 12))
    for col, label in THEFT_TYPE_COLS.items():
        vals = [weighted_mean_loss(df, col, 'OUT_OF_POCKET_LOSS_RECENT_INCIDENT', 'FINAL_ITS_WEIGHT', y) for y in years]
        plt.plot(years, vals, label=label, marker='o', color=COLOR_MAP_TYPES[label])
    plt.title('Mean Out-of-Pocket Loss by Identity Theft Type', pad=20)
    plt.ylabel('Average Loss ($)')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, facecolor='white')
    plt.tight_layout()
    save_dual_formats(plots_folder, 'plot5_oop_loss_by_type')
    plt.close()

    print("Plotting social cost by demographic group...")
    demo_tasks = [
        ('social_cost_by_age_group.csv', 'Age Group', 6),
        ('social_cost_by_victim_race_ethnicity.csv', 'Victim Race/Ethnicity', 7),
        ('social_cost_by_victim_educational_attainment.csv', 'Victim Educational Attainment', 8),
        ('social_cost_by_victim_household_income.csv', 'Victim Household Income', 9)
    ]
    for file, col, num in demo_tasks:
        plot_demographic_costs(sc_folder, plots_folder, file, col, num)

    print(f"\nData saved to {plots_folder}")


if __name__ == '__main__':
    main()
