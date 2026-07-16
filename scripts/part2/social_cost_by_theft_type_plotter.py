import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# To run this script, make sure you are in the root (IDT_and_Breaches) directory and run:
# python scripts/part2/social_cost_by_theft_type_plotter.py

"""
Plots: Social Cost by Identity Theft Type

Generates two publication-ready figures from the theft-type cost analysis:

  Plot A — Cost per Victim by Theft Type, Longitudinal
    Line chart showing total social cost per victim (2021$) for each of the
    five theft types across all survey years, with the all-victim overall
    average from social_cost_analysis.csv overlaid as a dashed black reference.

  Plot B — Cost Relative to Overall Average by Theft Type, 2021
    Horizontal bar chart showing each theft type's cost per victim as a
    multiple of the all-victim overall average for 2021. A vertical reference
    line at 1.0x marks the overall average. Types ordered by multiple ascending
    so the most expensive sits at the top.

Inputs:
    - data/processed/part2/breach_cost_analysis/social_cost_by_theft_type.csv
    - data/processed/part2/social_cost/social_cost_analysis.csv

Outputs (PNG + PDF):
    - plots/part2/by_theft/plotA_cost_per_victim_by_theft_type_longitudinal
    - plots/part2/by_theft/plotB_cost_relative_to_overall_2021
"""

# --- STYLE (matches incident_trends_plotter.py) ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 22,
    'figure.titlesize': 30,
    'lines.linewidth': 6,
    'lines.markersize': 12,
})

# Theft type color map — matches COLOR_MAP_TYPES in incident_trends_plotter.py
COLOR_MAP = {
    'Existing Bank Account Misuse':   'tomato',
    'Existing Credit Card Misuse':    'teal',
    'Other Existing Account Misuse':  'purple',
    'New Account Fraud':              'darkorange',
    'Other Misuse of Personal Info':  'forestgreen',
}


def save_dual_formats(output_folder, base_name):
    plt.savefig(os.path.join(output_folder, f"{base_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, f"{base_name}.pdf"), bbox_inches='tight')


def plot_A(df_type, df_overall, output_folder):
    """
    Line chart: Total Social Cost per Victim by theft type over time.
    Overall average overlaid as dashed black reference line.
    """
    print("Generating Plot A: Cost per Victim by Theft Type, Longitudinal...")

    _, ax = plt.subplots(figsize=(14, 12))

    # One line per theft type (excluding 'Multiple Types' overlap row)
    for theft_type, color in COLOR_MAP.items():
        subset = df_type[df_type['Theft Type'] == theft_type].sort_values('Year')
        if subset.empty:
            continue
        ax.plot(
            subset['Year'],
            subset['Total Social Cost per Victim ($)'],
            marker='o',
            color=color,
            label=theft_type,
        )

    # Overall average — dashed black reference
    overall = df_overall.sort_values('Year')
    ax.plot(
        overall['Year'],
        overall['Total Social Cost per Victim ($)'],
        marker='o',
        color='black',
        linestyle='--',
        linewidth=4,
        markersize=10,
        label='All Victims (Overall Avg.)',
    )

    ax.set_title('Social Cost per Victim by Theft Type (2021$)', pad=20)
    ax.set_ylabel('Avg. Social Cost per Victim (2021$)', labelpad=15)
    ax.set_xlabel('Year', labelpad=15)
    ax.set_xticks(sorted(df_type['Year'].unique()))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='white')

    plt.tight_layout()
    save_dual_formats(output_folder, 'plotA_cost_per_victim_by_theft_type_longitudinal')
    plt.close()
    print("  Saved plotA.")


def plot_B(df_type, df_overall, output_folder):
    """
    Horizontal bar chart: Each theft type's cost as a multiple of the
    all-victim overall average for 2021. Reference line at 1.0x.
    Types ordered by multiple ascending (most expensive at top).
    """
    print("Generating Plot B: Cost Relative to Overall Average by Theft Type, 2021...")

    overall_2021 = df_overall[df_overall['Year'] == 2021]['Total Social Cost per Victim ($)'].iloc[0]

    df_2021 = df_type[
        (df_type['Year'] == 2021) &
        (df_type['Theft Type'] != 'Multiple Types')
    ].copy()

    df_2021['Multiple of Overall'] = df_2021['Total Social Cost per Victim ($)'] / overall_2021
    df_2021 = df_2021.sort_values('Multiple of Overall', ascending=True)

    colors = [COLOR_MAP.get(t, 'steelblue') for t in df_2021['Theft Type']]

    _, ax = plt.subplots(figsize=(20, 10))

    bars = ax.barh(
        df_2021['Theft Type'],
        df_2021['Multiple of Overall'],
        color=colors,
        edgecolor='white',
        linewidth=0.8,
    )

    # Reference line at 1.0x (overall average)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=3, label='Overall Average (1.0×)')

    # Annotate each bar with the multiple and the dollar value
    for bar, (_, row) in zip(bars, df_2021.iterrows()):
        multiple = row['Multiple of Overall']
        cost = row['Total Social Cost per Victim ($)']
        ax.text(
            multiple + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f'{multiple:.2f}× (${cost:,.0f})',
            va='center',
            ha='left',
            fontsize=20,
            fontweight='bold',
        )

    ax.set_title('Social Cost Relative to Overall Average by Theft Type, 2021', pad=20)
    ax.set_xlabel('Multiple of All-Victim Average', labelpad=15)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.1f}×'))
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white')

    plt.tight_layout()
    save_dual_formats(output_folder, 'plotB_cost_relative_to_overall_2021')
    plt.close()
    print("  Saved plotB.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    type_csv    = os.path.join(root_dir, 'data/processed/part2/breach_cost_analysis/social_cost_by_theft_type.csv')
    overall_csv = os.path.join(root_dir, 'data/processed/part2/social_cost/social_cost_analysis.csv')
    output_folder = os.path.join(root_dir, 'plots/part2/by_theft')
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading theft-type cost data from '{type_csv}'...")
    try:
        df_type = pd.read_csv(type_csv)
    except FileNotFoundError:
        print(f"ERROR: '{type_csv}' not found. Run social_cost_by_theft_type.py first.")
        return

    print(f"Loading overall social cost data from '{overall_csv}'...")
    try:
        df_overall = pd.read_csv(overall_csv)
    except FileNotFoundError:
        print(f"ERROR: '{overall_csv}' not found. Run calculate_social_cost.py first.")
        return

    plot_A(df_type, df_overall, output_folder)
    plot_B(df_type, df_overall, output_folder)

    print(f"\nAll plots saved to '{output_folder}'.")


if __name__ == '__main__':
    main()
