import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def main():
    # Load the same data used in your Wilcoxon test
    root_dir = os.getcwd()
    idt_path = os.path.join(root_dir, 'data/processed/part5/conversion_data.csv')
    df = pd.read_csv(idt_path)

    # We will recreate the "Deltas" from your Saturated Model results
    # These are the 11 shocks identified in your previous output
    deltas = [
        84879, 32727, 387005, 261617, 479299, 
        1097493, 391077, 465742, -994121, 591928, -282992
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Histogram with Normal Curve Fit
    count, bins, ignored = ax1.hist(deltas, bins=8, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    mu, std = np.mean(deltas), np.std(deltas)
    x = np.linspace(min(deltas), max(deltas), 100)
    p = stats.norm.pdf(x, mu, std)
    ax1.plot(x, p, 'r', linewidth=2, label='Normal Fit')
    ax1.set_title(f'Histogram of IDT Deltas\n(Mean: {mu:,.0f}, Std: {std:,.0f})')
    ax1.set_xlabel('Change in Estimated Victims (Delta)')
    ax1.legend()

    # 2. Q-Q Plot
    # A Q-Q plot compares the actual data points against where they 'should' be 
    # if the data were perfectly normal (the red line).
    stats.probplot(deltas, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')

    plt.tight_layout()
    plot_path = os.path.join(root_dir, 'reports/figures/normality_check.png')
    plt.savefig(plot_path)
    print(f"Normality plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()