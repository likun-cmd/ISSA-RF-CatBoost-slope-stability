import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator

matplotlib.use('TkAgg')
# 1. Load data
distributions = pd.read_excel("data.xlsx")

# 2. Use standard Greek letters to replace mathematical symbol characters
variable_mapping = {
    'γ': 'γ',  # gamma U+03B3
    'φ': 'φ',  # phi U+03C6
    'α': 'α'   # alpha U+03B1
}
variables = ['γ', 'C', 'φ', 'α', 'H', 'ru']  #

# Set y-axis range for each variable
y_ranges = {
    'γ': [0, 40],
    'C': [0, 800],
    'φ': [0, 60],
    'α': [0, 70],
    'H': [0, 600],
    'ru': [0, 0.5]
}


def generate_contrast_colors(n):
    main_colors = sns.husl_palette(n_colors=n, h=0.01)
    contrast_colors = []
    for color in main_colors:
        # Convert color to HSV space
        h, s, v = color[0], color[1], color[2]
        # Adjust hue to create contrast color
        h_contrast = (h + 0.5) % 1.0  # Complementary color
        # Reduce saturation and increase brightness
        contrast_color = hsv_to_rgb([h_contrast, s * 0.6, min(v * 1.2, 1.0)])
        contrast_colors.append(contrast_color)
    return main_colors, contrast_colors


main_colors, scatter_colors = generate_contrast_colors(6)

# 3. Set font configuration supporting Greek letters
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,  # Thicken axes
    'lines.linewidth': 1.2,
    'savefig.dpi': 600,
    'figure.dpi': 100,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False  # Critical: disable LaTeX engine
})

# 4. Create canvas and subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Generate 6 different colors
colors = sns.husl_palette(n_colors=6, h=0.01)

# 5. Loop to draw each subplot
for idx, var in enumerate(variables):
    ax = axes[idx]
    color = main_colors[idx]
    scatter_color = scatter_colors[idx]

    # Get data corresponding to the original column name
    original_var = next(
        k for k, v in variable_mapping.items() if v == var) if var in variable_mapping.values() else var
    subset = distributions[original_var].dropna()

    # Draw combined chart - remove transparency
    sns.violinplot(y=subset, inner=None, color=color,
                   linewidth=0.8, cut=0, ax=ax, saturation=0.85)
    sns.boxplot(y=subset, width=0.15, color='white',
                linewidth=1.2, flierprops={'marker': 'o', 'markersize': 3, 'markeredgecolor': 'black'},
                showmeans=True,
                meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markersize': 5},
                medianprops={'color': color, 'linewidth': 2},
                ax=ax)
    # Draw scatter points with contrast color - remove transparency
    sns.stripplot(y=subset, color=scatter_color, alpha=1.0,  # Remove transparency
                  size=3, jitter=0.2, edgecolor='none', ax=ax)

    # Set specified y-axis range
    ax.set_ylim(y_ranges[var])

    # Set inward tick marks
    ax.tick_params(axis='both', which='major',
                   direction='in',
                   bottom=True, left=True, top=False, right=False,
                   length=6)  # Thicken major tick marks

    ax.tick_params(axis='both', which='minor',
                   direction='in',
                   bottom=True, left=True, top=False, right=False,
                   length=4)  # Thicken minor tick marks

    # Add minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Statistical annotation (keep English display) - remove transparency
    mean = subset.mean()
    sem = subset.sem()
    ax.text(0.05, 0.95,
            f"Mean = {mean:.1f}±{sem:.1f}\nn = {len(subset)}",
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=1.0, edgecolor='gray', pad=2))  # Remove transparency

    # Display title using mapped variable names
    ax.set_title(f'Parameter {var}', pad=12, fontweight='bold')  # Display Unicode characters directly

    # Hide x-axis labels and ticks
    ax.set_xlabel('')
    ax.set_xticks([])

    # Set y-axis label
    if idx % 3 == 0:  # Only display y-axis label on left subplots
        ax.set_ylabel('Value', labelpad=10, fontsize=10)
    else:
        ax.set_ylabel('')

    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # Use despine to remove unnecessary borders, keeping only left and bottom
    sns.despine(ax=ax, left=False, bottom=True, top=True, right=True)


# Adjust bottom margin to leave space for figure title
plt.subplots_adjust(bottom=0.08, wspace=0.3, hspace=0.4)  # Adjust subplot spacing

# 6. Adjust layout and save
plt.tight_layout(pad=3.0)

# Save as high-resolution images
plt.savefig('Multi_Axis_Plot.pdf', bbox_inches='tight', dpi=600)
plt.savefig('Multi_Axis_Plot.png', bbox_inches='tight', dpi=600, transparent=False)  # Remove transparent background
plt.show()
