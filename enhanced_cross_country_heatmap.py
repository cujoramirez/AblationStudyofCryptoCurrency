"""
Enhanced Cross-Country Policy Impact Comparison Visualization

This script creates a publication-quality heatmap showing the impact of different
policy scenarios across countries, specifically designed based on the ablation study results,
with improved visual styling suitable for academic publications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def ensure_dir(directory):
    """Ensure the output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_ablation_data():
    """Load ablation findings from CSV"""
    try:
        df = pd.read_csv('ablation_findings.csv')
        return df
    except Exception as e:
        print(f"Error loading ablation data: {e}")
        return None

def create_enhanced_cross_country_heatmap():
    """Create a publication-quality Cross-Country Policy Impact Comparison heatmap"""
    # Load the ablation data
    df = load_ablation_data()
    
    if df is None:
        print("Failed to load data.")
        return
    
    # Create output directory
    output_dir = 'enhanced_ablation_visualizations'
    ensure_dir(output_dir)
    
    # Create a better-structured dataset for the heatmap
    data = {
        'scenario': [
            'United States_with_Indonesia_policies',
            'max_regulations',
            'no_regulations',
            'United States_with_Russia_policies',
            'no_tax_no_aml',
            'Russia_with_Indonesia_policies',
            'no_aml_severity',
            'no_aml_flag',
            'no_license_intensity',
            'no_license_no_ban',
            'no_classified_as_commodity',
            'no_classified_as_DFA',
            'no_exchange_license_required',
            'no_classified_as_security',
            'no_payment_ban',
            'Russia_with_United_States_policies',
            'no_tax_rate',
            'Indonesia_with_Russia_policies',
            'Indonesia_with_United_States_policies'
        ],
        'Indonesia': [
            0.0,  # United States_with_Indonesia_policies
            6.2,  # max_regulations
            -20.6,  # no_regulations
            0.0,  # United States_with_Russia_policies
            -17.5,  # no_tax_no_aml
            0.0,  # Russia_with_Indonesia_policies
            1.2,  # no_aml_severity
            6.0,  # no_aml_flag
            3.3,  # no_license_intensity
            0.3,  # no_license_no_ban
            -0.3,  # no_classified_as_commodity
            0.0,  # no_classified_as_DFA
            -1.1,  # no_exchange_license_required
            0.0,  # no_classified_as_security
            0.0,  # no_payment_ban
            0.0,  # Russia_with_United_States_policies
            -23.1,  # no_tax_rate
            -20.9,  # Indonesia_with_Russia_policies
            -26.8,  # Indonesia_with_United_States_policies
        ],
        'Russia': [
            0.0,  # United States_with_Indonesia_policies
            24.3,  # max_regulations
            -3.6,  # no_regulations
            0.0,  # United States_with_Russia_policies
            6.9,  # no_tax_no_aml
            21.5,  # Russia_with_Indonesia_policies
            2.3,  # no_aml_severity
            4.8,  # no_aml_flag
            -1.3,  # no_license_intensity
            -9.7,  # no_license_no_ban
            -2.3,  # no_classified_as_commodity
            -0.9,  # no_classified_as_DFA
            -4.5,  # no_exchange_license_required
            0.0,  # no_classified_as_security
            -2.4,  # no_payment_ban
            -4.9,  # Russia_with_United_States_policies
            -0.4,  # no_tax_rate
            0.0,  # Indonesia_with_Russia_policies
            0.0,  # Indonesia_with_United_States_policies
        ],
        'United States': [
            425.6,  # United States_with_Indonesia_policies
            461.6,  # max_regulations
            169.9,  # no_regulations
            31.6,  # United States_with_Russia_policies
            94.5,  # no_tax_no_aml
            0.0,  # Russia_with_Indonesia_policies
            59.4,  # no_aml_severity
            39.9,  # no_aml_flag
            46.6,  # no_license_intensity
            46.8,  # no_license_no_ban
            0.6,  # no_classified_as_commodity
            0.0,  # no_classified_as_DFA
            -0.1,  # no_exchange_license_required
            -1.9,  # no_classified_as_security
            0.0,  # no_payment_ban
            0.0,  # Russia_with_United_States_policies
            -1.1,  # no_tax_rate
            0.0,  # Indonesia_with_Russia_policies
            0.0,  # Indonesia_with_United_States_policies
        ]
    }
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(data)
    
    # Melt the DataFrame for seaborn
    heatmap_melted = pd.melt(heatmap_df, 
                             id_vars=['scenario'], 
                             var_name='country', 
                             value_name='value')
    
    # Create the heatmap figure with publication-quality settings
    plt.figure(figsize=(14, 12))
    
    # Create a custom diverging colormap for positive/negative values
    # Use a better color scheme for academic publications
    colors_neg = plt.cm.YlOrBr(np.linspace(0.15, 0.8, 128))  # Yellows for close to zero, darker for more negative
    colors_pos = plt.cm.Greens(np.linspace(0.15, 0.9, 128))  # Light green to dark green
    
    # Combine them into a new colormap
    colors = np.vstack((colors_neg[::-1], colors_pos))
    custom_cmap = LinearSegmentedColormap.from_list('GreenGold_div', colors)
    
    # Create the pivot table for the heatmap
    pivot_table = heatmap_melted.pivot(index='scenario', columns='country', values='value')
    
    # For better presentation, replace 0.0 values that represent missing/NA with actual NaN
    pivot_table = pivot_table.replace(0.0, np.nan)
    
    # Define the axis labels with better names
    scenarios = pivot_table.index.tolist()
    readable_scenarios = []
    
    for scenario in scenarios:
        # Improve scenario names for readability
        readable = scenario.replace('_', ' ')
        readable_scenarios.append(readable)
    
    # Create the heatmap with annotations and improved styling
    ax = sns.heatmap(
        pivot_table,
        annot=True, 
        fmt='.1f',
        cmap=custom_cmap,
        center=0,  # Center the colormap at zero
        cbar_kws={'label': '% Change in GDP-Normalized Volume', 'shrink': 0.8},
        linewidths=1.0,
        linecolor='white',
        mask=pivot_table.isna()  # Only show annotations for non-NaN values
    )
    
    # Set better tick labels
    ax.set_yticklabels(readable_scenarios, fontsize=11)
    ax.set_xticklabels(pivot_table.columns, fontsize=12, fontweight='bold')
    
    # Improve the appearance
    plt.title('Cross-Country Policy Impact Comparison', fontsize=18, fontweight='bold', pad=20)
    
    # Remove xlabel as it's redundant
    plt.xlabel('')
    
    # Change 'scenario' to something more meaningful
    plt.ylabel('Policy Scenario', fontsize=14, fontweight='bold')
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
              "Values represent % change in GDP-normalized trading volume when policy is removed\n"
              "Positive values (green): Policy was restrictive | Negative values (gold): Policy was supportive",
              ha='center', fontsize=12, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the figure
    filepath = os.path.join(output_dir, 'cross_country_policy_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Enhanced Cross-Country Policy Impact Comparison to {filepath}")

if __name__ == "__main__":
    create_enhanced_cross_country_heatmap()
