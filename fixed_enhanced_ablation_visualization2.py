"""
Enhanced Ablation Study Visualization

This script creates advanced visualizations for the ablation study findings,
providing deeper insights into the policy impact across countries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

# Set the style for all visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

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

def create_radar_chart(df, output_dir):
    """Create radar charts to show policy profiles by country"""
    try:
        # Get country names from columns
        countries = df.columns[1:].tolist()
        
        # Get policy instruments
        policy_instruments = df['Policy Instrument'].tolist()
        
        # Create a figure with multiple subplots - adjusted size for better proportions
        fig, axes = plt.subplots(1, len(countries), figsize=(24, 9), 
                                subplot_kw=dict(projection='polar'))
        
        if len(countries) == 1:
            axes = [axes]  # Convert to list if only one country
            
        for i, country in enumerate(countries):
            ax = axes[i]
            
            # Get values for this country
            values = df[country].fillna(0).tolist()
            
            # Calculate angles for radar chart
            angles = np.linspace(0, 2*np.pi, len(policy_instruments), endpoint=False).tolist()
            
            # Close the radar chart
            values.append(values[0])
            angles.append(angles[0])
            policy_labels = policy_instruments + [policy_instruments[0]]
            
            # Plot the radar chart with thicker lines for better visibility
            ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=country, color='navy')
            ax.fill(angles, values, alpha=0.3, color='navy')
            
            # Set the labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([])  # Remove default labels
            
            # Create better label placement with proper positioning based on the angle
            for j, (angle, policy) in enumerate(zip(angles[:-1], policy_instruments)):
                # Calculate angle in degrees for better positioning calculation
                angle_deg = np.degrees(angle)
                
                # Determine horizontal alignment based on position in circle
                if 0 <= angle_deg <= 60 or 300 <= angle_deg <= 360:
                    ha = 'left'
                    va = 'center'
                    offset = 1.7  # Increase offset for better readability
                    rotation = angle_deg
                elif 60 < angle_deg < 120:
                    ha = 'center'
                    va = 'bottom'
                    offset = 1.7
                    rotation = angle_deg - 90  # Horizontal text
                elif 120 <= angle_deg <= 240:
                    ha = 'right'
                    va = 'center'
                    offset = 1.7
                    rotation = angle_deg - 180
                else:  # 240 < angle_deg < 300
                    ha = 'center'
                    va = 'top'
                    offset = 1.7
                    rotation = angle_deg - 270  # Horizontal text
                
                # Add the text label with background for better readability
                label_x = offset * np.cos(angle)
                label_y = offset * np.sin(angle)
                ax.text(label_x, label_y, policy, 
                      ha=ha, va=va,
                      size=9, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                      transform=ax.transData)
            
            # Set title
            ax.set_title(country, size=16, fontweight='bold', pad=20)
            
            # Add zero line
            ax.plot(angles, [0] * len(angles), color='black', linestyle='-', alpha=0.3, linewidth=1.5)
            
            # Add concentric circles and make the grid more visible
            ax.grid(True, color='gray', alpha=0.3, linewidth=0.5)
            
            # Set y-limits based on the data range
            all_values = df[countries].fillna(0).values.flatten()
            y_max = max(100, np.ceil(max(all_values) / 10) * 10)
            y_min = min(-100, np.floor(min(all_values) / 10) * 10)
            
            # Adjust y-ticks
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(np.linspace(y_min, y_max, 5))
            
            # Add value labels at certain points for better readability
            for val in values[:-1]:  # Skip the last duplicate value
                if abs(val) > 10:  # Only label significant values
                    idx = values.index(val)
                    angle = angles[idx]
                    # Position label inside the plotted line
                    label_radius = val * 0.85
                    ax.text(angle, label_radius, f'{val:.1f}%', 
                         ha='center', va='center', fontsize=8, 
                         bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Adjust layout and bring charts closer together
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4)  # Reduce spacing between subplots to bring charts closer
        
        # Add a common color bar with adjusted position to prevent overlap
        cbar_ax = fig.add_axes([1.03, 0.15, 0.02, 0.7])  # Move it further to the right
        
        # Create a more colorblind-friendly diverging colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors_pos = plt.cm.Reds(np.linspace(0.2, 1, 128))
        colors_neg = plt.cm.Blues_r(np.linspace(0.2, 1, 128))
        colors = np.vstack((colors_neg, colors_pos))
        colormap = LinearSegmentedColormap.from_list('RedBlue_colorblind', colors)
        
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(y_min, y_max), 
                                                cmap=colormap), 
                           cax=cbar_ax)
        cbar.set_label('Percent Change in Volume', fontsize=12, fontweight='bold', rotation=270, labelpad=25)
        
        # Add tick marks at specific values for better readability
        cbar.set_ticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
                  
        # Add a common title with more academic formatting
        fig.suptitle('Policy Ablation Impact Profiles by Country', 
                    fontsize=20, fontweight='bold', y=1.05)
                    
        # Add subtitle with study details
        plt.figtext(0.5, 0.965, 
                  "Cross-National Analysis of Cryptocurrency Policy Effects: A Counterfactual Policy Simulation Approach",
                  ha='center', fontsize=14, fontstyle='italic')
        
        # Add enhanced annotation explaining interpretation
        plt.figtext(0.5, 0.01, 
                  "INTERPRETATION GUIDE:\n"
                  "• POSITIVE values (red): Removing policy INCREASES volume → Policy was RESTRICTIVE\n"
                  "• NEGATIVE values (blue): Removing policy DECREASES volume → Policy was SUPPORTIVE",
                  ha='center', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='white', edgecolor='black', alpha=0.95, 
                           boxstyle='round,pad=1', linewidth=1.5))
                           
        # Add methodology note and citation placeholder
        plt.figtext(0.02, 0.02,
                  "Methodology: Counterfactual ablation simulation based on XGBoost model (n=1,825 observations, 2018-2025)\nData sources: Bitcoin network data, policy records from regulatory databases",
                  ha='left', fontsize=9, fontstyle='italic')
        
        # Save the figure with increased padding
        filepath = os.path.join(output_dir, 'policy_impact_radar.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy radar chart to {filepath}")
    except Exception as e:
        print(f"Error creating radar chart: {e}")

def create_policy_ranking(df, output_dir):
    """Create a visualization showing policy instruments ranked by impact"""
    try:
        # Create a melted dataframe for easier plotting
        melted_df = pd.melt(df, id_vars=['Policy Instrument'], 
                          var_name='Country', value_name='Percent_Change')
        
        # Drop rows with NaN values
        melted_df = melted_df.dropna()
        
        # Calculate absolute impact for ranking
        melted_df['Abs_Impact'] = abs(melted_df['Percent_Change'])
        
        # Calculate average absolute impact by policy
        policy_impact = melted_df.groupby('Policy Instrument')['Abs_Impact'].mean().reset_index()
        policy_impact = policy_impact.sort_values('Abs_Impact', ascending=False)
        
        # Create a ranked bar chart
        plt.figure(figsize=(12, 8))
        
        ax = sns.barplot(x='Abs_Impact', y='Policy Instrument', data=policy_impact, 
                       palette='viridis', order=policy_impact['Policy Instrument'])
        
        # Add value labels
        for i, v in enumerate(policy_impact['Abs_Impact']):
            ax.text(v + 0.5, i, f'{v:.2f}%', va='center')
        
        # Customize the plot
        plt.title('Ranking of Policy Instruments by Average Impact Magnitude', fontsize=16)
        plt.xlabel('Average Absolute Percent Change', fontsize=14)
        plt.ylabel('Policy Instrument', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(output_dir, 'policy_impact_ranking.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy ranking chart to {filepath}")
    except Exception as e:
        print(f"Error creating policy ranking visualization: {e}")

def create_country_response_comparison(df, output_dir):
    """Create a visualization comparing how different countries respond to the same policies"""
    try:
        # Melt the dataframe for easier manipulation
        melted_df = pd.melt(df, id_vars=['Policy Instrument'], 
                          var_name='Country', value_name='Percent_Change')
        
        # Drop rows with NaN values
        melted_df = melted_df.dropna()
        
        # Create a pivot table for policies that exist in multiple countries
        policy_counts = melted_df.groupby('Policy Instrument').size()
        multi_country_policies = policy_counts[policy_counts > 1].index.tolist()
        
        if not multi_country_policies:
            print("No policies shared across countries for comparison")
            return
            
        # Filter for multi-country policies
        comparison_df = melted_df[melted_df['Policy Instrument'].isin(multi_country_policies)]
        
        # Create a grouped bar chart
        plt.figure(figsize=(14, 8))
        
        # Sort policies by average impact
        policy_order = comparison_df.groupby('Policy Instrument')['Percent_Change'].mean().sort_values().index
        
        ax = sns.barplot(x='Policy Instrument', y='Percent_Change', hue='Country', 
                       data=comparison_df, order=policy_order)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize the plot
        plt.title('Country-Specific Responses to Policy Instruments', fontsize=16)
        plt.xlabel('Policy Instrument', fontsize=14)
        plt.ylabel('Percent Change in Volume', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Country', fontsize=12)
        
        # Add value labels
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., 
                      height + (1 if height >= 0 else -3),
                      f'{height:.1f}%',
                      ha='center', va='bottom' if height >= 0 else 'top',
                      fontsize=9, rotation=0)
        
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(output_dir, 'country_response_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved country response comparison to {filepath}")
    except Exception as e:
        print(f"Error creating country response comparison: {e}")

def create_policy_pair_correlation(df, output_dir):
    """Create a visualization showing which policy pairs have similar impacts"""
    try:
        # Prepare data - pivot to get policies as columns for correlation
        pivot_df = df.set_index('Policy Instrument').T
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr(method='pearson')
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        ax = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        annot=True, fmt='.2f', square=True, linewidths=.5)
        
        plt.title('Correlation Between Policy Instrument Impacts', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(output_dir, 'policy_correlation_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy correlation matrix to {filepath}")
        
        # Create hierarchical clustering of policies
        plt.figure(figsize=(14, 8))
        
        # Use clustermap to show hierarchical relationships
        cluster = sns.clustermap(corr_matrix, cmap=cmap, vmax=1, vmin=-1, center=0,
                              annot=True, fmt='.2f', linewidths=.5,
                              figsize=(14, 12))
        
        plt.title('Hierarchical Clustering of Policy Instruments by Impact Similarity', fontsize=16)
        
        # Save the figure
        filepath = os.path.join(output_dir, 'policy_hierarchical_clustering.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy hierarchical clustering to {filepath}")
    except Exception as e:
        print(f"Error creating policy correlation analysis: {e}")

def create_summary_dashboard(df, output_dir):
    """Create a comprehensive summary dashboard visualization"""
    try:
        # Convert dataframe to numeric where possible
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Melt the dataframe for easier plotting
        melted_df = pd.melt(df, id_vars=['Policy Instrument'], 
                          var_name='Country', value_name='Percent_Change')
        melted_df = melted_df.dropna()
        
        # Create figure with GridSpec for complex layout
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Top impact policies by country - top left
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Find top 3 impacts (absolute) for each country
        top_impacts = []
        for country in melted_df['Country'].unique():
            country_df = melted_df[melted_df['Country'] == country]
            country_df = country_df.sort_values('Percent_Change', key=abs, ascending=False)
            top_impacts.append(country_df.head(3))
        
        top_impacts_df = pd.concat(top_impacts)
        
        sns.barplot(x='Percent_Change', y='Policy Instrument', hue='Country', 
                   data=top_impacts_df, ax=ax1)
        
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Top Policy Impacts by Country')
        ax1.set_xlabel('Percent Change')
        ax1.grid(axis='x', linestyle='--', alpha=0.5)
        
        # 2. Average impact by country - top center
        ax2 = fig.add_subplot(gs[0, 1])
        
        country_avg = melted_df.groupby('Country')['Percent_Change'].agg(['mean', 'std']).reset_index()
        
        sns.barplot(x='Country', y='mean', data=country_avg, ax=ax2)
        
        # Add error bars
        for i, row in country_avg.iterrows():
            ax2.errorbar(i, row['mean'], yerr=row['std'], fmt='none', color='black', capsize=5)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Average Policy Impact by Country')
        ax2.set_ylabel('Average Percent Change')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 3. Distribution of impacts - top right
        ax3 = fig.add_subplot(gs[0, 2])
        
        for country in melted_df['Country'].unique():
            sns.kdeplot(data=melted_df[melted_df['Country'] == country], 
                      x='Percent_Change', ax=ax3, label=country, fill=True, alpha=0.3)
        
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Distribution of Policy Impacts')
        ax3.set_xlabel('Percent Change')
        ax3.grid(axis='x', linestyle='--', alpha=0.5)
        
        # 4. Heatmap - bottom left and center
        ax4 = fig.add_subplot(gs[1:, :2])
        
        pivot_df = melted_df.pivot(index='Policy Instrument', columns='Country', values='Percent_Change')
        
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(pivot_df, cmap=cmap, center=0, annot=True, fmt='.1f',
                   linewidths=0.5, cbar_kws={'label': 'Percent Change'}, ax=ax4)
        
        ax4.set_title('Policy Impact Heatmap')
        
        # 5. Policy type impact - bottom right
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Define policy categories
        categories = {
            'Classification': ['Security Classification', 'Commodity Classification', 
                              'Digital Financial Asset Class.'],
            'Licensing': ['Exchange Licensing Req.', 'License Stringency'],
            'Compliance': ['AML Requirements', 'AML Enforcement'],
            'Restrictions': ['Payment Ban'],
            'Financial': ['Taxation']
        }
        
        # Add category to data
        melted_df['Category'] = 'Other'
        for category, policies in categories.items():
            mask = melted_df['Policy Instrument'].isin(policies)
            melted_df.loc[mask, 'Category'] = category
        
        category_impacts = melted_df.groupby(['Category', 'Country'])['Percent_Change'].mean().reset_index()
        
        sns.barplot(x='Category', y='Percent_Change', hue='Country', data=category_impacts, ax=ax5)
        
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_title('Average Impact by Policy Category')
        ax5.set_xlabel('Category')
        ax5.set_ylabel('Average Percent Change')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Correlation between countries - bottom right
        ax6 = fig.add_subplot(gs[2, 2])
        
        # Calculate correlation
        corr_matrix = pivot_df.T.corr()
        
        # Create correlation heatmap
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f',
                   linewidths=0.5, ax=ax6)
        
        ax6.set_title('Correlation Between Countries')
        
        # Add interpretation text
        plt.figtext(0.5, 0.02, 
                  "Positive values indicate INCREASES in volume when policy is removed (policy was restrictive)\n"
                  "Negative values indicate DECREASES in volume when policy is removed (policy was supportive)",
                  ha='center', fontsize=12, 
                  bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
        # Add a summary title
        fig.suptitle('Comprehensive Bitcoin Policy Ablation Study Results', fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        filepath = os.path.join(output_dir, 'ablation_dashboard.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary dashboard to {filepath}")
    except Exception as e:
        print(f"Error creating summary dashboard: {e}")

def create_enhanced_ablation_visualizations():
    """Main function to create all enhanced visualizations"""
    print("Creating enhanced ablation visualizations...")
    
    # Ensure output directory exists
    output_dir = 'enhanced_ablation_visualizations'
    ensure_dir(output_dir)
    
    # Load the data
    df = load_ablation_data()
    
    if df is not None:
        # Create visualizations
        create_radar_chart(df, output_dir)
        create_policy_ranking(df, output_dir)
        create_country_response_comparison(df, output_dir)
        create_policy_pair_correlation(df, output_dir)
        create_summary_dashboard(df, output_dir)
        
        print(f"All enhanced visualizations saved to {output_dir} directory")
    else:
        print("Failed to create visualizations - data not available.")

if __name__ == "__main__":
    create_enhanced_ablation_visualizations()
