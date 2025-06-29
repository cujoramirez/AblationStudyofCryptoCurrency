"""
Enhanced Bitcoin Policy Analysis Visualizations
Provides improved visualizations for the Bitcoin policy study
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_enhanced_policy_visualizations(scenarios_file='scenarios.csv', output_dir='enhanced_visualizations'):
    """
    Create enhanced policy visualizations from the scenarios data
    
    Args:
        scenarios_file: Path to the scenarios CSV file
        output_dir: Directory to save visualizations
    """
    print("Creating enhanced policy visualizations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load scenarios data
    try:
        scenarios_df = pd.read_csv(scenarios_file)
    except FileNotFoundError:
        print(f"Error: Could not find scenarios file {scenarios_file}")
        return
    
    # Group scenarios into categories for better visualization
    scenario_categories = {
        'baseline': ['baseline'],
        'single_policy_ablation': [s for s in scenarios_df['scenario'].unique() if s.startswith('no_') and '_no_' not in s],
        'multi_policy_ablation': [s for s in scenarios_df['scenario'].unique() if '_no_' in s],
        'cross_country': [s for s in scenarios_df['scenario'].unique() if 'with_' in s and 'policies' in s],
        'adoption': ['high_adoption', 'low_adoption'],
        'extreme': ['no_regulations', 'max_regulations']
    }
    
    # Calculate impact of each scenario compared to baseline
    impact_data = []
    
    for country in scenarios_df['country'].unique():
        country_data = scenarios_df[scenarios_df['country'] == country]
        
        # Get baseline
        baseline = country_data[country_data['scenario'] == 'baseline']
        baseline_avg = baseline['predicted_norm_volume'].mean()
        baseline_end = baseline.iloc[-1]['predicted_norm_volume'] if len(baseline) > 0 else 0
        
        # Compare each scenario to baseline
        for scenario in country_data['scenario'].unique():
            if scenario != 'baseline':
                scenario_data = country_data[country_data['scenario'] == scenario]
                scenario_avg = scenario_data['predicted_norm_volume'].mean()
                scenario_end = scenario_data.iloc[-1]['predicted_norm_volume'] if len(scenario_data) > 0 else 0
                
                avg_pct_change = (scenario_avg - baseline_avg) / baseline_avg * 100
                end_pct_change = (scenario_end - baseline_end) / baseline_end * 100
                
                # Determine category for grouping in visualizations
                category = 'other'
                for cat_name, cat_scenarios in scenario_categories.items():
                    if scenario in cat_scenarios:
                        category = cat_name
                        break
                
                impact_data.append({
                    'country': country,
                    'scenario': scenario,
                    'category': category,
                    'avg_pct_change': avg_pct_change,
                    'end_pct_change': end_pct_change
                })
    
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        
        # 1. Enhanced Cross-Country Policy Impact Heatmap
        pivot_data = impact_df.pivot_table(
            index='scenario', 
            columns='country', 
            values='avg_pct_change',
            aggfunc='mean'
        )
        
        # Sort scenarios by their average impact across countries
        pivot_data['avg_impact'] = pivot_data.mean(axis=1)
        pivot_data = pivot_data.sort_values('avg_impact', ascending=False)
        pivot_data = pivot_data.drop(columns=['avg_impact'])
        
        # Create enhanced heatmap with better formatting
        plt.figure(figsize=(14, max(10, len(pivot_data) * 0.5)))
        
        # Create heatmap with enhanced visual appearance
        ax = sns.heatmap(pivot_data, annot=True, cmap='RdYlGn_r', center=0, fmt='.1f',
                  cbar_kws={'label': '% Change in Volume', 'shrink': 0.8},
                  linewidths=0.5, linecolor='lightgray')
        
        # Improve title and labels
        plt.title('Cross-Country Policy Impact Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Country', fontsize=12, fontweight='bold')
        plt.ylabel('Policy Scenario', fontsize=12, fontweight='bold')
        
        # Add grid for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        
        # Add a border around the heatmap
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        
        # Add annotations to help interpretation
        plt.figtext(0.02, 0.02, 
                   "Red = Negative impact on volume\nGreen = Positive impact on volume", 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/enhanced_policy_impact_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Category-grouped heatmap
        # Group scenarios by category first
        impact_categories = impact_df.copy()
        
        # Create a hierarchical index for better visualization
        cat_pivot = impact_categories.pivot_table(
            index=['category', 'scenario'],
            columns='country', 
            values='avg_pct_change',
            aggfunc='mean'
        )
        
        if len(cat_pivot) > 0:
            plt.figure(figsize=(14, max(12, len(cat_pivot) * 0.4)))
            
            # Create heatmap with category divisions
            ax = sns.heatmap(cat_pivot, annot=True, cmap='RdYlGn_r', center=0, fmt='.1f',
                      cbar_kws={'label': '% Change in Volume', 'shrink': 0.8},
                      linewidths=0.5, linecolor='lightgray')
            
            plt.title('Policy Impact by Category and Country', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Country', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/policy_impact_heatmap_by_category.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Enhanced boxplot showing distribution of impacts by category
        plt.figure(figsize=(16, 10))
        
        # Create a more informative boxplot
        ax = sns.boxplot(x='category', y='avg_pct_change', data=impact_df, 
                   palette='Set3', width=0.6, showfliers=False)
        
        # Add individual points for better data visualization
        sns.stripplot(x='category', y='avg_pct_change', data=impact_df, 
                     color='black', alpha=0.5, jitter=True, size=4)
        
        # Add median labels
        medians = impact_df.groupby('category')['avg_pct_change'].median().round(1)
        for i, median in enumerate(medians):
            ax.text(i, median, f'Median: {median}%', 
                   horizontalalignment='center', fontweight='bold',
                   color='darkblue', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlabel('Policy Category', fontsize=14, fontweight='bold')
        plt.ylabel('Percent Change in Volume', fontsize=14, fontweight='bold')
        plt.title('Distribution of Policy Impacts by Category', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=30, ha='right', fontsize=12)
        plt.yticks(fontsize=11)
        
        # Add annotations
        categories = impact_df['category'].unique()
        counts = impact_df.groupby('category').size()
        for i, category in enumerate(categories):
            if category in counts:
                plt.text(i, impact_df['avg_pct_change'].min() - 5, 
                       f'n={counts[category]}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/enhanced_impact_distribution_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Policy impact correlation matrix
        # Only for scenarios that appear across all countries
        if len(pivot_data.columns) > 1:  # More than one country
            plt.figure(figsize=(10, 8))
            corr = pivot_data.T.corr()  # Transpose to get correlations between countries
            mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                       mask=mask, square=True, linewidths=0.5)
            plt.title('Policy Impact Correlation Between Countries', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/policy_impact_country_correlation.png', dpi=300)
            plt.close()
            
        # 5. Combined impact visualization by policy type
        # Group similar policies
        policy_types = {
            'Classification': [s for s in impact_df['scenario'].unique() if 'classified' in s],
            'Licensing': [s for s in impact_df['scenario'].unique() if 'license' in s],
            'Taxation': [s for s in impact_df['scenario'].unique() if 'tax' in s],
            'AML/Compliance': [s for s in impact_df['scenario'].unique() if 'aml' in s],
            'Restrictions': [s for s in impact_df['scenario'].unique() if 'ban' in s or 'restrict' in s],
        }
        
        # Add policy type column
        impact_df['policy_type'] = 'Other'
        for policy_type, scenarios in policy_types.items():
            mask = impact_df['scenario'].isin(scenarios)
            impact_df.loc[mask, 'policy_type'] = policy_type
            
        # Create policy type impact plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='policy_type', y='avg_pct_change', hue='country', data=impact_df)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Impact by Policy Type and Country', fontsize=14, fontweight='bold')
        plt.xlabel('Policy Type', fontsize=12)
        plt.ylabel('Average % Change in Volume', fontsize=12)
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/policy_type_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced policy visualizations created in {output_dir}/ directory.")
    else:
        print("No impact data available for visualization.")

if __name__ == "__main__":
    create_enhanced_policy_visualizations()
