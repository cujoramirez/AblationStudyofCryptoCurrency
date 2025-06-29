"""
Visualize Policy Ablation Findings

This script creates visualizations from the ablation findings to better understand
the impact of different policy instruments across countries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ensure_dir(directory):
    """Ensure the output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_ablation_data():
    """Load ablation findings from CSV"""
    try:
        # First try with the original column names from the CSV
        df = pd.read_csv('ablation_findings.csv')
        
        # Check if we have the expected columns
        if 'Policy Instrument' in df.columns:
            # Data is already in the right format
            pass
        else:
            # Try reshaping from the scenarios.csv directly
            scenarios = pd.read_csv('scenarios.csv')
            # Process the data (similar to calculate_ablation_impact.py)
            grouped = scenarios.groupby(['country', 'scenario'])['predicted_norm_volume'].mean().reset_index()
            baseline = grouped[grouped['scenario'] == 'baseline'].rename(
                columns={'predicted_norm_volume': 'baseline'}
            )[['country', 'baseline']]
            
            # Get non-baseline scenarios
            policy_data = grouped[grouped['scenario'] != 'baseline']
            
            # Merge with baseline
            merged = policy_data.merge(baseline, on='country')
            
            # Calculate percentage change
            merged['percent_change'] = ((merged['predicted_norm_volume'] - merged['baseline']) / 
                                       merged['baseline'] * 100)
            
            # Clean up scenario names to get policy instrument names
            merged['Policy Instrument'] = merged['scenario'].str.replace('no_', '').str.replace('_', ' ').str.title()
            
            # Select and rename columns
            df = merged[['country', 'Policy Instrument', 'percent_change']]
            
            # Pivot to get countries as columns
            df = df.pivot(index='Policy Instrument', columns='country', values='percent_change').reset_index()
        
        return df
    except Exception as e:
        print(f"Error loading ablation data: {e}")
        return None

def create_policy_heatmap(df, output_dir):
    """Create heatmap of policy impacts by country"""
    try:
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Reshape data for heatmap
        pivot_df = df.set_index('Policy Instrument')
        
        # Replace NaN with 0 for visualization purposes
        pivot_df = pivot_df.fillna(0)
        
        # Create a mask for NaN values to show them differently
        mask = pivot_df == 0
        
        # Create the heatmap
        ax = sns.heatmap(pivot_df, annot=True, cmap='RdBu_r', center=0, 
                         fmt='.2f', cbar_kws={'label': '% Change in Volume'},
                         mask=mask)
        
        # Customize the plot
        plt.title('Policy Ablation Impact by Country', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(output_dir, 'policy_impact_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy heatmap to {filepath}")
    except Exception as e:
        print(f"Error creating policy heatmap: {e}")

def create_policy_barplot(df, output_dir):
    """Create bar plot showing policy impacts by country"""
    try:
        # Melt the dataframe to get it in the right format for a grouped bar plot
        melted_df = pd.melt(df, id_vars=['Policy Instrument'], 
                           var_name='Country', value_name='Percent Change')
        
        # Filter out NaN values
        melted_df = melted_df.dropna()
        
        # Set up the figure
        plt.figure(figsize=(14, 10))
        
        # Create the bar plot
        ax = sns.barplot(x='Policy Instrument', y='Percent Change', hue='Country', 
                         data=melted_df, palette='viridis')
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize the plot
        plt.title('Policy Ablation Impact by Country and Policy', fontsize=16)
        plt.xlabel('Policy Instrument', fontsize=14)
        plt.ylabel('Percent Change in Volume', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Country', fontsize=12)
        
        # Add value labels on top of bars
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., 
                        height + (5 if height >= 0 else -10),
                        f'{height:.2f}%',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, rotation=0)
        
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(output_dir, 'policy_impact_barplot.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy bar plot to {filepath}")
    except Exception as e:
        print(f"Error creating policy bar plot: {e}")

def create_policy_category_plot(df, output_dir):
    """Create a plot showing policies grouped by category"""
    try:
        # Define policy categories
        categories = {
            'Classification': ['Security Classification', 'Commodity Classification', 
                              'Digital Financial Asset Class.'],
            'Licensing': ['Exchange Licensing Req.', 'License Stringency'],
            'Compliance': ['AML Requirements', 'AML Enforcement'],
            'Restrictions': ['Payment Ban'],
            'Financial': ['Taxation']
        }
        
        # Create a new dataframe with category information
        result = []
        for category, policies in categories.items():
            for policy in policies:
                # Find this policy in the dataframe
                policy_row = df[df['Policy Instrument'] == policy]
                if not policy_row.empty:
                    for country in df.columns[1:]:  # Skip the Policy Instrument column
                        if country in policy_row.columns:
                            value = policy_row[country].values[0]
                            if not pd.isna(value):
                                result.append({
                                    'Category': category,
                                    'Policy': policy,
                                    'Country': country,
                                    'Impact': value
                                })
        
        # Convert to DataFrame
        category_df = pd.DataFrame(result)
        
        if category_df.empty:
            print("No category data available for visualization")
            return
            
        # Calculate average impact by category and country
        avg_by_category = category_df.groupby(['Category', 'Country'])['Impact'].mean().reset_index()
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Category', y='Impact', hue='Country', data=avg_by_category, palette='Set2')
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize the plot
        plt.title('Average Policy Impact by Category and Country', fontsize=16)
        plt.xlabel('Policy Category', fontsize=14)
        plt.ylabel('Average % Change in Volume', fontsize=14)
        plt.legend(title='Country', fontsize=12)
        
        # Add value labels
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.5 if height >= 0 else -2),
                        f'{height:.2f}%',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, rotation=0)
        
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(output_dir, 'policy_category_impact.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved policy category plot to {filepath}")
    except Exception as e:
        print(f"Error creating policy category plot: {e}")

def visualize_ablation_findings():
    """Main function to create all visualizations"""
    print("Creating visualizations of ablation findings...")
    
    # Ensure output directory exists
    output_dir = 'ablation_visualizations'
    ensure_dir(output_dir)
    
    # Load the data
    df = load_ablation_data()
    
    if df is not None:
        # Create visualizations
        create_policy_heatmap(df, output_dir)
        create_policy_barplot(df, output_dir)
        create_policy_category_plot(df, output_dir)
        
        print(f"All visualizations saved to {output_dir} directory")
    else:
        print("Failed to create visualizations - data not available.")

if __name__ == "__main__":
    visualize_ablation_findings()
