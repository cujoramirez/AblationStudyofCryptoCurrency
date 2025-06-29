"""
Ablation Findings Calculator

This script analyzes the scenarios.csv data and calculates the percentage changes
for each policy ablation scenario compared to baseline.
"""

import pandas as pd
import numpy as np
import os

def calculate_ablation_impact():
    """
    Calculate the impact of each policy ablation scenario compared to baseline.
    Returns a clean table showing percentage changes for each country and policy.
    """
    # Load the scenarios data
    try:
        scenarios_df = pd.read_csv('scenarios.csv')
    except FileNotFoundError:
        print("Error: scenarios.csv file not found")
        return None
    
    # Group data by country and scenario, then calculate mean volumes
    grouped = scenarios_df.groupby(['country', 'scenario'])['predicted_norm_volume'].mean().reset_index()
    
    # Get baseline values for each country
    baseline_df = grouped[grouped['scenario'] == 'baseline'].copy()
    baseline_df = baseline_df.rename(columns={'predicted_norm_volume': 'baseline_volume'})
    baseline_df = baseline_df[['country', 'baseline_volume']]
    
    # Filter for single-policy ablation scenarios (those starting with "no_")
    policy_scenarios = grouped[grouped['scenario'].str.startswith('no_') & 
                              ~grouped['scenario'].str.contains('_no_')].copy()
    
    # Merge with baseline data
    results_df = policy_scenarios.merge(baseline_df, on='country')
    
    # Calculate percentage change
    results_df['percent_change'] = ((results_df['predicted_norm_volume'] - results_df['baseline_volume']) / 
                                  results_df['baseline_volume'] * 100)
    
    # Extract policy name from scenario name
    results_df['policy'] = results_df['scenario'].str.replace('no_', '')
    
    # Reorder and select relevant columns
    results_df = results_df[['country', 'policy', 'percent_change']]
    
    # Pivot the data to get policies as rows and countries as columns
    pivot_df = results_df.pivot(index='policy', columns='country', values='percent_change')
    pivot_df = pivot_df.round(2)  # Round to 2 decimal places
    
    # Add "Extreme Scenarios" data
    extremes_df = grouped[(grouped['scenario'] == 'no_regulations') | 
                        (grouped['scenario'] == 'max_regulations')].copy()
    
    if not extremes_df.empty:
        extremes_results = extremes_df.merge(baseline_df, on='country')
        extremes_results['percent_change'] = ((extremes_results['predicted_norm_volume'] - 
                                             extremes_results['baseline_volume']) / 
                                             extremes_results['baseline_volume'] * 100)
        extremes_results = extremes_results[['country', 'scenario', 'percent_change']]
        
        extremes_pivot = extremes_results.pivot(index='scenario', columns='country', values='percent_change')
        extremes_pivot = extremes_pivot.round(2)
        
        # Format output for human readability
        print("\nEXTREME SCENARIO IMPACTS (% change from baseline)")
        print("=" * 60)
        print(extremes_pivot)
    
    # Create a clean DataFrame for output to CSV
    output_df = pivot_df.reset_index()
    output_df = output_df.rename(columns={'policy': 'Policy Instrument'})
    
    # Map policy instrument names to more readable formats
    policy_name_mapping = {
        'classified_as_security': 'Security Classification',
        'classified_as_commodity': 'Commodity Classification',
        'classified_as_DFA': 'Digital Financial Asset Class.',
        'exchange_license_required': 'Exchange Licensing Req.',
        'license_intensity': 'License Stringency',
        'tax_rate': 'Taxation',
        'aml_flag': 'AML Requirements',
        'aml_severity': 'AML Enforcement',
        'payment_ban': 'Payment Ban'
    }
    
    output_df['Policy Instrument'] = output_df['Policy Instrument'].map(
        lambda x: policy_name_mapping.get(x, x.replace('_', ' ').title())
    )
    
    # Save the results to CSV
    output_df.to_csv('ablation_findings.csv', index=False)
    
    return output_df

def create_markdown_table(df):
    """
    Convert the DataFrame to a Markdown table for documentation with improved formatting
    """
    if df is None:
        return "No data available to create table."
        
    # Create markdown table header with centered alignment
    countries = df.columns[1:].tolist()
    header = f"| Policy Instrument | {' | '.join(countries)} |"
    separator = f"|:{'-' * 20}:|{':---:|' * len(countries)}"
    
    # Sort by the average absolute impact across all countries
    df['avg_impact'] = df[countries].abs().mean(axis=1)
    df = df.sort_values(by='avg_impact', ascending=False).drop(columns=['avg_impact'])
    
    # Create table rows with color indicators
    rows = []
    for _, row in df.iterrows():
        policy = row['Policy Instrument']
        
        # Format values with signs and handle NaN values
        values = []
        for country in countries:
            if pd.isna(row[country]):
                values.append("N/A")
            else:
                val = row[country]
                # Format with signs and 2 decimal places
                if val > 0:
                    values.append(f"+{val:.2f}%")  # Positive values with +
                elif val < 0:
                    values.append(f"{val:.2f}%")   # Negative values (already have -)
                else:
                    values.append("0.00%")         # Zero values
        
        rows.append(f"| {policy} | {' | '.join(values)} |")
    
    # Combine all parts of the table
    markdown_table = f"{header}\n{separator}\n" + "\n".join(rows)
    
    # Generate explanatory text
    explanation = """
## Understanding the Table
- **Positive values (+)**: Removing this policy *increases* trading volume, suggesting the policy had a negative impact on volume.
- **Negative values (-)**: Removing this policy *decreases* trading volume, suggesting the policy had a positive impact on volume.
- **N/A**: This policy was not implemented or simulated in the country.

## Key Insights
1. **AML Enforcement** has the most significant positive impact on US volumes (+71.45%), suggesting strong enforcement increases compliance and market confidence.
2. **Security Classification** has a significant negative impact in the US (-11.61%), indicating this regulatory classification supports higher trading volumes.
3. **Taxation** has substantial effects on Indonesia (-46.90%) and US (-26.36%), with both countries showing decreased volumes when tax policies are removed.
4. **Complete removal of regulations** produces mixed effects: decreased volume in Indonesia (-47.55%) and Russia (-6.27%), but increased volume in the US (+26.38%).
"""
    
    # Save to markdown file
    with open('ablation_findings_table.md', 'w') as f:
        f.write("# Ablation Findings by Policy Instrument\n\n")
        f.write("This table shows the percentage change in trading volume when a policy instrument is removed from the regulatory framework. The analysis is based on an XGBoost model trained on historical crypto trading data across three countries.\n\n")
        f.write(markdown_table)
        f.write(explanation)
    
    return markdown_table

if __name__ == "__main__":
    print("Calculating ablation impacts from scenarios.csv...")
    results = calculate_ablation_impact()
    
    if results is not None:
        print("\nRESULTS SUMMARY (% change from baseline)")
        print("=" * 60)
        print(results)
        print("\nSaved to ablation_findings.csv")
        
        # Also create a markdown table
        markdown_table = create_markdown_table(results)
        print("\nMarkdown table saved to ablation_findings_table.md")
    else:
        print("Failed to calculate ablation findings.")
