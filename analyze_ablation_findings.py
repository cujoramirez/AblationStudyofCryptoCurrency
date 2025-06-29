"""
Comprehensive Analysis of Bitcoin Policy Ablation Study

This script analyzes the findings from the ablation study, generating insights and recommendations
based on the policy impacts across different countries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import markdown
from tabulate import tabulate

def load_ablation_data():
    """Load ablation findings from CSV"""
    try:
        df = pd.read_csv('ablation_findings.csv')
        return df
    except Exception as e:
        print(f"Error loading ablation data: {e}")
        return None

def ensure_dir(directory):
    """Ensure the output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_key_statistics(df):
    """Calculate key statistics from the ablation data"""
    stats = {}
    
    # Convert dataframe to numeric where possible
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get countries
    countries = df.columns[1:].tolist()
    
    # Calculate statistics for each country
    for country in countries:
        country_data = df[country].dropna()
        stats[country] = {
            'mean': country_data.mean(),
            'median': country_data.median(),
            'std': country_data.std(),
            'min': country_data.min(),
            'max': country_data.max(),
            'most_restrictive': df.loc[df[country].idxmax(), 'Policy Instrument'] if not country_data.empty and not pd.isna(country_data.idxmax()) else 'N/A',
            'most_supportive': df.loc[df[country].idxmin(), 'Policy Instrument'] if not country_data.empty and not pd.isna(country_data.idxmin()) else 'N/A',
        }
    
    # Calculate overall statistics
    all_values = pd.Series([value for country in countries for value in df[country].dropna()])
    stats['Overall'] = {
        'mean': all_values.mean(),
        'median': all_values.median(),
        'std': all_values.std(),
        'min': all_values.min(),
        'max': all_values.max(),
    }
    
    return stats

def analyze_policy_types(df):
    """Analyze impacts by policy type"""
    
    # Define policy categories
    categories = {
        'Classification': ['Security Classification', 'Commodity Classification', 
                         'Digital Financial Asset Class.'],
        'Licensing': ['Exchange Licensing Req.', 'License Stringency'],
        'Compliance': ['AML Requirements', 'AML Enforcement'],
        'Restrictions': ['Payment Ban'],
        'Financial': ['Taxation'],
        'Overall': ['Regulations']
    }
    
    # Calculate average impact by category and country
    category_impacts = {}
    
    for category, policies in categories.items():
        category_df = df[df['Policy Instrument'].isin(policies)]
        
        for country in df.columns[1:]:
            if category not in category_impacts:
                category_impacts[category] = {}
            
            country_data = category_df[country].dropna()
            if not country_data.empty:
                category_impacts[category][country] = country_data.mean()
            else:
                category_impacts[category][country] = None
    
    return category_impacts

def generate_insights(df, stats, category_impacts):
    """Generate insights from the analysis"""
    insights = []
    
    # Get countries
    countries = df.columns[1:].tolist()
    
    # Insight 1: Overall impact of regulations
    insights.append({
        'title': 'Overall Impact of Regulations',
        'description': 'Effects of complete removal of regulations on trading volume',
        'findings': {}
    })
    
    for country in countries:
        regulations_row = df[df['Policy Instrument'] == 'Regulations']
        if not regulations_row.empty and country in regulations_row.columns:
            impact = regulations_row[country].iloc[0]
            if pd.notna(impact):
                direction = 'increase' if impact > 0 else 'decrease'
                insights[0]['findings'][country] = f"Removing all regulations would {direction} volume by {abs(impact):.2f}%"
    
    # Insight 2: Most impactful policies
    insights.append({
        'title': 'Most Impactful Policies',
        'description': 'Policies with the highest absolute impact on trading volume',
        'findings': {}
    })
    
    for country in countries:
        country_data = df[country].dropna()
        if not country_data.empty:
            highest_impact_idx = country_data.abs().idxmax()
            policy = df.loc[highest_impact_idx, 'Policy Instrument']
            impact = country_data.loc[highest_impact_idx]
            insights[1]['findings'][country] = f"{policy} ({impact:.2f}%)"
    
    # Insight 3: Policy category impacts
    insights.append({
        'title': 'Policy Category Analysis',
        'description': 'Impact of different policy categories on trading volume',
        'findings': {}
    })
    
    for country in countries:
        country_insights = []
        for category, impacts in category_impacts.items():
            if country in impacts and impacts[country] is not None:
                direction = 'restrictive' if impacts[country] > 0 else 'supportive'
                country_insights.append(f"{category} policies are {direction} ({impacts[country]:.2f}%)")
        
        insights[2]['findings'][country] = country_insights
    
    # Insight 4: Cross-country comparison
    insights.append({
        'title': 'Cross-Country Policy Response',
        'description': 'How countries differ in their response to similar policies',
        'findings': {}
    })
    
    # Find policies that exist in multiple countries for comparison
    melted_df = pd.melt(df, id_vars=['Policy Instrument'], 
                       var_name='Country', value_name='Impact').dropna()
    policy_counts = melted_df.groupby('Policy Instrument').size()
    shared_policies = policy_counts[policy_counts > 1].index.tolist()
    
    for policy in shared_policies:
        policy_row = df[df['Policy Instrument'] == policy]
        policy_impacts = {country: policy_row[country].iloc[0] 
                         for country in countries 
                         if country in policy_row.columns and not pd.isna(policy_row[country].iloc[0])}
        
        if len(policy_impacts) > 1:
            impact_text = ', '.join([f"{country}: {impact:.2f}%" for country, impact in policy_impacts.items()])
            insights[3]['findings'][policy] = impact_text
    
    return insights

def create_summary_report(df, stats, category_impacts, insights):
    """Create a comprehensive summary report"""
    try:
        # Create output directory
        output_dir = 'analysis_output'
        ensure_dir(output_dir)
        
        # Start building the report in Markdown
        report_md = []
        
        # Title and introduction
        report_md.append("# Bitcoin Policy Ablation Study: Comprehensive Analysis\n")
        report_md.append("## Executive Summary\n")
        report_md.append("This report analyzes the impact of different cryptocurrency policy instruments on ")
        report_md.append("trading volumes across the United States, Russia, and Indonesia. The analysis is based ")
        report_md.append("on an ablation study, where each policy instrument was removed to observe the resulting ")
        report_md.append("percentage change in trading volume compared to the baseline scenario.\n")
        
        # Add interpretation note
        report_md.append("> **Note on interpretation:** Positive values indicate that removing the policy ")
        report_md.append("> instrument would increase trading volume (i.e., the policy was restrictive). ")
        report_md.append("> Negative values indicate that removing the policy would decrease volume ")
        report_md.append("> (i.e., the policy was supportive to trading volume).\n")
        
        # Add the ablation table
        report_md.append("## Ablation Study Results\n")
        report_md.append("The following table shows the percentage change in trading volume when each policy ")
        report_md.append("instrument is removed from the regulatory framework:\n")
        
        # Convert dataframe to markdown table
        table_data = tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt='.2f')
        report_md.append(table_data + "\n")
        
        # Key findings
        report_md.append("## Key Findings\n")
        
        # Add insights to the report
        for insight in insights:
            report_md.append(f"### {insight['title']}\n")
            report_md.append(f"{insight['description']}\n")
            
            for item, finding in insight['findings'].items():
                if isinstance(finding, list):
                    report_md.append(f"**{item}:**\n")
                    for point in finding:
                        report_md.append(f"- {point}\n")
                else:
                    report_md.append(f"**{item}:** {finding}\n")
            
            report_md.append("\n")
        
        # Statistical summary
        report_md.append("## Statistical Summary\n")
        report_md.append("### Summary Statistics by Country\n")
        
        # Create a table of statistics
        stats_table = []
        headers = ['Country', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Most Restrictive', 'Most Supportive']
        
        for country, country_stats in stats.items():
            if country != 'Overall':
                stats_table.append([
                    country, 
                    f"{country_stats['mean']:.2f}%", 
                    f"{country_stats['median']:.2f}%",
                    f"{country_stats['std']:.2f}%",
                    f"{country_stats['min']:.2f}%",
                    f"{country_stats['max']:.2f}%",
                    country_stats['most_restrictive'],
                    country_stats['most_supportive']
                ])
        
        stats_md = tabulate(stats_table, headers=headers, tablefmt='pipe')
        report_md.append(stats_md + "\n")
        
        # Policy category analysis
        report_md.append("### Policy Category Impact Analysis\n")
        report_md.append("Average percentage change in volume when policies in each category are removed:\n")
        
        # Create a category impact table
        category_table = []
        cat_headers = ['Category'] + list(next(iter(category_impacts.values())).keys())
        
        for category, impacts in category_impacts.items():
            row = [category]
            for country in cat_headers[1:]:
                if country in impacts and impacts[country] is not None:
                    row.append(f"{impacts[country]:.2f}%")
                else:
                    row.append("N/A")
            category_table.append(row)
        
        cat_md = tabulate(category_table, headers=cat_headers, tablefmt='pipe')
        report_md.append(cat_md + "\n")
        
        # Policy recommendations
        report_md.append("## Policy Recommendations\n")
        
        # Generate recommendations based on findings
        recommendations = []
        
        # Find the most supportive policies (negative values)
        supportive_policies = {}
        for country in df.columns[1:]:
            country_policies = df[['Policy Instrument', country]].dropna()
            supportive = country_policies[country_policies[country] < 0].sort_values(country)
            if not supportive.empty:
                supportive_policies[country] = supportive.iloc[0:min(3, len(supportive))]
        
        if supportive_policies:
            report_md.append("### Policies to Maintain or Strengthen\n")
            report_md.append("The following policies appear to support cryptocurrency trading volume and could be maintained or strengthened:\n")
            
            for country, policies in supportive_policies.items():
                report_md.append(f"**{country}:**\n")
                for _, row in policies.iterrows():
                    policy = row['Policy Instrument']
                    impact = row[country]
                    report_md.append(f"- {policy}: Removing this policy would reduce volume by {abs(impact):.2f}%\n")
        
        # Find the most restrictive policies (positive values)
        restrictive_policies = {}
        for country in df.columns[1:]:
            country_policies = df[['Policy Instrument', country]].dropna()
            restrictive = country_policies[country_policies[country] > 0].sort_values(country, ascending=False)
            if not restrictive.empty:
                restrictive_policies[country] = restrictive.iloc[0:min(3, len(restrictive))]
        
        if restrictive_policies:
            report_md.append("### Policies to Reconsider or Refine\n")
            report_md.append("The following policies appear to significantly restrict cryptocurrency trading volume and might be candidates for refinement:\n")
            
            for country, policies in restrictive_policies.items():
                report_md.append(f"**{country}:**\n")
                for _, row in policies.iterrows():
                    policy = row['Policy Instrument']
                    impact = row[country]
                    report_md.append(f"- {policy}: Removing this policy would increase volume by {impact:.2f}%\n")
        
        # Conclusion
        report_md.append("## Conclusion\n")
        report_md.append("This ablation study reveals significant differences in how cryptocurrency policy instruments ")
        report_md.append("affect trading volumes across different countries. Some key observations include:\n")
        
        # Generate overall conclusions
        us_data = df['United States'].dropna()
        russia_data = df['Russia'].dropna() 
        indonesia_data = df['Indonesia'].dropna()
        
        if not us_data.empty:
            avg_us = us_data.mean()
            direction_us = "restrictive" if avg_us > 0 else "supportive"
            report_md.append(f"- United States policies are overall {direction_us} to trading volume (avg: {avg_us:.2f}%)\n")
        
        if not russia_data.empty:
            avg_russia = russia_data.mean()
            direction_russia = "restrictive" if avg_russia > 0 else "supportive"
            report_md.append(f"- Russian policies are overall {direction_russia} to trading volume (avg: {avg_russia:.2f}%)\n")
        
        if not indonesia_data.empty:
            avg_indonesia = indonesia_data.mean()
            direction_indonesia = "restrictive" if avg_indonesia > 0 else "supportive"
            report_md.append(f"- Indonesian policies are overall {direction_indonesia} to trading volume (avg: {avg_indonesia:.2f}%)\n")
        
        report_md.append("\nPolicy makers should consider these findings when crafting cryptocurrency regulations, ")
        report_md.append("particularly noting the different impacts of similar policies across jurisdictions, which suggests ")
        report_md.append("that local market conditions and complementary policies play important roles in determining outcomes.\n")
        
        # Join all lines and write to file
        report_text = '\n'.join(report_md)
        
        with open(os.path.join(output_dir, 'comprehensive_analysis.md'), 'w') as f:
            f.write(report_text)
        
        # Convert to HTML if markdown is available
        try:
            html = markdown.markdown(report_text, extensions=['tables'])
            with open(os.path.join(output_dir, 'comprehensive_analysis.html'), 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <title>Bitcoin Policy Ablation Study Analysis</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        h1, h2, h3 {{ color: #333; }}
                        blockquote {{ background-color: #f9f9f9; border-left: 5px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px; }}
                    </style>
                </head>
                <body>
                    {html}
                </body>
                </html>
                """)
        except:
            print("Markdown conversion failed - HTML report not created")
        
        print(f"Comprehensive analysis report created in {output_dir}/")
        return os.path.join(output_dir, 'comprehensive_analysis.md')
    
    except Exception as e:
        print(f"Error creating summary report: {e}")
        return None

def analyze_ablation_findings():
    """Main function to analyze ablation findings"""
    print("Analyzing ablation findings...")
    
    # Load the data
    df = load_ablation_data()
    
    if df is not None:
        # Calculate key statistics
        stats = calculate_key_statistics(df)
        
        # Analyze policy types
        category_impacts = analyze_policy_types(df)
        
        # Generate insights
        insights = generate_insights(df, stats, category_impacts)
        
        # Create summary report
        report_path = create_summary_report(df, stats, category_impacts, insights)
        
        if report_path:
            print(f"Analysis complete. Report saved to {report_path}")
        else:
            print("Failed to create analysis report.")
    else:
        print("Failed to analyze data - ablation findings not available.")

if __name__ == "__main__":
    analyze_ablation_findings()
