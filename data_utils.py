#!/usr/bin/env python3

"""
Data utilities for Bitcoin policy ablation study
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def verify_chainalysis_data(chainalysis_data, countries):
    """
    Verify that Chainalysis data has been loaded correctly
    
    Args:
        chainalysis_data: DataFrame containing loaded Chainalysis data
        countries: List of expected countries
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    if chainalysis_data is None:
        print("ERROR: Chainalysis data is None!")
        return False
        
    # Check if we have data for all required countries
    for country in countries:
        if country not in chainalysis_data['country'].unique():
            print(f"WARNING: Missing Chainalysis data for {country}")
    
    # Check if we have data for all years
    for year in range(2021, 2025):
        if year not in chainalysis_data['year'].unique():
            print(f"WARNING: Missing Chainalysis data for year {year}")
    
    # Verify columns
    required_columns = ['country', 'year', '%_crypto_owning', 'on_chain_usd_volume_share']
    for col in required_columns:
        if col not in chainalysis_data.columns:
            alt_col = 'pct_crypto_owning' if col == '%_crypto_owning' else None
            if alt_col and alt_col in chainalysis_data.columns:
                print(f"INFO: Found '{alt_col}' instead of '{col}', will rename.")
            else:
                print(f"ERROR: Required column '{col}' not found in Chainalysis data!")
                return False
    
    print("Chainalysis data verification passed!")
    return True


def visualize_chainalysis_data(panel, countries, output_dir='.'):
    """
    Create visualizations of Chainalysis data trends
    
    Args:
        panel: DataFrame containing the final panel dataset with all features
        countries: List of countries to include in visualizations
        output_dir: Directory to save visualization files
    """
    # Plot crypto ownership percentage by country over time
    plt.figure(figsize=(12, 6))
    for country in countries:
        country_data = panel[panel['country'] == country].copy()
        country_data = country_data.sort_values('date')
        plt.plot(country_data['date'], country_data['%_crypto_owning'], 
                 'o-', label=country)
    
    plt.title('Cryptocurrency Ownership Percentage by Country')
    plt.xlabel('Date')
    plt.ylabel('% of Population Owning Crypto')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chainalysis_ownership_trends.png'))
    plt.close()
    
    # Plot on-chain volume share by country over time
    plt.figure(figsize=(12, 6))
    for country in countries:
        country_data = panel[panel['country'] == country].copy()
        country_data = country_data.sort_values('date')
        plt.plot(country_data['date'], country_data['on_chain_usd_volume_share'], 
                 'o-', label=country)
    
    plt.title('On-Chain USD Volume Share by Country')
    plt.xlabel('Date')
    plt.ylabel('Share of Global On-Chain Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chainalysis_volume_trends.png'))
    plt.close()


def create_adoption_projections(chainalysis_data, forecast_end_date='2025-04-30'):
    """
    Create projections of adoption metrics beyond available data
    
    Args:
        chainalysis_data: DataFrame containing historical Chainalysis data
        forecast_end_date: End date for forecasting (string in YYYY-MM-DD format)
        
    Returns:
        DataFrame: Original data plus projected values
    """
    print("Creating adoption projections...")
    
    # Make a copy of the data to avoid modifying the original
    data = chainalysis_data.copy()
    
    # Get the last year in the data
    last_year = data['year'].max()
    
    # Extract the year from forecast_end_date
    forecast_end_year = int(forecast_end_date.split('-')[0])
    
    # If we already have data up to the forecast end year, no projection needed
    if last_year >= forecast_end_year:
        return data
    
    # Create projections for missing years
    projection_years = range(last_year + 1, forecast_end_year + 1)
    projections = []
    
    for country in data['country'].unique():
        # Get country-specific data
        country_data = data[data['country'] == country].sort_values('year')
        
        # Calculate average yearly growth rates
        pct_owning_growth = np.mean(country_data['%_crypto_owning'].pct_change().dropna())
        volume_share_growth = np.mean(country_data['on_chain_usd_volume_share'].pct_change().dropna())
        
        # Set floor and ceiling values to avoid unrealistic projections
        pct_owning_growth = max(-0.1, min(0.2, pct_owning_growth))
        volume_share_growth = max(-0.1, min(0.2, volume_share_growth))
        
        # Use the last available values as starting point
        last_pct_owning = country_data['%_crypto_owning'].iloc[-1]
        last_volume_share = country_data['on_chain_usd_volume_share'].iloc[-1]
        
        # Generate projections for future years
        for year in projection_years:
            # Update values using growth rates
            last_pct_owning *= (1 + pct_owning_growth)
            last_volume_share *= (1 + volume_share_growth)
            
            # Cap the values to reasonable ranges
            last_pct_owning = min(0.5, max(0.01, last_pct_owning))  # 1% to 50% adoption
            last_volume_share = min(0.5, max(0.01, last_volume_share))  # 1% to 50% volume share
            
            projections.append({
                'country': country,
                'year': year,
                '%_crypto_owning': last_pct_owning,
                'on_chain_usd_volume_share': last_volume_share
            })
    
    # Add projections to original data
    projection_df = pd.DataFrame(projections)
    combined_data = pd.concat([data, projection_df], ignore_index=True)
    
    print(f"Created projections through {forecast_end_year}.")
    return combined_data
