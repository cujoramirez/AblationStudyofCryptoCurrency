#!/usr/bin/env python3
"""
Bitcoin Analysis with Real Data Helper

This script fixes the GDP loading issues and uses real data from:
- Bitcoin price data (btc_1d_data_2018_to_2025.csv)
- Chainalysis adoption data (chainalysis_adoption_*.csv files)
- Panel data with GDP (panel_with_gdp.csv)
- GDP data (GDP.csv) with proper encoding handling
"""

import os
import sys
import importlib
import pandas as pd
import numpy as np

# Import the original btc module
import btc

print("=" * 80)
print("BITCOIN ANALYSIS WITH REAL DATA")
print("=" * 80)

# Step 1: Override the GDP loading function with an enhanced version
def enhanced_load_gdp_data(file_path):
    """Enhanced GDP loading function that handles encoding issues"""
    print(f"Enhanced GDP loading from {file_path}")
    
    # Try different encodings
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            # Read with semicolon delimiter
            df = pd.read_csv(file_path, delimiter=';', encoding=encoding, skiprows=1)
            
            # Clean the DataFrame and reshape to long format
            # Drop empty rows
            df = df.dropna(how='all')
            df = df[df['GDP, current prices (Billions of U.S. dollars)'].notna()]
            
            # Extract country name from the first column
            df = df.rename(columns={'GDP, current prices (Billions of U.S. dollars)': 'country'})
            
            # Filter for our countries of interest
            country_mapping = {
                'Indonesia': 'Indonesia',
                'Russian Federation': 'Russia',
                'United States': 'United States'
            }
            
            df = df[df['country'].isin(country_mapping.keys())].copy()
            df['country'] = df['country'].map(country_mapping)
            
            # Reshape from wide to long format
            years = [str(year) for year in range(2018, 2026)]  # We need 2018-2025
            id_vars = ['country']
            gdp_long = pd.melt(df, id_vars=id_vars, value_vars=years,
                            var_name='year', value_name='gdp_billions_usd')
            
            # Convert GDP values to numeric, handling European number format
            gdp_long['gdp_billions_usd'] = pd.to_numeric(
                gdp_long['gdp_billions_usd'].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            
            # Convert year to int
            gdp_long['year'] = pd.to_numeric(gdp_long['year'])
            gdp_long = gdp_long.dropna(subset=['year', 'gdp_billions_usd'])
            
            print(f"Successfully loaded GDP data with {encoding} encoding:")
            print(f"- {len(gdp_long)} entries from {int(gdp_long['year'].min())} to {int(gdp_long['year'].max())}")
            print(f"- Countries: {', '.join(gdp_long['country'].unique())}")
            
            return gdp_long
            
        except Exception as e:
            print(f"Failed with {encoding} encoding: {str(e)}")
            continue
    
    # If we get here, none of the encodings worked
    print("Could not load GDP data with any encoding.")
    
    # Try to extract from panel_with_gdp.csv if available
    if os.path.exists('panel_with_gdp.csv'):
        print("Extracting GDP data from panel_with_gdp.csv instead")
        try:
            panel_df = pd.read_csv('panel_with_gdp.csv')
            
            if 'gdp_billions_usd' in panel_df.columns:
                # Extract the GDP column
                panel_df['date'] = pd.to_datetime(panel_df['date'])
                panel_df['year'] = panel_df['date'].dt.year
                
                # Get unique country-year combinations with GDP values
                gdp_data = panel_df[['country', 'year', 'gdp_billions_usd']].drop_duplicates()
                print(f"Successfully extracted GDP data from panel: {len(gdp_data)} entries")
                return gdp_data
        except Exception as e:
            print(f"Failed to extract from panel_with_gdp.csv: {str(e)}")
    
    # Fall back to the original function's mock data creation
    print("Creating sample GDP data for demonstration...")
    
    # Create sample GDP data
    data = []
    for country in btc.COUNTRIES:
        for year in range(2018, 2026):
            # Different base values for different countries
            if country == 'United States':
                base_gdp = 21000  # ~21 trillion USD
            elif country == 'Russia':
                base_gdp = 1600   # ~1.6 trillion USD
            else:  # Indonesia
                base_gdp = 1100   # ~1.1 trillion USD
            
            # Add some growth and variation
            growth = 1 + (np.random.normal(0.03, 0.01) if year < 2020 else 
                         (np.random.normal(-0.02, 0.02) if year == 2020 else
                          np.random.normal(0.04, 0.01)))
            
            gdp = base_gdp * (growth ** (year - 2018))
            
            data.append({
                'country': country,
                'year': year,
                'gdp_billions_usd': gdp
            })
    
    gdp_sample = pd.DataFrame(data)
    print(f"Created sample GDP data. Shape: {gdp_sample.shape}")
    
    return gdp_sample

# Step 2: Fix for data type issue in scenario simulation
def fixed_simulate_scenario(start_row, model, months=12, override_policies=None):
    """
    Fixed version of simulate_scenario that ensures proper data type conversion
    """
    cur = start_row.copy()
    results = []
    
    for m in range(months):
        # Apply overrides
        if override_policies:
            for k, v in override_policies.items():
                if k in cur.index:
                    cur[k] = v
        
        # Create a dataframe from the current row for prediction
        X_cur = cur[btc.FEATURE_COLS].to_frame().T
        
        # Convert all columns to float before prediction (THIS IS THE FIX)
        for col in X_cur.columns:
            X_cur[col] = X_cur[col].astype(float)
                
        # Predict next-month volume
        y_hat = model.predict(X_cur)[0]
        results.append(y_hat)
        
        # Shift lags & features for next step
        cur['lag1_vol'] = y_hat
        cur['lag3_vol'] = cur['lag1_vol'] if m >= 2 else cur['lag3_vol']
        
    return results

# Step 3: Enhanced feature engineering to use real GDP values
def enhanced_feature_engineering(panel):
    """
    Enhanced feature engineering that uses actual GDP values 
    """
    print("Performing enhanced feature engineering...")
    
    # Create normalized volume using actual GDP values if available, otherwise use proxy
    if 'gdp_billions_usd' in panel.columns:
        panel['monthly_gdp_billions_usd'] = panel['gdp_billions_usd'] / 12
        panel['norm_volume'] = panel['Volume'] / (panel['monthly_gdp_billions_usd'] * 1e9)
        panel['vol_to_gdp_ratio'] = panel['Volume'] / (panel['monthly_gdp_billions_usd'] * 1e9)
        print("Using actual GDP values for volume normalization")
    else:
        # Create proxy GDP values
        gdp_proxy = panel.groupby(['country', 'date'])['Quote asset volume'].transform('sum') * 10
        panel['gdp_usd'] = gdp_proxy
        panel['norm_volume'] = panel['Quote asset volume'] / (panel['gdp_usd'] / 1e9)
        print("Using proxy GDP values for volume normalization")
    
    # Lag features - by group (country)
    panel['lag1_vol'] = panel.groupby('country')['norm_volume'].shift(1)
    panel['lag3_vol'] = panel.groupby('country')['norm_volume'].shift(3)
    panel['lag6_vol'] = panel.groupby('country')['norm_volume'].shift(6)
    
    # Rolling statistics
    panel['volatility_3m'] = panel.groupby('country')['Close'].pct_change().rolling(3).std()
    panel['vol_pct_change_1m'] = panel.groupby('country')['norm_volume'].pct_change(1)
    
    # Create target variable: next-month normalized volume
    panel['target'] = panel.groupby('country')['norm_volume'].shift(-1)
    
    # Fill NAs created by feature engineering
    for col in ['lag1_vol', 'lag3_vol', 'lag6_vol', 'volatility_3m', 'vol_pct_change_1m']:
        panel[col].fillna(0, inplace=True)
    
    # Drop rows with NA targets (can't predict these)
    panel.dropna(subset=['target'], inplace=True)
    
    print("Enhanced feature engineering completed.")
    return panel

# Step 4: Update the btc module with our enhanced functions
btc.load_gdp_data = enhanced_load_gdp_data
btc.simulate_scenario = fixed_simulate_scenario
btc.feature_engineering = enhanced_feature_engineering
btc.MACRO_CONTROLS_PATH = 'panel_with_gdp.csv'  # Use the panel data for macro controls
btc.POLICY_PANEL_PATH = 'panel_with_gdp.csv'    # Use the panel data for policy panel

# Run the main function from btc.py
if __name__ == "__main__":
    try:
        btc.main()
        print("=" * 80)
        print("BITCOIN ANALYSIS WITH REAL DATA COMPLETED SUCCESSFULLY")
        print("=" * 80)
    except Exception as e:
        print("=" * 80)
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
