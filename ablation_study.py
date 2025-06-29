#!/usr/bin/env python3

"""
Bitcoin Ablation Study: Cryptocurrency Regulation and Adoption Analysis
A comparative study of the United States, Russia, and Indonesia with GDP normalization

This script implements an end-to-end machine learning pipeline for analyzing 
the effects of regulatory policies on Bitcoin volume and adoption across
different countries, using GDP data for proper volume normalization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import joblib
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Constants
COUNTRIES = ['United States', 'Russia', 'Indonesia']
START_DATE = '2018-01-01'
TRAIN_END_DATE = '2022-12-31'
VAL_START_DATE = '2023-01-01'
VAL_END_DATE = '2025-05-17'  # Current date in prompt is May 17, 2025
FORECAST_MONTHS = 12  # Number of months to forecast for policy scenarios

# File paths
BTC_DATA_PATH = 'btc_1d_data_2018_to_2025.csv'
CHAINALYSIS_FILES = [
    'chainalysis_adoption_2021_Version3.csv',
    'chainalysis_adoption_2022_Version4.csv',
    'chainalysis_adoption_2023_Version4.csv', 
    'chainalysis_adoption_2024_Version4.csv'
]
GDP_DATA_PATH = 'GDP.csv'
MACRO_CONTROLS_PATH = 'macro_controls_monthly.csv'
POLICY_PANEL_PATH = 'policy_panel_monthly.csv'
OUTPUT_PANEL_PATH = 'panel_with_gdp.csv'
MODEL_SAVE_PATH = 'xgb_btc_policy_ablation.pkl'
SCENARIO_SAVE_PATH = 'scenarios.csv'
VISUALIZATION_DIR = 'visualizations'

# Feature columns for the model
FEATURE_COLS = [
    'lag1_vol', 'lag3_vol', 'volatility_3m', 'vol_pct_change_1m',
    'classified_as_security', 'classified_as_commodity', 'classified_as_DFA',
    'exchange_license_required', 'license_intensity',
    'tax_rate', 'aml_flag', 'aml_severity',
    'payment_ban', '%_crypto_owning',
    'gdp_growth', 'usd_fx_rate', 'cpi_inflation',
    'vol_to_gdp_ratio'  # New GDP-normalized feature
]


def load_and_prep_btc_data(file_path):
    """
    Load and preprocess Bitcoin OHLCV data
    """
    print(f"Loading Bitcoin data from {file_path}...")
    
    try:
        # Load BTC data
        btc_data = pd.read_csv(file_path)
        
        # Convert timestamps to datetime
        btc_data['Open time'] = pd.to_datetime(btc_data['Open time'])
        btc_data['Close time'] = pd.to_datetime(btc_data['Close time'])
        
        # Set date as index
        btc_data.set_index('Open time', inplace=True)
        
        # Resample to monthly data
        monthly_btc = btc_data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Quote asset volume': 'sum',
            'Number of trades': 'sum'
        })
        
        # Reset index for easier manipulation
        monthly_btc.reset_index(inplace=True)
        
        # Rename date column
        monthly_btc.rename(columns={'Open time': 'date'}, inplace=True)
        
        print(f"Successfully processed Bitcoin data. Shape: {monthly_btc.shape}")
        
        return monthly_btc
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Creating sample BTC data for demonstration...")
        
        # Create a date range from 2018-01-01 to 2025-05-17
        date_range = pd.date_range(start=START_DATE, end=VAL_END_DATE, freq='M')
        
        # Create sample data
        sample_data = pd.DataFrame({
            'date': date_range,
            'Open': np.random.normal(20000, 10000, len(date_range)),
            'High': np.random.normal(25000, 10000, len(date_range)),
            'Low': np.random.normal(18000, 8000, len(date_range)),
            'Close': np.random.normal(22000, 10000, len(date_range)),
            'Volume': np.random.normal(100000, 50000, len(date_range)),
            'Quote asset volume': np.random.normal(2000000000, 1000000000, len(date_range)),
            'Number of trades': np.random.normal(1000000, 500000, len(date_range))
        })
        
        print(f"Created sample Bitcoin data. Shape: {sample_data.shape}")
        
        return sample_data


def load_chainalysis_data(file_paths):
    """
    Load and combine Chainalysis adoption data
    """
    print("Loading Chainalysis adoption data...")
    
    try:
        all_data = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                # Load the CSV file
                data = pd.read_csv(file_path)
                
                # Rename column to match expected format if needed
                if 'pct_crypto_owning' in data.columns:
                    data = data.rename(columns={'pct_crypto_owning': '%_crypto_owning'})
                
                all_data.append(data)
            else:
                print(f"Warning: File {file_path} not found.")
        
        if all_data:
            chainalysis_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully loaded Chainalysis data. Shape: {chainalysis_data.shape}")
            print(f"Countries: {chainalysis_data['country'].unique()}")
            print(f"Years: {chainalysis_data['year'].unique()}")
            return chainalysis_data
        else:
            raise FileNotFoundError("No Chainalysis data files found.")
            
    except Exception as e:
        print(f"Error loading Chainalysis data: {str(e)}")
        print("Creating sample Chainalysis data for demonstration...")
        
        # Create sample data for all countries and years
        countries = COUNTRIES
        years = range(2021, 2025)
        
        data = []
        for country in countries:
            for year in years:
                data.append({
                    'country': country,
                    'year': year,
                    '%_crypto_owning': np.random.uniform(0.05, 0.25),
                    'on_chain_usd_volume_share': np.random.uniform(0.01, 0.15)
                })
        
        sample_data = pd.DataFrame(data)
        print(f"Created sample Chainalysis data. Shape: {sample_data.shape}")
        
        return sample_data


def load_macro_controls(file_path):
    """
    Load macroeconomic control data
    """
    print(f"Loading macroeconomic controls from {file_path}...")
    
    try:
        macro_data = pd.read_csv(file_path)
        macro_data['date'] = pd.to_datetime(macro_data['date'])
        print(f"Successfully loaded macro controls. Shape: {macro_data.shape}")
        return macro_data
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Creating sample macroeconomic data for demonstration...")
        
        # Create a date range from 2018-01-01 to 2025-05-17
        date_range = pd.date_range(start=START_DATE, end=VAL_END_DATE, freq='M')
        
        # Sample data for each country
        data = []
        for country in COUNTRIES:
            for date in date_range:
                data.append({
                    'date': date,
                    'country': country,
                    'gdp_growth': np.random.normal(0.02, 0.01),
                    'usd_fx_rate': np.random.uniform(0.5, 100),
                    'cpi_inflation': np.random.normal(0.03, 0.015)
                })
        
        sample_data = pd.DataFrame(data)
        print(f"Created sample macroeconomic data. Shape: {sample_data.shape}")
        
        return sample_data


def load_policy_panel(file_path):
    """
    Load policy panel data
    """
    print(f"Loading policy panel from {file_path}...")
    
    try:
        policy_data = pd.read_csv(file_path)
        policy_data['date'] = pd.to_datetime(policy_data['date'])
        print(f"Successfully loaded policy panel. Shape: {policy_data.shape}")
        return policy_data
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Creating sample policy panel data for demonstration...")
        
        # Create a date range from 2018-01-01 to 2025-05-17
        date_range = pd.date_range(start=START_DATE, end=VAL_END_DATE, freq='M')
        
        # Sample data for each country
        data = []
        for country in COUNTRIES:
            for date in date_range:
                # Create different policy regimes by date for realism
                year = date.year
                month = date.month
                
                # Different countries have different policies
                if country == 'United States':
                    security = 1 if year >= 2020 else 0
                    commodity = 1
                    dfa = 0
                    license = 1 if year >= 2019 else 0
                    license_intensity = 0.7 if year >= 2019 else 0.3
                    tax_rate = 0.20 if year >= 2021 else 0.15
                    aml_flag = 1
                    aml_severity = 0.8 if year >= 2020 else 0.6
                    payment_ban = 0
                elif country == 'Russia':
                    security = 0
                    commodity = 1 if year >= 2021 else 0
                    dfa = 1 if year >= 2022 else 0
                    license = 1 if year >= 2022 else 0
                    license_intensity = 0.9 if year >= 2022 else 0.2
                    tax_rate = 0.13
                    aml_flag = 1 if year >= 2021 else 0
                    aml_severity = 0.7 if year >= 2021 else 0.3
                    payment_ban = 1 if year >= 2023 else 0
                else:  # Indonesia
                    security = 0
                    commodity = 1
                    dfa = 0
                    license = 1 if year >= 2020 else 0
                    license_intensity = 0.6
                    tax_rate = 0.30 if year >= 2022 else 0.25
                    aml_flag = 1
                    aml_severity = 0.5
                    payment_ban = 1 if (2019 <= year <= 2021) else 0
                
                # Add some randomness for policy changes
                if np.random.random() < 0.05:  # 5% chance of policy change
                    security = 1 - security
                if np.random.random() < 0.05:
                    commodity = 1 - commodity
                if np.random.random() < 0.05:
                    dfa = 1 - dfa
                    
                data.append({
                    'date': date,
                    'country': country,
                    'classified_as_security': security,
                    'classified_as_commodity': commodity,
                    'classified_as_DFA': dfa,
                    'exchange_license_required': license,
                    'license_intensity': license_intensity,
                    'tax_rate': tax_rate,
                    'aml_flag': aml_flag,
                    'aml_severity': aml_severity,
                    'payment_ban': payment_ban
                })
        
        sample_data = pd.DataFrame(data)
        print(f"Created sample policy panel data. Shape: {sample_data.shape}")
        
        return sample_data


def load_gdp_data(file_path):
    """
    Load GDP data from IMF dataset
    """
    print(f"Loading GDP data from {file_path}...")
    
    try:
        # Process the CSV with semicolon delimiter
        df = pd.read_csv(file_path, delimiter=';', encoding='utf-8-sig', skiprows=1)
        
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
        
        # Convert GDP values to numeric, handling European number format if needed
        gdp_long['gdp_billions_usd'] = pd.to_numeric(gdp_long['gdp_billions_usd'].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Convert year to int
        gdp_long['year'] = pd.to_numeric(gdp_long['year'])
        gdp_long = gdp_long.dropna(subset=['year', 'gdp_billions_usd'])
        
        print(f"Successfully loaded GDP data: {len(gdp_long)} entries from {int(gdp_long['year'].min())} to {int(gdp_long['year'].max())}")
        print(f"Countries: {', '.join(gdp_long['country'].unique())}")
        
        return gdp_long
        
    except Exception as e:
        print(f"Error loading GDP data: {str(e)}")
        print("Creating sample GDP data for demonstration...")
        
        # Create sample GDP data
        data = []
        for country in COUNTRIES:
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


def create_panel(btc_data, chainalysis_data, macro_data, policy_data, gdp_data, output_path):
    """
    Create a merged panel dataset from all sources, incorporating GDP data
    """
    print("Creating panel dataset with GDP integration...")
    
    # Create multi-country BTC data
    countries_btc = []
    for country in COUNTRIES:
        country_btc = btc_data.copy()
        country_btc['country'] = country
        countries_btc.append(country_btc)
    
    multi_country_btc = pd.concat(countries_btc, ignore_index=True)
    
    # Convert Chainalysis yearly data to monthly
    chain_monthly = []
    for _, row in chainalysis_data.iterrows():
        year = row['year']
        # Create 12 monthly entries for each yearly record
        for month in range(1, 13):
            date = pd.to_datetime(f"{year}-{month:02d}-01")
            chain_monthly.append({
                'date': date,
                'country': row['country'],
                '%_crypto_owning': row['%_crypto_owning'],
                'on_chain_usd_volume_share': row['on_chain_usd_volume_share']
            })
    
    chainalysis_monthly = pd.DataFrame(chain_monthly)
    
    # --- Process GDP Data ---
    # Convert annual GDP data to monthly
    # First, add year column to btc_data dates
    multi_country_btc['year'] = multi_country_btc['date'].dt.year
    
    # Merge annual GDP data with BTC data
    print("Merging GDP data with panel...")
    gdp_data_renamed = gdp_data.rename(columns={'gdp_billions_usd': 'annual_gdp_billions_usd'})
    
    # Merge all data sources
    panel = multi_country_btc.merge(
        macro_data, on=['date', 'country'], how='left'
    ).merge(
        policy_data, on=['date', 'country'], how='left'
    ).merge(
        chainalysis_monthly, on=['date', 'country'], how='left'
    ).merge(
        gdp_data_renamed, on=['country', 'year'], how='left'
    )
    
    # Sort by country and date for proper filling
    panel = panel.sort_values(['country', 'date'])
    
    # Compute monthly GDP - divide annual GDP by 12 as a basic approach
    panel['monthly_gdp_billions_usd'] = panel['annual_gdp_billions_usd'] / 12
    
    # Create GDP-normalized volume metrics
    panel['vol_to_gdp_ratio'] = panel['Volume'] / (panel['monthly_gdp_billions_usd'] * 1e6)
    panel['usd_vol_to_gdp_ratio'] = panel['Quote asset volume'] / (panel['monthly_gdp_billions_usd'] * 1e9)
    
    # Ensure we have values for each country before 2021 (when we start having Chainalysis data)
    # For dates before our earliest data point, use the earliest value we have
    first_chainalysis_year = chainalysis_monthly['date'].dt.year.min() if not chainalysis_monthly.empty else 2021
    earliest_data = chainalysis_monthly.groupby('country').first().reset_index()
    
    for _, row in earliest_data.iterrows():
        country_mask = (panel['country'] == row['country']) & (panel['date'].dt.year < first_chainalysis_year)
        for col in ['%_crypto_owning', 'on_chain_usd_volume_share']:
            panel.loc[country_mask, col] = row[col]
    
    # Forward fill Chainalysis data for missing months
    panel['%_crypto_owning'] = panel.groupby('country')['%_crypto_owning'].fillna(method='ffill')
    panel['on_chain_usd_volume_share'] = panel.groupby('country')['on_chain_usd_volume_share'].fillna(method='ffill')
    
    # Backward fill for any remaining NaNs in those columns
    panel['%_crypto_owning'] = panel.groupby('country')['%_crypto_owning'].fillna(method='bfill')
    panel['on_chain_usd_volume_share'] = panel.groupby('country')['on_chain_usd_volume_share'].fillna(method='bfill')
    
    # Fill any remaining NAs with zeros
    panel.fillna(0, inplace=True)
    
    # Save panel to CSV
    print(f"Saving GDP-integrated panel dataset to {output_path}...")
    panel.to_csv(output_path, index=False)
    print(f"Panel dataset created with shape: {panel.shape}")
    
    return panel


def feature_engineering(panel):
    """
    Apply feature engineering to the panel dataset with GDP normalization
    """
    print("Performing feature engineering with GDP normalization...")
    
    # Normalize volume using actual GDP data
    if 'vol_to_gdp_ratio' not in panel.columns:
        panel['vol_to_gdp_ratio'] = panel['Volume'] / (panel['monthly_gdp_billions_usd'] * 1e6)
    
    # Use our GDP-normalized volume as the main metric
    panel['norm_volume'] = panel['vol_to_gdp_ratio']
    
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
    
    print("Feature engineering completed with GDP normalization.")
    return panel


def train_test_split(panel, train_end_date, val_start_date):
    """
    Split data into training and validation sets
    """
    print(f"Splitting data: Train until {train_end_date}, Validation from {val_start_date}")
    
    train_data = panel[panel['date'] <= train_end_date].copy()
    val_data = panel[(panel['date'] >= val_start_date)].copy()
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Extract features and target
    X_train = train_data[FEATURE_COLS]
    y_train = train_data['target']
    
    X_val = val_data[FEATURE_COLS]
    y_val = val_data['target']
    
    return X_train, y_train, X_val, y_val, train_data, val_data


def train_model(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost regression model
    """
    print("Training XGBoost model...")
    
    # Initialize XGBoost regressor
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Fit model with validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    print("Model training completed.")
    return model


def evaluate_model(model, X_val, y_val, val_data):
    """
    Evaluate model performance and generate visualizations
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Create evaluation plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual GDP-Normalized Volume')
    plt.ylabel('Predicted GDP-Normalized Volume')
    plt.title('Actual vs. Predicted GDP-Normalized Volume')
    
    # Draw a linear regression line
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(y_val.values.reshape(-1, 1), y_pred.reshape(-1, 1))
    plt.plot(y_val, reg.predict(y_val.values.reshape(-1, 1)), 'b-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_gdp_normalized.png')
    plt.close()
    
    # Create country-specific evaluation plots
    val_data['predicted'] = y_pred
    
    plt.figure(figsize=(15, 10))
    
    for i, country in enumerate(COUNTRIES):
        plt.subplot(len(COUNTRIES), 1, i+1)
        
        country_data = val_data[val_data['country'] == country].sort_values('date')
        
        plt.plot(country_data['date'], country_data['target'], 'o-', label='Actual')
        plt.plot(country_data['date'], country_data['predicted'], 's--', label='Predicted')
        
        plt.title(f'{country}: Actual vs Predicted GDP-Normalized Volume')
        plt.ylabel('GDP-Norm. Vol.')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('country_predictions_gdp_normalized.png')
    plt.close()
    
    print("Model evaluation completed and plots saved.")
    
    return val_data  # Return for further analysis


def create_shap_plots(model, X_val):
    """
    Generate SHAP value explanations
    """
    print("Generating SHAP explanations...")
    
    # Create a directory for SHAP visualizations if it doesn't exist
    shap_dir = "shap_analysis"
    if not os.path.exists(shap_dir):
        os.makedirs(shap_dir)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)
    
    # Global feature importance (bar plot)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_vals, X_val, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{shap_dir}/shap_feature_importance_bar.png')
    plt.close()
    
    # Global feature importance (beeswarm plot)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_vals, X_val, show=False)
    plt.title('SHAP Feature Impact Distribution')
    plt.tight_layout()
    plt.savefig(f'{shap_dir}/shap_feature_importance_beeswarm.png')
    plt.close()
    
    # Categorize features for organized analysis
    feature_categories = {
        'policy': [
            'classified_as_security', 'classified_as_commodity', 
            'classified_as_DFA', 'exchange_license_required', 
            'license_intensity', 'tax_rate', 'aml_flag',
            'aml_severity', 'payment_ban'
        ],
        'adoption': ['%_crypto_owning', 'on_chain_usd_volume_share'],
        'macro': ['gdp_growth', 'usd_fx_rate', 'cpi_inflation', 'vol_to_gdp_ratio'],
        'market': ['lag1_vol', 'lag3_vol', 'volatility_3m', 'vol_pct_change_1m']
    }
    
    # Generate dependence plots by category
    for category, features in feature_categories.items():
        category_features = [f for f in features if f in X_val.columns]
        
        if not category_features:
            continue
            
        for feature in category_features:
            plt.figure(figsize=(12, 6))
            shap.dependence_plot(feature, shap_vals, X_val, show=False)
            plt.title(f'SHAP Dependence Plot: {feature}')
            plt.tight_layout()
            plt.savefig(f'{shap_dir}/shap_dependence_{feature.replace("%", "pct")}.png')
            plt.close()
    
    print(f"SHAP explanations generated and plots saved to {shap_dir}/ directory.")
    return shap_vals


def simulate_scenario(start_row, model, months=12, override_policies=None):
    """
    Simulate different policy scenarios and forecast volume
    
    Args:
        start_row: pd.Series of last-known features
        model: Trained XGBoost model
        months: Number of months to forecast
        override_policies: dict of {feature: new_value} for policy changes
        
    Returns:
        List of predicted normalized volumes
    """
    cur = start_row.copy()
    results = []
    
    for m in range(months):
        # Apply overrides
        if override_policies:
            for k, v in override_policies.items():
                if k in cur.index:
                    cur[k] = v
                    
        # Predict next-month volume
        # Make sure we're using only the columns the model knows about
        features_only = {k: cur[k] for k in FEATURE_COLS if k in cur.index}
        features_df = pd.DataFrame([features_only])
        
        y_hat = model.predict(features_df)[0]
        results.append(y_hat)
        
        # Shift lags & features for next step
        # Update features for next prediction
        cur['lag1_vol'] = y_hat
        cur['lag3_vol'] = cur['lag1_vol'] if m >= 2 else cur['lag3_vol']
        cur['volatility_3m'] = cur['volatility_3m'] * 0.9  # Simple decay
        cur['vol_pct_change_1m'] = 0.1 * np.random.randn()  # Some random variation
        
    return results


def run_ablation_scenarios(panel, model):
    """
    Run comprehensive policy ablation scenarios and save results
    """
    print("Running policy ablation scenarios...")
    
    # Create a directory for results if it doesn't exist
    if not os.path.exists(VISUALIZATION_DIR):
        os.makedirs(VISUALIZATION_DIR)
    
    scenarios = []
    
    # Get the most recent data point for each country
    # We'll use this as the starting point for all scenarios
    latest_data = {}
    for country in COUNTRIES:
        country_data = panel[panel['country'] == country].sort_values('date')
        if len(country_data) > 0:
            latest_data[country] = country_data.iloc[-1]
    
    # Define policy fields that can be modified
    policy_fields = [
        'classified_as_security', 'classified_as_commodity', 
        'classified_as_DFA', 'exchange_license_required',
        'license_intensity', 'tax_rate', 'aml_flag',
        'aml_severity', 'payment_ban'
    ]
    
    # For each country, run multiple policy scenarios
    for country, start_row in latest_data.items():
        print(f"Running scenarios for {country}...")
        
        # 1. Baseline scenario - current policies
        baseline = simulate_scenario(start_row, model, months=FORECAST_MONTHS)
        
        for i, vol in enumerate(baseline):
            scenarios.append({
                'country': country,
                'scenario': 'baseline',
                'month': i + 1,
                'predicted_norm_volume': vol
            })
        
        # 2. Individual policy ablation (removing one policy at a time)
        for field in policy_fields:
            # Only run if the field is active (value > 0)
            if start_row[field] > 0:
                scenario_name = f"no_{field}"
                results = simulate_scenario(start_row, model, months=FORECAST_MONTHS, 
                                          override_policies={field: 0})
                
                for i, vol in enumerate(results):
                    scenarios.append({
                        'country': country,
                        'scenario': scenario_name,
                        'month': i + 1,
                        'predicted_norm_volume': vol
                    })
        
        # 3. Policy combinations
        # 3.1 No regulations at all
        no_regulation_overrides = {field: 0 for field in policy_fields}
        results = simulate_scenario(start_row, model, months=FORECAST_MONTHS, 
                                  override_policies=no_regulation_overrides)
        
        for i, vol in enumerate(results):
            scenarios.append({
                'country': country,
                'scenario': 'no_regulations',
                'month': i + 1,
                'predicted_norm_volume': vol
            })
        
        # 3.2 Maximum regulations
        max_regulation_overrides = {
            'classified_as_security': 1,
            'classified_as_commodity': 1, 
            'classified_as_DFA': 1,
            'exchange_license_required': 1,
            'license_intensity': 1.0,
            'tax_rate': 0.5,
            'aml_flag': 1,
            'aml_severity': 1.0,
            'payment_ban': 1
        }
        results = simulate_scenario(start_row, model, months=FORECAST_MONTHS, 
                                  override_policies=max_regulation_overrides)
        
        for i, vol in enumerate(results):
            scenarios.append({
                'country': country,
                'scenario': 'max_regulations',
                'month': i + 1,
                'predicted_norm_volume': vol
            })
        
        # 4. Interesting combinations
        # 4.1 No AML and no tax
        if start_row['tax_rate'] > 0 and start_row['aml_flag'] > 0:
            scenario_name = "no_tax_no_aml"
            results = simulate_scenario(
                start_row, model, months=FORECAST_MONTHS,
                override_policies={'tax_rate': 0, 'aml_flag': 0, 'aml_severity': 0}
            )
            
            for i, vol in enumerate(results):
                scenarios.append({
                    'country': country,
                    'scenario': scenario_name,
                    'month': i + 1,
                    'predicted_norm_volume': vol
                })
                
        # 4.2 No license requirements and no ban
        if start_row['exchange_license_required'] > 0 or start_row['payment_ban'] > 0:
            scenario_name = "no_license_no_ban"
            results = simulate_scenario(
                start_row, model, months=FORECAST_MONTHS,
                override_policies={
                    'exchange_license_required': 0, 
                    'license_intensity': 0, 
                    'payment_ban': 0
                }
            )
            
            for i, vol in enumerate(results):
                scenarios.append({
                    'country': country,
                    'scenario': scenario_name,
                    'month': i + 1,
                    'predicted_norm_volume': vol
                })
        
        # 5. Cross-country policy transplants
        other_countries = [c for c in COUNTRIES if c != country]
        for other_country in other_countries:
            if other_country in latest_data:
                other_row = latest_data[other_country]
                
                # Extract policy settings from the other country
                policy_overrides = {}
                for field in policy_fields:
                    policy_overrides[field] = other_row[field]
                
                scenario_name = f"{country}_with_{other_country.replace(' ', '_')}_policies"
                results = simulate_scenario(start_row, model, months=FORECAST_MONTHS, 
                                          override_policies=policy_overrides)
                
                for i, vol in enumerate(results):
                    scenarios.append({
                        'country': country,
                        'scenario': scenario_name,
                        'month': i + 1,
                        'predicted_norm_volume': vol
                    })
    
    # Save all scenarios to CSV
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df.to_csv(SCENARIO_SAVE_PATH, index=False)
    print(f"Scenario results saved to {SCENARIO_SAVE_PATH}")
    
    # Create visualizations
    visualize_scenarios(scenarios_df)
    
    return scenarios_df


def visualize_scenarios(scenarios_df):
    """
    Create visualizations of scenario results
    """
    print("Visualizing policy ablation scenarios...")
    
    # Group scenarios into categories for better visualization
    scenario_categories = {
        'baseline': ['baseline'],
        'single_policy_ablation': [s for s in scenarios_df['scenario'].unique() if s.startswith('no_') and '_no_' not in s],
        'multi_policy_ablation': [s for s in scenarios_df['scenario'].unique() if '_no_' in s],
        'cross_country': [s for s in scenarios_df['scenario'].unique() if 'with_' in s and 'policies' in s],
        'extreme': ['no_regulations', 'max_regulations']
    }
    
    # 1. Line plots of predicted volumes by country and scenario category
    for country in scenarios_df['country'].unique():
        country_scenarios = scenarios_df[scenarios_df['country'] == country]
        baseline_data = country_scenarios[country_scenarios['scenario'] == 'baseline']
        
        # Create separate plots for each category
        for category_name, category_scenarios in scenario_categories.items():
            matching_scenarios = [s for s in category_scenarios if s in country_scenarios['scenario'].unique()]
            if not matching_scenarios and category_name != 'baseline':
                continue
                
            plt.figure(figsize=(14, 8))
            
            # Always plot baseline for reference
            plt.plot(baseline_data['month'], baseline_data['predicted_norm_volume'], 
                    marker='o', linewidth=3, color='black', label='baseline')
            
            # Plot category-specific scenarios
            for scenario in matching_scenarios:
                if scenario == 'baseline':
                    continue  # Already plotted
                    
                data = country_scenarios[country_scenarios['scenario'] == scenario]
                if len(data) > 0:
                    plt.plot(data['month'], data['predicted_norm_volume'], marker='o', label=scenario)
            
            plt.xlabel('Forecast Month')
            plt.ylabel('Predicted GDP-Normalized Volume')
            plt.title(f'{category_name.replace("_", " ").title()} Scenarios: {country}')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{VISUALIZATION_DIR}/scenario_{category_name}_{country.replace(" ", "_")}.png')
            plt.close()
    
    # 2. Calculate impact of scenarios compared to baseline
    impact_data = []
    
    for country in scenarios_df['country'].unique():
        country_data = scenarios_df[scenarios_df['country'] == country]
        baseline = country_data[country_data['scenario'] == 'baseline']
        baseline_avg = baseline['predicted_norm_volume'].mean()
        baseline_end = baseline.iloc[-1]['predicted_norm_volume'] if len(baseline) > 0 else 0
        
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
        
        # 3. Bar charts of impact by country
        for country in impact_df['country'].unique():
            country_impact = impact_df[impact_df['country'] == country].sort_values('avg_pct_change')
            
            # Limit to top and bottom 15 impacts if there are too many
            if len(country_impact) > 30:
                top = country_impact.nlargest(15, 'avg_pct_change')
                bottom = country_impact.nsmallest(15, 'avg_pct_change')
                country_impact = pd.concat([top, bottom])
            
            plt.figure(figsize=(16, 10))
            sns.barplot(x='scenario', y='avg_pct_change', data=country_impact, hue='category', dodge=False)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('Policy Scenario')
            plt.ylabel('% Change in Volume (Average over forecast period)')
            plt.title(f'Policy Impact on GDP-Normalized Trading Volume: {country}')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{VISUALIZATION_DIR}/policy_impact_avg_{country.replace(" ", "_")}.png')
            plt.close()
        
        # 4. Heatmap of policy impacts across countries
        pivot_data = impact_df.pivot_table(
            index='scenario', 
            columns='country', 
            values='avg_pct_change',
            aggfunc='mean'
        )
        
        # Sort scenarios by their average impact
        pivot_data['avg_impact'] = pivot_data.mean(axis=1)
        pivot_data = pivot_data.sort_values('avg_impact', ascending=False)
        pivot_data = pivot_data.drop(columns=['avg_impact'])
        
        plt.figure(figsize=(12, max(8, len(pivot_data) * 0.4)))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f',
                   cbar_kws={'label': '% Change in GDP-Normalized Volume'})
        plt.title('Cross-Country Policy Impact Comparison')
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATION_DIR}/policy_impact_heatmap.png')
        plt.close()
        
        # 5. Distribution of impacts by category
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='category', y='avg_pct_change', data=impact_df)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Scenario Category')
        plt.ylabel('% Change in GDP-Normalized Volume')
        plt.title('Distribution of Policy Impacts by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATION_DIR}/impact_distribution_by_category.png')
        plt.close()
    
    print(f"Scenario visualizations created in {VISUALIZATION_DIR}/ directory.")


def main():
    """
    Main execution pipeline
    """
    print("=" * 80)
    print("BITCOIN ABLATION STUDY: REGULATION & ADOPTION ANALYSIS")
    print("A comparative study of the United States, Russia, and Indonesia")
    print("Using GDP data for volume normalization")
    print("=" * 80)
    
    # 1. Load all data sources
    btc_data = load_and_prep_btc_data(BTC_DATA_PATH)
    chainalysis_data = load_chainalysis_data(CHAINALYSIS_FILES)
    macro_data = load_macro_controls(MACRO_CONTROLS_PATH)
    policy_data = load_policy_panel(POLICY_PANEL_PATH)
    gdp_data = load_gdp_data(GDP_DATA_PATH)
    
    # 2. Create merged panel dataset with GDP integration
    panel = create_panel(btc_data, chainalysis_data, macro_data, policy_data, gdp_data, OUTPUT_PANEL_PATH)
    
    # 3. Apply feature engineering with GDP normalization
    panel = feature_engineering(panel)
    
    # 4. Split into training and validation sets
    X_train, y_train, X_val, y_val, train_data, val_data = train_test_split(
        panel, TRAIN_END_DATE, VAL_START_DATE
    )
    
    # 5. Train an XGBoost model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 6. Evaluate the model
    eval_results = evaluate_model(model, X_val, y_val, val_data)
    
    # 7. Generate SHAP explanations
    shap_vals = create_shap_plots(model, X_val)
    
    # 8. Run policy ablation scenarios
    scenario_results = run_ablation_scenarios(panel, model)
    
    # 9. Save the model for future use
    joblib.dump(model, MODEL_SAVE_PATH)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Generated files:")
    print(f"- GDP-integrated panel dataset: {OUTPUT_PANEL_PATH}")
    print(f"- Trained model: {MODEL_SAVE_PATH}")
    print(f"- Evaluation plots: model_evaluation_gdp_normalized.png, country_predictions_gdp_normalized.png")
    print(f"- SHAP plots: shap_analysis/shap_feature_importance_bar.png, etc.")
    print(f"- Scenario results: {SCENARIO_SAVE_PATH}")
    print(f"- Scenario visualizations: {VISUALIZATION_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
