#!/usr/bin/env python3
"""
Bitcoin Ablation Study: End-to-End Pipeline
Analysis of Cryptocurrency Regulation and Adoption across the United States, Russia, and Indonesia
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import joblib
import warnings

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
FEATURE_COLS = [
    'lag1_vol', 'lag3_vol', 'volatility_3m', 'vol_pct_change_1m',
    'classified_as_security', 'classified_as_commodity', 'classified_as_DFA',
    'exchange_license_required', 'license_intensity',
    'tax_rate', 'aml_flag', 'aml_severity',
    'payment_ban', '%_crypto_owning',
    'gdp_growth', 'usd_fx_rate', 'cpi_inflation',
    'vol_to_gdp_ratio', 'gdp_billions_usd'
]

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
MACRO_CONTROLS_PATH = 'panel_with_gdp.csv'  # Use the panel data for macro controls
POLICY_PANEL_PATH = 'panel_with_gdp.csv'    # Use the panel data for policy panel
GDP_DATA_PATH = 'GDP.csv'
OUTPUT_PANEL_PATH = 'panel.csv'
MODEL_SAVE_PATH = 'xgb_btc_policy_ablation.pkl'
SCENARIO_SAVE_PATH = 'scenarios.csv'
VISUALIZATION_DIR = 'visualizations'


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
    
    except Exception as e:
        print(f"Error loading Bitcoin data: {e}")
        return None


def load_chainalysis_data(file_paths):
    """
    Load and merge Chainalysis adoption data from multiple years
    """
    try:
        # Load all Chainalysis data files and concatenate them
        chain_data_list = []
        
        for file_path in file_paths:
            print(f"Loading Chainalysis data from {file_path}...")
            try:
                df = pd.read_csv(file_path)
                chain_data_list.append(df)
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found")
                continue
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not chain_data_list:
            raise FileNotFoundError("No Chainalysis data files could be loaded")
            
        # Concatenate all dataframes
        chain_data = pd.concat(chain_data_list, ignore_index=True)
        
        # Ensure column names are consistent
        chain_data = chain_data.rename(columns={
            'pct_crypto_owning': 'pct_crypto_owning',
            'on_chain_usd_volume_share': 'on_chain_usd_volume_share'
        })
        
        print(f"Loaded Chainalysis data: {len(chain_data)} entries from {chain_data['year'].min()} to {chain_data['year'].max()}")
        print(f"Countries included: {', '.join(chain_data['country'].unique())}")
        return chain_data
    
    except Exception as e:
        print(f"Error processing Chainalysis data: {e}")
        return None


def load_macro_controls(file_path):
    """
    Extract macroeconomic control variables from panel data
    """
    try:
        if os.path.exists(file_path):
            print(f"Loading macro controls from {file_path}")
            
            panel_df = pd.read_csv(file_path)
            macro_cols = ['date', 'country', 'gdp_growth', 'usd_fx_rate', 'cpi_inflation']
            
            # Check if all required columns exist
            missing_cols = [col for col in macro_cols if col not in panel_df.columns]
            if not missing_cols:
                macro_data = panel_df[macro_cols].copy()
                macro_data['date'] = pd.to_datetime(macro_data['date'])
                print(f"Successfully extracted macro controls: {len(macro_data)} entries")
                return macro_data
            else:
                print(f"Missing macro columns in {file_path}: {missing_cols}")
                return None
    except Exception as e:
        print(f"Error loading macro controls: {e}")
        return None


def load_policy_panel(file_path):
    """
    Extract policy panel data from panel data
    """
    try:
        if os.path.exists(file_path):
            print(f"Loading policy panel from {file_path}")
            
            panel_df = pd.read_csv(file_path)
            policy_cols = ['date', 'country', 'classified_as_security', 'classified_as_commodity', 
                          'classified_as_DFA', 'exchange_license_required', 'license_intensity',
                          'tax_rate', 'aml_flag', 'aml_severity', 'payment_ban']
            
            # Check if all required columns exist
            missing_cols = [col for col in policy_cols if col not in panel_df.columns]
            if not missing_cols:
                policy_data = panel_df[policy_cols].copy()
                policy_data['date'] = pd.to_datetime(policy_data['date'])
                print(f"Successfully extracted policy data: {len(policy_data)} entries")
                return policy_data
            else:
                print(f"Missing policy columns in {file_path}: {missing_cols}")
                return None
    except Exception as e:
        print(f"Error loading policy panel: {e}")
        return None


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
        
        # Convert year to datetime by setting to January 1st of each year
        gdp_long['year'] = pd.to_numeric(gdp_long['year'])
        gdp_long = gdp_long.dropna(subset=['year', 'gdp_billions_usd'])
        
        print(f"Successfully loaded GDP data: {len(gdp_long)} entries from {int(gdp_long['year'].min())} to {int(gdp_long['year'].max())}")
        print(f"Countries: {', '.join(gdp_long['country'].unique())}")
        
        return gdp_long
        
    except Exception as e:
        print(f"Error loading GDP data: {str(e)}")
        
        # Try to extract from panel_with_gdp.csv if available
        if os.path.exists('panel_with_gdp.csv'):
            print("Extracting GDP data from panel_with_gdp.csv")
            panel_df = pd.read_csv('panel_with_gdp.csv')
            
            if 'annual_gdp_billions_usd' in panel_df.columns or 'gdp_billions_usd' in panel_df.columns:
                # Extract the GDP column (either name)
                gdp_col = 'annual_gdp_billions_usd' if 'annual_gdp_billions_usd' in panel_df.columns else 'gdp_billions_usd'
                
                # Extract year from date
                panel_df['date'] = pd.to_datetime(panel_df['date'])
                if 'year' not in panel_df.columns:
                    panel_df['year'] = panel_df['date'].dt.year
                
                # Get unique country-year combinations with GDP values
                gdp_data = panel_df[['country', 'year', gdp_col]].drop_duplicates().rename(columns={gdp_col: 'gdp_billions_usd'})
                print(f"Successfully extracted GDP data: {len(gdp_data)} entries")
                return gdp_data
            else:
                print("No GDP column found in panel_with_gdp.csv")
                return None
        return None


def create_panel(btc_data, chainalysis_data, macro_data, policy_data, gdp_data, output_path):
    """
    Create a merged panel dataset from all sources
    """
    print("Creating panel dataset...")
    
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
                '%_crypto_owning': row['pct_crypto_owning'],  # Adjust column name if needed
                'on_chain_usd_volume_share': row['on_chain_usd_volume_share']
            })
    
    chainalysis_monthly = pd.DataFrame(chain_monthly)
    
    # Sort data for merging
    multi_country_btc = multi_country_btc.sort_values(['country', 'date'])
    macro_data = macro_data.sort_values(['country', 'date']) if macro_data is not None else None
    policy_data = policy_data.sort_values(['country', 'date']) if policy_data is not None else None
    
    # Merge all data sources
    panel = multi_country_btc
    
    if macro_data is not None:
        panel = panel.merge(macro_data, on=['date', 'country'], how='left')
    
    if policy_data is not None:
        panel = panel.merge(policy_data, on=['date', 'country'], how='left')
    
    panel = panel.merge(chainalysis_monthly, on=['date', 'country'], how='left')
    
    # Add GDP data if available
    if gdp_data is not None:
        # Convert year to match with panel data
        gdp_data = gdp_data.copy()
        panel['year'] = pd.DatetimeIndex(panel['date']).year
        panel = panel.merge(gdp_data[['country', 'year', 'gdp_billions_usd']], 
                           on=['country', 'year'], how='left')
        
        # Calculate monthly GDP (simple division by 12)
        panel['monthly_gdp_billions_usd'] = panel['gdp_billions_usd'] / 12
        
        # Calculate volume to GDP ratio
        panel['vol_to_gdp_ratio'] = panel['Volume'] / (panel['monthly_gdp_billions_usd'] * 1e9)
        panel['usd_vol_to_gdp_ratio'] = panel['Quote asset volume'] / (panel['monthly_gdp_billions_usd'] * 1e9)
    
    # Sort by country and date for proper filling
    panel = panel.sort_values(['country', 'date'])
    
    # Ensure we have values for each country before 2021 (when we start having data)
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
    print(f"Saving panel dataset to {output_path}...")
    panel.to_csv(output_path, index=False)
    print(f"Panel dataset created with shape: {panel.shape}")
    
    return panel


def feature_engineering(panel):
    """
    Apply feature engineering to the panel dataset
    """
    print("Performing feature engineering...")
    
    # Create normalized volume using actual GDP values if available, otherwise use proxy
    if 'monthly_gdp_billions_usd' in panel.columns:
        panel['norm_volume'] = panel['Quote asset volume'] / (panel['monthly_gdp_billions_usd'] * 1e9)
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
    
    print("Feature engineering completed.")
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
    
    # Fit model with validation set (without early stopping)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    print("Model training completed.")
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance
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
    plt.xlabel('Actual Normalized Volume')
    plt.ylabel('Predicted Normalized Volume')
    plt.title('Actual vs. Predicted Normalized Volume')
    
    # Draw a linear regression line
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(y_val.values.reshape(-1, 1), y_pred.reshape(-1, 1))
    plt.plot(y_val, reg.predict(y_val.values.reshape(-1, 1)), 'b-', alpha=0.5)
    
    plt.savefig('model_evaluation.png')
    plt.close()
    
    print("Model evaluation completed and plot saved.")
    
    return mae, rmse


def shap_explanations(model, X_val):
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
        'macro': ['gdp_growth', 'usd_fx_rate', 'cpi_inflation'],
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
        X_cur = cur[FEATURE_COLS].to_frame().T
        # Handle missing features
        for col in FEATURE_COLS:
            if col not in X_cur:
                X_cur[col] = 0
                
        # Convert all columns to float before prediction
        for col in X_cur.columns:
            X_cur[col] = X_cur[col].astype(float)
                
        y_hat = model.predict(X_cur)[0]
        results.append(y_hat)
        
        # Shift lags & features for next step
        # In a real implementation, we'd properly update all features
        # but for simplicity in this demonstration:
        cur['lag1_vol'] = y_hat
        cur['lag3_vol'] = cur['lag1_vol'] if m >= 2 else cur['lag3_vol']
        
    return results


def run_scenarios(model, val_data):
    """
    Run multiple policy scenarios and save results
    """
    print("Running policy ablation scenarios...")
    
    scenarios = []
    
    # Get the most recent data point for each country
    countries = val_data['country'].unique()
    
    for country in countries:
        country_data = val_data[val_data['country'] == country].sort_values('date')
        start_row = country_data.iloc[-1]
        
        # Baseline: no overrides
        baseline = simulate_scenario(start_row, model)
        
        for i, vol in enumerate(baseline):
            scenarios.append({
                'country': country,
                'scenario': 'baseline',
                'month': i + 1,
                'predicted_norm_volume': vol
            })
        
        # Define policy fields that can be overridden
        policy_fields = [
            'classified_as_security', 'classified_as_commodity', 
            'classified_as_DFA', 'exchange_license_required',
            'license_intensity', 'tax_rate', 'aml_flag',
            'aml_severity', 'payment_ban'
        ]
        
        # Single-policy ablations
        for field in policy_fields:
            # Only run scenario if the field is currently active
            if start_row[field] != 0:
                scenario_name = f"no_{field}"
                results = simulate_scenario(start_row, model, override_policies={field: 0})
                
                for i, vol in enumerate(results):
                    scenarios.append({
                        'country': country,
                        'scenario': scenario_name,
                        'month': i + 1,
                        'predicted_norm_volume': vol
                    })
        
        # Double-policy ablations (example with two important policies)
        if start_row['tax_rate'] > 0 and start_row['aml_flag'] > 0:
            scenario_name = "no_tax_no_aml"
            results = simulate_scenario(
                start_row, 
                model, 
                override_policies={'tax_rate': 0, 'aml_flag': 0}
            )
            
            for i, vol in enumerate(results):
                scenarios.append({
                    'country': country,
                    'scenario': scenario_name,
                    'month': i + 1,
                    'predicted_norm_volume': vol
                })
        
        # For each country, apply other countries' full regulatory regimes
        other_countries = [c for c in countries if c != country]
        
        for other_country in other_countries:
            other_country_data = val_data[val_data['country'] == other_country].sort_values('date')
            other_country_row = other_country_data.iloc[-1]
            
            # Get all policy fields from the other country
            policy_overrides = {}
            for field in policy_fields:
                policy_overrides[field] = other_country_row[field]
            
            # Run scenario with borrowed policy regime
            scenario_name = f"{country}_with_{other_country.replace(' ', '_')}_policies"
            results = simulate_scenario(start_row, model, override_policies=policy_overrides)
            
            for i, vol in enumerate(results):
                scenarios.append({
                    'country': country,
                    'scenario': scenario_name,
                    'month': i + 1,
                    'predicted_norm_volume': vol
                })
        
        # Additional adoption scenarios
        adoption_scenarios = {
            'high_adoption': {'%_crypto_owning': min(start_row['%_crypto_owning'] * 1.5, 0.5)},
            'low_adoption': {'%_crypto_owning': max(start_row['%_crypto_owning'] * 0.5, 0.01)},
        }
        
        for scenario_name, overrides in adoption_scenarios.items():
            results = simulate_scenario(start_row, model, override_policies=overrides)
            
            for i, vol in enumerate(results):
                scenarios.append({
                    'country': country,
                    'scenario': scenario_name,
                    'month': i + 1,
                    'predicted_norm_volume': vol
                })
        
        # Extreme policy scenarios
        # 1. No regulations at all
        no_regulation_overrides = {field: 0 for field in policy_fields}
        results = simulate_scenario(start_row, model, override_policies=no_regulation_overrides)
        
        for i, vol in enumerate(results):
            scenarios.append({
                'country': country,
                'scenario': 'no_regulations',
                'month': i + 1,
                'predicted_norm_volume': vol
            })
        
        # 2. Maximum regulations
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
        results = simulate_scenario(start_row, model, override_policies=max_regulation_overrides)
        
        for i, vol in enumerate(results):
            scenarios.append({
                'country': country,
                'scenario': 'max_regulations',
                'month': i + 1,
                'predicted_norm_volume': vol
            })
    
    # Save scenarios to CSV
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df.to_csv(SCENARIO_SAVE_PATH, index=False)
    print(f"Scenarios saved to {SCENARIO_SAVE_PATH}")
    
    return scenarios_df


def visualize_scenarios(scenarios_df):
    """
    Create visualizations of scenario results
    """
    print("Visualizing scenarios...")
    
    # Create a directory for visualizations if it doesn't exist
    viz_dir = VISUALIZATION_DIR
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Group scenarios into categories for better visualization
    scenario_categories = {
        'baseline': ['baseline'],
        'single_policy_ablation': [s for s in scenarios_df['scenario'].unique() if s.startswith('no_') and '_no_' not in s],
        'multi_policy_ablation': [s for s in scenarios_df['scenario'].unique() if '_no_' in s],
        'cross_country': [s for s in scenarios_df['scenario'].unique() if 'with_' in s and 'policies' in s],
        'adoption': ['high_adoption', 'low_adoption'],
        'extreme': ['no_regulations', 'max_regulations']
    }
    
    # Line plots of predicted volumes by country and scenario category
    for country in scenarios_df['country'].unique():
        country_scenarios = scenarios_df[scenarios_df['country'] == country]
        
        # Plot baseline in all charts for reference
        baseline_data = country_scenarios[country_scenarios['scenario'] == 'baseline']
        
        # Create separate plots for each category to avoid overcrowding
        for category_name, category_scenarios in scenario_categories.items():
            # Skip categories with no matching scenarios
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
            plt.ylabel('Predicted Normalized Volume')
            plt.title(f'{category_name.replace("_", " ").title()} Scenarios: {country}')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/scenario_{category_name}_{country.replace(" ", "_")}.png')
            plt.close()
    
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
        
        # Bar chart of average impact by country
        for country in impact_df['country'].unique():
            country_impact = impact_df[impact_df['country'] == country].sort_values('avg_pct_change')
            
            # Limit to top and bottom 15 impacts if there are too many
            if len(country_impact) > 30:
                top = country_impact.nlargest(15, 'avg_pct_change')
                bottom = country_impact.nsmallest(15, 'avg_pct_change')
                country_impact = pd.concat([top, bottom])
            
            plt.figure(figsize=(16, 10))
            # Use seaborn's barplot with hue for better visualization
            sns.barplot(x='scenario', y='avg_pct_change', hue='category', data=country_impact)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('Policy Scenario')
            plt.ylabel('% Change in Volume (Average over forecast period)')
            plt.title(f'Policy Impact on Trading Volume: {country}')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/policy_impact_avg_{country.replace(" ", "_")}.png')
            plt.close()
    
    print(f"Scenario visualizations created in {viz_dir}/ directory.")


def export_results(model, model_path):
    """
    Export the trained model and results
    """
    print(f"Exporting model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model exported successfully.")


def main():
    """
    Main execution pipeline
    """
    print("=" * 80)
    print("BITCOIN ABLATION STUDY: REGULATION & ADOPTION ANALYSIS")
    print("A comparative study of the United States, Russia, and Indonesia")
    print("=" * 80)
    
    # 1. Load data
    btc_data = load_and_prep_btc_data(BTC_DATA_PATH)
    chainalysis_data = load_chainalysis_data(CHAINALYSIS_FILES)
    
    # 1.3 Load remaining data
    macro_data = load_macro_controls(MACRO_CONTROLS_PATH)
    policy_data = load_policy_panel(POLICY_PANEL_PATH)
    gdp_data = load_gdp_data(GDP_DATA_PATH)
    
    # 2. Create panel dataset
    panel = create_panel(btc_data, chainalysis_data, macro_data, policy_data, gdp_data, OUTPUT_PANEL_PATH)
    
    # 3. Feature engineering
    panel = feature_engineering(panel)
    
    # 4. Train/test split
    X_train, y_train, X_val, y_val, train_data, val_data = train_test_split(
        panel, TRAIN_END_DATE, VAL_START_DATE
    )
    
    # 5. Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 6. Evaluate model
    evaluate_model(model, X_val, y_val)
    
    # 7. SHAP explanations
    shap_explanations(model, X_val)
    
    # 8. Run policy ablation scenarios
    scenarios_df = run_scenarios(model, val_data)
    
    # 9. Visualize scenarios
    visualize_scenarios(scenarios_df)
    
    # 10. Export results
    export_results(model, MODEL_SAVE_PATH)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Generated files:")
    print(f"- Panel dataset: {OUTPUT_PANEL_PATH}")
    print(f"- Trained model: {MODEL_SAVE_PATH}")
    print(f"- Scenario results: {SCENARIO_SAVE_PATH}")
    print("- Visualizations: model_evaluation.png, visualizations/ directory, etc.")
    print("=" * 80)


if __name__ == "__main__":
    main()
