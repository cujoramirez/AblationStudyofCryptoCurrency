#!/usr/bin/env python3
"""
Bitcoin Regulation Analysis - CLI Interface
This script provides a command-line interface to run the Bitcoin policy ablation study.
It includes options to run specific parts of the analysis or the complete pipeline.
"""

import argparse
import time
import sys

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Constants
COUNTRIES = ['United States', 'Russia', 'Indonesia']
FEATURE_COLS = [
    'lag1_vol', 'lag3_vol', 'volatility_3m', 'vol_pct_change_1m',
    'classified_as_security', 'classified_as_commodity', 'classified_as_DFA',
    'exchange_license_required', 'license_intensity',
    'tax_rate', 'aml_flag', 'aml_severity',
    'payment_ban', '%_crypto_owning',
    'gdp_growth', 'usd_fx_rate', 'cpi_inflation'
]

def create_mock_data():
    """Create simplified mock data for demonstration"""
    print("Creating mock data for demonstration...")
    
    # Create dates
    dates = pd.date_range(start='2018-01-01', end='2025-04-30', freq='M')
    
    # Create Bitcoin price data (simple trend with some noise)
    btc_prices = []
    base_price = 10000
    for i, date in enumerate(dates):
        # Add upward trend and seasonal component with some randomness
        price = base_price + i * 100 + 2000 * np.sin(i/12) + np.random.normal(0, 1000)
        volume = 100000 + 10000 * np.sin(i/6) + np.random.normal(0, 5000)
        
        btc_prices.append({
            'date': date,
            'price': max(price, 1000),  # Keep prices positive
            'volume': max(volume, 10000)  # Keep volume positive
        })
    
    btc_df = pd.DataFrame(btc_prices)
    
    # Expand to countries
    data_rows = []
    for country in COUNTRIES:
        for i, row in btc_df.iterrows():
            date = row['date']
            year = date.year
            price = row['price']
            
            # Create different policy regimes by year
            if country == 'United States':
                policy_data = {
                    'classified_as_security': 1 if year >= 2021 else 0,
                    'classified_as_commodity': 1,
                    'classified_as_DFA': 0,
                    'exchange_license_required': 1 if year >= 2020 else 0,
                    'license_intensity': 0.7 if year >= 2020 else 0,
                    'tax_rate': 0.15 if year >= 2022 else 0.1,
                    'aml_flag': 1,
                    'aml_severity': 0.8,
                    'payment_ban': 0
                }
                volume_multiplier = 1.0
            elif country == 'Russia':
                policy_data = {
                    'classified_as_security': 0,
                    'classified_as_commodity': 1 if year >= 2020 else 0,
                    'classified_as_DFA': 1 if year >= 2021 else 0,
                    'exchange_license_required': 0,
                    'license_intensity': 0.0,
                    'tax_rate': 0.13,
                    'aml_flag': 1 if year >= 2022 else 0,
                    'aml_severity': 0.4 if year >= 2022 else 0,
                    'payment_ban': 1 if year >= 2022 else 0
                }
                volume_multiplier = 0.5  # Lower relative volume
            else:  # Indonesia
                policy_data = {
                    'classified_as_security': 0,
                    'classified_as_commodity': 1,
                    'classified_as_DFA': 0,
                    'exchange_license_required': 1 if year >= 2021 else 0,
                    'license_intensity': 0.6 if year >= 2021 else 0,
                    'tax_rate': 0.22,
                    'aml_flag': 1,
                    'aml_severity': 0.5,
                    'payment_ban': 1 if 2020 <= year <= 2022 else 0  # Temporary ban
                }
                volume_multiplier = 0.3  # Lower relative volume
            
            # Economic indicators
            gdp_growth = 0.02 + 0.01 * np.sin(i/12) + np.random.normal(0, 0.005)  # ~2% with seasonality
            inflation = 0.03 + 0.02 * (year >= 2022 and year <= 2023) + np.random.normal(0, 0.003)  # Higher in 2022-2023
            
            # Effect of policies on volume (simple model)
            # Payment ban significantly reduces volume
            if policy_data['payment_ban'] > 0:
                volume_multiplier *= 0.5
                
            # Higher taxes slightly reduce volume    
            volume_multiplier *= (1 - policy_data['tax_rate'] * 0.3)
            
            # AML requirements slightly reduce volume
            if policy_data['aml_flag'] > 0:
                volume_multiplier *= (1 - policy_data['aml_severity'] * 0.2)
                
            # Add crypto ownership rate (~5-20%)
            crypto_ownership = 0.05 + 0.15 * np.random.random()
            
            # Final volume with policy effects
            final_volume = row['volume'] * volume_multiplier
            
            # Create data row
            data_row = {
                'date': date,
                'country': country,
                'price': price,
                'volume': final_volume,
                'gdp_growth': gdp_growth,
                'usd_fx_rate': 1.0 if country == 'United States' else (70 if country == 'Russia' else 14000),
                'cpi_inflation': inflation,
                '%_crypto_owning': crypto_ownership,
            }
            
            # Add policy fields
            data_row.update(policy_data)
            data_rows.append(data_row)
    
    # Create final dataframe
    panel = pd.DataFrame(data_rows)
    
    # Add normalized volume (relative to a country's scale)
    for country in COUNTRIES:
        mask = panel['country'] == country
        panel.loc[mask, 'norm_volume'] = panel.loc[mask, 'volume'] / panel.loc[mask, 'volume'].mean()
    
    # Create lag features
    panel = panel.sort_values(['country', 'date'])
    panel['lag1_vol'] = panel.groupby('country')['norm_volume'].shift(1)
    panel['lag3_vol'] = panel.groupby('country')['norm_volume'].shift(3)
    panel['volatility_3m'] = panel.groupby('country')['price'].rolling(3).std().reset_index(level=0, drop=True)
    panel['vol_pct_change_1m'] = panel.groupby('country')['norm_volume'].pct_change(1)
    
    # Create target: next month's normalized volume
    panel['target'] = panel.groupby('country')['norm_volume'].shift(-1)
    
    # Fill missing values
    for col in ['lag1_vol', 'lag3_vol', 'volatility_3m', 'vol_pct_change_1m']:
        panel[col] = panel[col].fillna(0)
    
    # Drop rows with missing target (can't predict these)
    panel = panel.dropna(subset=['target'])
    
    print(f"Created panel data with {len(panel)} rows")
    
    # Save to CSV for inspection
    panel.to_csv('mock_panel.csv', index=False)
    return panel


def train_and_evaluate():
    """Train a model and evaluate results"""
    # Create data
    panel = create_mock_data()
    
    # Split into train/test
    train_data = panel[panel['date'] < '2023-01-01'].copy()
    val_data = panel[panel['date'] >= '2023-01-01'].copy()
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Extract features and target
    X_train = train_data[FEATURE_COLS]
    y_train = train_data['target']
    X_val = val_data[FEATURE_COLS]
    y_val = val_data['target']
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,  # Reduced for speed
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Simple fit without early stopping
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Plot results by country
    plt.figure(figsize=(15, 10))
    
    for i, country in enumerate(COUNTRIES):
        plt.subplot(len(COUNTRIES), 1, i+1)
        
        country_data = val_data[val_data['country'] == country]
        country_preds = model.predict(country_data[FEATURE_COLS])
        
        plt.plot(country_data['date'], country_data['target'], 'o-', label='Actual')
        plt.plot(country_data['date'], country_preds, 's--', label='Predicted')
        
        plt.title(f'{country}: Actual vs Predicted Normalized Volume')
        plt.ylabel('Norm. Volume')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('bitcoin_predictions.png')
    print("Saved prediction plot to bitcoin_predictions.png")
    
    # Run a simple policy scenario
    print("\nRunning policy scenarios...")
    
    # Get the latest data points
    latest_data = {}
    for country in COUNTRIES:
        country_data = val_data[val_data['country'] == country].sort_values('date')
        if len(country_data) > 0:
            latest_data[country] = country_data.iloc[-1]
    
    # Define scenarios
    scenarios = {
        'baseline': {},
        'no_payment_ban': {'payment_ban': 0},
        'high_tax': {'tax_rate': 0.30},
        'strict_aml': {'aml_severity': 1.0},
        'no_regulation': {'payment_ban': 0, 'tax_rate': 0.05, 'aml_severity': 0.1}
    }
    
    # Run scenarios and plot
    plt.figure(figsize=(15, 10))
    
    for i, country in enumerate(COUNTRIES):
        plt.subplot(len(COUNTRIES), 1, i+1)
        
        start_row = latest_data[country]
        
        for scenario, changes in scenarios.items():
            # Create a copy for this scenario
            scenario_data = start_row.copy()
            
            # Apply changes
            for k, v in changes.items():
                scenario_data[k] = v
              # Predict - ensure numeric values by converting to float
            pred_features = scenario_data[FEATURE_COLS].to_frame().T.astype(float)
            pred_volume = model.predict(pred_features)[0]
            
            # Plot as bar
            plt.bar(scenario, pred_volume, alpha=0.7)
        
        plt.title(f'{country}: Policy Scenario Impact')
        plt.ylabel('Predicted Norm. Volume')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('bitcoin_policy_scenarios.png')
    print("Saved policy scenario plot to bitcoin_policy_scenarios.png")
    
    return model


if __name__ == "__main__":
    print("===== BITCOIN REGULATION POLICY ANALYSIS =====")
    print("A comparative study of the United States, Russia, and Indonesia")
    print("================================================\n")
    
    model = train_and_evaluate()
    
    print("\n===== ANALYSIS COMPLETE =====")
    print("Outputs:")
    print("- mock_panel.csv: Generated panel dataset")
    print("- bitcoin_predictions.png: Model prediction plots")
    print("- bitcoin_policy_scenarios.png: Policy scenario comparison")
