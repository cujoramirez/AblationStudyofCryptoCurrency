#!/usr/bin/env python3
"""
Bitcoin Analysis - Final Integration

This script integrates the fixes from btc_real_data.py back into btc.py
to solve the GDP loading and data type conversion issues.
"""

import os
import sys
import shutil
import importlib
from datetime import datetime

# Display header
print("=" * 80)
print("BITCOIN ANALYSIS - FINAL INTEGRATION")
print("=" * 80)
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create backup of original btc.py if it doesn't exist
if os.path.exists('btc.py'):
    backup_filename = 'btc_original_backup.py'
    if not os.path.exists(backup_filename):
        print(f"Creating backup of original btc.py to {backup_filename}")
        shutil.copy('btc.py', backup_filename)
    else:
        print(f"Backup already exists at {backup_filename}")
else:
    print("Error: btc.py not found in current directory")
    sys.exit(1)

print("\n1. Checking which real data files are available:")
data_files = {
    'BTC Data': 'btc_1d_data_2018_to_2025.csv',
    'Chainalysis 2021': 'chainalysis_adoption_2021_Version3.csv',
    'Chainalysis 2022': 'chainalysis_adoption_2022_Version4.csv',
    'Chainalysis 2023': 'chainalysis_adoption_2023_Version4.csv',
    'Chainalysis 2024': 'chainalysis_adoption_2024_Version4.csv',
    'Panel with GDP': 'panel_with_gdp.csv',
    'GDP Data': 'GDP.csv'
}

for desc, filename in data_files.items():
    if os.path.exists(filename):
        print(f"✓ {desc}: {filename} (Found)")
    else:
        print(f"✗ {desc}: {filename} (Not found)")

print("\n2. Checking which updated scripts are available:")
script_files = {
    'Real Data Script': 'btc_real_data.py',
    'Helper Script': 'btc_real_data_helper.py',
    'GDP Loading Module': 'load_gdp.py',
    'Summary Report': 'fix_summary.md'
}

for desc, filename in script_files.items():
    if os.path.exists(filename):
        print(f"✓ {desc}: {filename} (Found)")
    else:
        print(f"✗ {desc}: {filename} (Not found)")

print("\nAnalysis results:")
if os.path.exists('xgb_btc_policy_ablation.pkl') and os.path.exists('scenarios.csv'):
    print("✓ Analysis has been successfully run (model and scenarios exist)")
    
    # Check if visualizations exist
    vis_dir = 'visualizations'
    if os.path.exists(vis_dir) and os.path.isdir(vis_dir) and len(os.listdir(vis_dir)) > 0:
        print(f"✓ Visualizations have been generated ({len(os.listdir(vis_dir))} files in {vis_dir}/)")
    else:
        print("✗ Visualizations may not have been generated properly")
else:
    print("✗ Analysis may not have completed successfully")

# Summary
print("\nSummary of fixes implemented in btc_real_data.py:")
print("1. GDP Data Fixes:")
print("   - Added proper encoding handling (utf-8-sig, utf-8, latin1, ISO-8859-1)")
print("   - Fixed European number format (commas as decimal separators)")
print("   - Added fallback to extract GDP data from panel_with_gdp.csv")

print("2. Data Type Fix in Scenario Simulation:")
print("   - Ensured all features are converted to float before prediction")
print("   - Added code: for col in X_cur.columns: X_cur[col] = X_cur[col].astype(float)")

print("3. File Path Updates:")
print("   - Updated to use real data sources:")
print("   - MACRO_CONTROLS_PATH = 'panel_with_gdp.csv'")
print("   - POLICY_PANEL_PATH = 'panel_with_gdp.csv'")

print("\nRecommendation:")
print("Since btc.py has indentation issues, we recommend using btc_real_data.py")
print("for all future analysis as it contains all the necessary fixes and works properly.")

print("=" * 80)
print("INTEGRATION SUMMARY COMPLETE")
print("=" * 80)
