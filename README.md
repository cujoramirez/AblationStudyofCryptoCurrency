# Bitcoin Policy Ablation Study

## Overview

This project implements a sophisticated machine learning analysis of Bitcoin trading volumes and adoption across three countries: the United States, Russia, and Indonesia. The study examines how different cryptocurrency regulatory policies impact trading volume and adoption rates from 2018 to 2025.

## Key Features

- **End-to-End ML Pipeline**: Data loading, preprocessing, feature engineering, model training, and evaluation
- **Policy Ablation Analysis**: Systematic analysis of how removing specific regulations affects crypto markets
- **Cross-Country Comparisons**: Analysis of how each country's regulatory approach affects outcomes
- **Adoption Metrics Integration**: Incorporates Chainalysis crypto adoption data from 2021-2024
- **Future Projections**: Forecasts volume through 2025 under different policy scenarios
- **Rich Visualizations**: Comprehensive visualization of results with SHAP model explanations

## Implementation Notes

The project includes two main implementation files:

1. **btc.py**: Original implementation (has some structure issues)
2. **btc_real_data.py**: Enhanced implementation with the following improvements:
   - Proper handling of GDP.csv with European number format (commas as decimal separators)
   - Multiple encoding support for data files
   - Robust fallback mechanisms for data sources
   - Fixed data type conversion in scenario simulation
   - Enhanced feature engineering with real GDP values

## Data Sources

The analysis incorporates several data sources:
- Bitcoin OHLCV daily data (2018-2025)
- Chainalysis crypto adoption metrics (2021-2024)
- Macroeconomic indicators for each country
- Detailed cryptocurrency policy panel data

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Full Analysis

```bash
python run_btc_analysis.py --full
```

### Running Specific Components

```bash
# Data preparation only
python run_btc_analysis.py --data

# Model training only
python run_btc_analysis.py --train

# Generate policy scenarios
python run_btc_analysis.py --scenarios

# Create visualizations
python run_btc_analysis.py --visualize

# Run a custom policy scenario
python run_btc_analysis.py --custom '{"tax_rate": 0, "payment_ban": 1}'
```

## Key Outputs

The analysis generates several outputs:
- `panel.csv`: The complete panel dataset used for analysis
- `scenarios.csv`: Results of different policy ablation scenarios
- `xgb_btc_policy_ablation.pkl`: The trained XGBoost model
- `visualizations/`: Directory containing all generated visualizations
- `shap_analysis/`: Directory containing model interpretability plots

## Policy Factors Analyzed

The study examines the impact of various regulatory factors:
- Security/Commodity/DFA classifications
- Exchange licensing requirements
- Tax policies
- Anti-Money Laundering (AML) regulations
- Payment bans
- Crypto adoption rates

## Methodology

1. **Data Integration**: Merges multiple data sources into a unified panel dataset
2. **Feature Engineering**: Creates lag features, volatility metrics, and normalized volumes
3. **Model Training**: Uses XGBoost to predict normalized trading volumes
4. **Policy Ablation**: Systematically removes or modifies policies to measure their impact
5. **Cross-Country Analysis**: Examines how different regulatory regimes compare
6. **Scenario Generation**: Projects outcomes under various policy configurations
7. **Visualization & Interpretation**: Creates comprehensive visualizations of results

## License

MIT License
