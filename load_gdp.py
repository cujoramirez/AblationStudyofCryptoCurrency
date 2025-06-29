"""
Helper module to load GDP data with proper encoding handling
"""
import pandas as pd
import os

def load_gdp_data(file_path='GDP.csv'):
    """
    Load GDP data from IMF dataset handling European number format
    
    Args:
        file_path: Path to the GDP.csv file
        
    Returns:
        Pandas DataFrame with GDP data in long format
    """
    print(f"Loading GDP data from {file_path}...")
    
    # Try different encodings
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'ISO-8859-1']
    
    gdp_data = None
    error_msgs = []
    
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
            
            # Convert GDP values to numeric, handling European number format (comma as decimal separator)
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
            error_msgs.append(f"Failed with {encoding} encoding: {str(e)}")
            continue
    
    # If we get here, none of the encodings worked
    print("Could not load GDP data with any encoding.")
    for msg in error_msgs:
        print(f"  {msg}")
    
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
    
    print("Failed to load GDP data from any source. Generating mock data...")
    return None


if __name__ == "__main__":
    # Test the function
    gdp_data = load_gdp_data()
    if gdp_data is not None:
        print(gdp_data.head())
        print(f"Shape: {gdp_data.shape}")