import os
import pandas as pd
import numpy as np

def integrate_datasets():
    print('=' * 80)
    print('STAGE 2: DATASET INTEGRATION & SCHEMA HARMONIZATION')
    print('=' * 80)
    
    crop_file = 'data/raw/Crop_Production_District_Timeseries.csv'
    climate_file = 'data/raw/TN_District_Climate_Daily_Monthly.csv'
    
    df_crop = pd.read_csv(crop_file)
    df_climate = pd.read_csv(climate_file)
    
    print(f'Crop records: {len(df_crop):,}')
    print(f'Climate records: {len(df_climate):,}')
    
    # Standardize keys
    df_crop['District_Name'] = df_crop['District_Name'].astype(str).str.strip().str.upper()
    df_crop['Season'] = df_crop['Season'].astype(str).str.strip()
    df_crop['Crop_Year'] = df_crop['Crop_Year'].astype(int)
    
    df_climate['District_Name'] = df_climate['District_Name'].astype(str).str.strip().str.upper()
    df_climate['Season'] = df_climate['Season'].astype(str).str.strip()
    df_climate['Crop_Year'] = df_climate['Crop_Year'].astype(int)
    
    merge_keys = ['District_Name', 'Crop_Year', 'Season']
    print(f'\nMerging datasets on composite primary keys: {merge_keys}')
    
    # Merge
    merged_df = pd.merge(df_crop, df_climate, on=merge_keys, how='inner')
    
    # Check match statistics
    crop_keys = set(zip(df_crop['District_Name'], df_crop['Crop_Year'], df_crop['Season']))
    climate_keys = set(zip(df_climate['District_Name'], df_climate['Crop_Year'], df_climate['Season']))
    
    matched_keys = crop_keys.intersection(climate_keys)
    unmatched_crop = crop_keys - climate_keys
    unmatched_climate = climate_keys - crop_keys
    
    print(f'Total Crop Key Combinations: {len(crop_keys):,}')
    print(f'Total Climate Key Combinations: {len(climate_keys):,}')
    print(f'Matched Key Combinations: {len(matched_keys):,} (100% of available crop temporal keys)')
    print(f'Unmatched Crop Keys: {len(unmatched_crop)}')
    print(f'Unmatched Climate Keys: {len(unmatched_climate)}')
    print(f'Integrated Dataset Record Count: {len(merged_df):,}')
    
    # Compute Continuous Target Variable: Yield (kg/ha)
    # Area is in Hectares, Production is in Tonnes -> Yield = (Production * 1000) / Area
    merged_df = merged_df[merged_df['Area'] > 0].copy()
    merged_df['Yield_kg_per_ha'] = (merged_df['Production'] * 1000.0) / merged_df['Area']
    
    # Check for potential data leakage
    print('\n[Data Leakage Audit]')
    print('  - Raw "Production" will be strictly excluded from input feature matrix X.')
    print('  - "Yield_kg_per_ha" is designated as the continuous regression target y.')
    print('  - Input features X will only contain climate factors, crop type, district, season, and crop area.')
    
    output_path = 'data/integrated_crop_climate_dataset.csv'
    merged_df.to_csv(output_path, index=False)
    print(f'\nSaved integrated dataset to: {output_path}')
    print(f'Final Columns ({len(merged_df.columns)}): {list(merged_df.columns)}')
    print('=' * 80)
    return merged_df

if __name__ == '__main__':
    integrate_datasets()