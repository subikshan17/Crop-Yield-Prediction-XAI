import os
import pandas as pd
import numpy as np

def inspect_all_datasets():
    raw_dir = 'data/raw'
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    print('=' * 80)
    print('STAGE 1: COMPREHENSIVE DATASET INSPECTION & PROFILING REPORT')
    print('=' * 80)
    
    datasets_info = {}
    
    for filename in sorted(files):
        filepath = os.path.join(raw_dir, filename)
        df = pd.read_csv(filepath)
        datasets_info[filename] = df
        
        print(f'\n--- FILENAME: {filename} ---')
        print(f'1. Shape (Rows, Columns): {df.shape[0]:,} rows, {df.shape[1]} columns')
        print(f'2. Column Names: {list(df.columns)}')
        print(f'3. Data Types:\n{df.dtypes.to_string()}')
        print(f'4. Missing Values Count & Percentage:')
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df)) * 100
        missing_df = pd.DataFrame({'Missing_Count': null_counts, 'Missing_Pct': null_pct})
        print(missing_df[missing_df['Missing_Count'] > 0].to_string() if null_counts.sum() > 0 else '   No missing values detected (0%).')
        print(f'5. Duplicate Records Count: {df.duplicated().sum():,}')
        
        # Check specific features if present
        if 'Crop' in df.columns:
            unique_crops = sorted(df['Crop'].dropna().unique())
            print(f'6. Unique Crops Count: {len(unique_crops)}')
            print(f'   Sample Crops: {unique_crops[:10]} ...')
        
        location_cols = [c for c in df.columns if any(loc in c.lower() for loc in ['district', 'state', 'location'])]
        for lcol in location_cols:
            unique_locs = sorted(df[lcol].dropna().unique())
            print(f'7. Unique Locations ({lcol}) Count: {len(unique_locs)}')
            print(f'   Sample Locations: {unique_locs[:8]} ...')
            
        year_cols = [c for c in df.columns if any(yc in c.lower() for yc in ['year', 'crop_year'])]
        for ycol in year_cols:
            years = df[ycol].dropna().unique()
            print(f'8. Year Range in {ycol}: {min(years)} to {max(years)} (Total: {len(years)} unique years)')
            
    # Check common columns between datasets
    print('\n' + '=' * 80)
    print('COMMON KEYS & SCHEMA OVERLAP ANALYSIS')
    print('=' * 80)
    file_list = list(datasets_info.keys())
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            f1, f2 = file_list[i], file_list[j]
            cols1 = set(datasets_info[f1].columns)
            cols2 = set(datasets_info[f2].columns)
            common = sorted(list(cols1.intersection(cols2)))
            print(f'* Overlap between [{f1}] and [{f2}]:')
            print(f'  Common Column Names: {common if common else "None directly identical; semantic key alignment required."}')
            
    print('\nDataset inspection completed successfully.\n')

if __name__ == '__main__':
    inspect_all_datasets()