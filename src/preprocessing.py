import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(input_file='data/integrated_crop_climate_dataset.csv', output_dir='data/processed'):
    print('=' * 80)
    print('STAGE 3: DATA PREPROCESSING & TIME-AWARE SPLITTING')
    print('=' * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    df = pd.read_csv(input_file)
    print(f'Initial records: {len(df):,}')
    
    # 1. Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates().copy()
    print(f'Duplicates removed: {initial_len - len(df)}')
    
    # 2. Check for impossible/invalid values and zero areas
    df = df[(df['Area'] > 0) & (df['Yield_kg_per_ha'] > 0)].copy()
    
    # 3. Outlier handling: Winsorize extreme 99.9th percentile values
    q999 = df['Yield_kg_per_ha'].quantile(0.999)
    df['Yield_kg_per_ha'] = df['Yield_kg_per_ha'].clip(upper=q999)
    print(f'Target Yield (kg/ha) range: {df["Yield_kg_per_ha"].min():.2f} to {df["Yield_kg_per_ha"].max():.2f}')
    
    # 4. Feature Selection
    climate_features = [
        'Temperature_Avg', 'Temperature_Min', 'Temperature_Max',
        'Rainfall_Precipitation', 'Relative_Humidity', 'Solar_Radiation',
        'Wind_Speed', 'Soil_Moisture'
    ]
    categorical_features = ['Crop', 'District_Name', 'Season']
    numerical_features = ['Area'] + climate_features
    
    feature_cols = categorical_features + numerical_features
    target_col = 'Yield_kg_per_ha'
    
    print(f'\n[Feature Matrix Definition]')
    print(f'  - Climate Features ({len(climate_features)}): {climate_features}')
    print(f'  - Categorical Features ({len(categorical_features)}): {categorical_features}')
    print(f'  - Numerical Features ({len(numerical_features)}): {numerical_features}')
    print(f'  - Target (y): {target_col}')
    print(f'  - Leakage Guard: Raw "Production" is excluded from X.')
    
    # 5. Time-Aware Train / Validation / Test Split
    print(f'\n[Time-Aware Temporal Splitting]')
    print('  - Training Set: Years 1997 - 2008 (Chronological Past)')
    print('  - Validation Set: Years 2009 - 2010 (Model Tuning & Early Stopping)')
    print('  - Test Set: Years 2011 - 2013 (Unseen Recent Evaluation)')
    
    train_df = df[df['Crop_Year'] <= 2008].copy()
    val_df = df[(df['Crop_Year'] >= 2009) & (df['Crop_Year'] <= 2010)].copy()
    test_df = df[df['Crop_Year'] >= 2011].copy()
    
    print(f'  - Train observations: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)')
    print(f'  - Validation observations: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)')
    print(f'  - Test observations: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)')
    
    # 6. Preprocessor / ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Fit preprocessor strictly on training data
    X_train_raw = train_df[feature_cols]
    y_train = train_df[target_col].values
    
    X_val_raw = val_df[feature_cols]
    y_val = val_df[target_col].values
    
    X_test_raw = test_df[feature_cols]
    y_test = test_df[target_col].values
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    transformed_feature_names = numerical_features + cat_names
    
    print(f'\nTransformed Feature Count: {X_train.shape[1]} columns')
    
    # Save processed arrays and metadata
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save DataFrames for SHAP feature naming and interpretable plots
    train_df.to_csv(os.path.join(output_dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_df.csv'), index=False)
    
    # Save preprocessor and feature names
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(transformed_feature_names, 'models/feature_names.pkl')
    joblib.dump(climate_features, 'models/climate_features.pkl')
    
    print('Preprocessing objects and datasets saved successfully.')
    print('=' * 80)

if __name__ == '__main__':
    preprocess_data()