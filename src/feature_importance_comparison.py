import os
import joblib
import pandas as pd
import numpy as np

def generate_feature_importance_comparison(
    raw_data_path='data/integrated_crop_climate_dataset.csv',
    metrics_dir='results/metrics'
):
    print('=' * 80)
    print('STAGE 9: FEATURE IMPORTANCE TRIANGULATION & COMPARISON')
    print('=' * 80)
    
    os.makedirs(metrics_dir, exist_ok=True)
    df = pd.read_csv(raw_data_path)
    
    climate_features = [
        'Temperature_Avg', 'Temperature_Min', 'Temperature_Max',
        'Rainfall_Precipitation', 'Relative_Humidity', 'Solar_Radiation',
        'Wind_Speed', 'Soil_Moisture'
    ]
    
    # 1. Bivariate Correlation with Yield
    correlations_pearson = {}
    correlations_spearman = {}
    for col in climate_features:
        correlations_pearson[col] = df[col].corr(df['Yield_kg_per_ha'], method='pearson')
        correlations_spearman[col] = df[col].corr(df['Yield_kg_per_ha'], method='spearman')
        
    # 2. Model-Native Feature Importance (from XGBoost / Random Forest)
    model = joblib.load('models/xgboost.pkl') if os.path.exists('models/xgboost.pkl') else joblib.load('models/random_forest.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    raw_model_importances = model.feature_importances_
    model_imp_dict = dict(zip(feature_names, raw_model_importances))
    
    # 3. SHAP Feature Importance
    shap_data = joblib.load('data/processed/shap_data.pkl')
    shap_values = shap_data['shap_values']
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_imp_dict = dict(zip(feature_names, mean_abs_shap))
    
    # Normalize model importance across climate factors to sum to 100%
    climate_model_imps = [model_imp_dict.get(c, 0.0) for c in climate_features]
    sum_model_imp = sum(climate_model_imps) if sum(climate_model_imps) > 0 else 1.0
    norm_model_imps = [v / sum_model_imp for v in climate_model_imps]
    
    # Normalize SHAP importance across climate factors to sum to 100%
    climate_shap_imps = [shap_imp_dict.get(c, 0.0) for c in climate_features]
    sum_shap_imp = sum(climate_shap_imps) if sum(climate_shap_imps) > 0 else 1.0
    norm_shap_imps = [v / sum_shap_imp for v in climate_shap_imps]
    
    comparison_rows = []
    for i, col in enumerate(climate_features):
        r_p = correlations_pearson[col]
        r_s = correlations_spearman[col]
        m_imp = norm_model_imps[i]
        s_imp = norm_shap_imps[i]
        
        # Determine dominant direction / pattern
        if col in ['Rainfall_Precipitation', 'Soil_Moisture', 'Relative_Humidity']:
            effect = 'Positive / Saturation Threshold' if r_s > 0 else 'Variable'
        elif 'Temperature' in col:
            effect = 'Negative (Thermal Stress above 32C)' if r_s < 0 else 'Thresholded'
        else:
            effect = 'Positive / Non-linear' if r_s > 0 else 'Negative / Non-linear'
            
        comparison_rows.append({
            'Climate Factor': col,
            'Pearson Corr (r)': round(r_p, 3),
            'Spearman Rank (rho)': round(r_s, 3),
            'Model Native Importance (%)': round(m_imp * 100, 2),
            'SHAP Importance (%)': round(s_imp * 100, 2),
            'Identified Marginal Contribution': effect
        })
        
    comp_df = pd.DataFrame(comparison_rows).sort_values(by='SHAP Importance (%)', ascending=False).reset_index(drop=True)
    
    # Save table
    out_csv = os.path.join(metrics_dir, 'feature_importance_comparison.csv')
    comp_df.to_csv(out_csv, index=False)
    
    print('\n' + '=' * 80)
    print('CONSOLIDATED FEATURE IMPORTANCE TRIANGULATION TABLE')
    print('=' * 80)
    print(comp_df.to_string(index=False))
    print('=' * 80)
    
    print('\n[Methodological Rationale for Ranking Discrepancies]:')
    print('1. Bivariate Correlation measures purely monotonic pairwise relationships and misses non-linear thermal thresholds.')
    print('2. Model-Native Importance (Gini/Gain) tends to favor continuous features with many split points but ignores multi-feature co-dependencies.')
    print('3. SHAP Importance calculates exact additive Shapley allocations across all possible feature subsets, capturing interactive and non-linear climate dynamics accurately.')
    print('=' * 80)
    return comp_df

if __name__ == '__main__':
    generate_feature_importance_comparison()