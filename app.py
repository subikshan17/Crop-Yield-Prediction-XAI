"""
Flask Backend API for Crop Yield Prediction
Loads the trained XGBoost model and preprocessing pipeline
to serve real-time crop yield predictions directly from user inputs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# ---------------------------------------------------------------------------
# Load model artifacts at startup
# ---------------------------------------------------------------------------
MODEL_DIR = 'models'
DATA_DIR = 'data'

preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
climate_features = joblib.load(os.path.join(MODEL_DIR, 'climate_features.pkl'))

# Load all available models for multi-model prediction
available_models = {}
MODEL_FILES = {
    'XGBoost Regressor': 'xgboost.pkl',
    'Random Forest': 'random_forest.pkl',
    'LightGBM': 'lightgbm.pkl',
    'Linear Regression': 'linear_regression.pkl',
    'Ridge Regression': 'ridge_regression.pkl',
}
for name, fname in MODEL_FILES.items():
    fpath = os.path.join(MODEL_DIR, fname)
    if os.path.exists(fpath):
        available_models[name] = joblib.load(fpath)

# Primary model for predictions
if 'XGBoost Regressor' in available_models:
    primary_model = available_models['XGBoost Regressor']
    primary_model_name = 'XGBoost Regressor'
elif 'Random Forest' in available_models:
    primary_model = available_models['Random Forest']
    primary_model_name = 'Random Forest'
else:
    raise FileNotFoundError('No trained model found in models/ directory.')

# Load integrated dataset for reference metadata
integrated_df = pd.read_csv(os.path.join(DATA_DIR, 'integrated_crop_climate_dataset.csv'))
available_crops = sorted(integrated_df['Crop'].unique().tolist())
available_districts = sorted(integrated_df['District_Name'].unique().tolist())
available_seasons = sorted(integrated_df['Season'].unique().tolist())

# Compute statistics per crop for the UI
crop_stats_cache = {}
for crop_name in available_crops:
    mask = integrated_df['Crop'] == crop_name
    if mask.sum() > 0:
        crop_stats_cache[crop_name] = {
            'mean': round(float(integrated_df.loc[mask, 'Yield_kg_per_ha'].mean()), 2),
            'min': round(float(integrated_df.loc[mask, 'Yield_kg_per_ha'].min()), 2),
            'max': round(float(integrated_df.loc[mask, 'Yield_kg_per_ha'].max()), 2),
            'median': round(float(integrated_df.loc[mask, 'Yield_kg_per_ha'].median()), 2),
            'count': int(mask.sum()),
        }

# Load and normalize metrics
metrics_df = pd.read_csv('results/metrics/model_comparison_table.csv')
metrics_df.columns = [c.strip() for c in metrics_df.columns]
col_map = {}
for c in metrics_df.columns:
    cl = c.lower()
    if 'model' in cl:
        col_map[c] = 'Model'
    elif 'mae' in cl:
        col_map[c] = 'MAE'
    elif 'rmse' in cl:
        col_map[c] = 'RMSE'
    elif 'mse' in cl and 'rmse' not in cl:
        col_map[c] = 'MSE'
    elif 'r2' in cl or 'r²' in cl or 'score' in cl:
        col_map[c] = 'R2'
    elif 'time' in cl:
        col_map[c] = 'Training_Time'
    elif 'mape' in cl:
        col_map[c] = 'MAPE'
metrics_df = metrics_df.rename(columns=col_map)

climate_shap_df = pd.read_csv('results/metrics/climate_shap_importance.csv')
shap_col_map = {}
for c in climate_shap_df.columns:
    cl = c.lower()
    if 'factor' in cl or 'feature' in cl:
        shap_col_map[c] = 'Feature'
    elif 'shap' in cl:
        shap_col_map[c] = 'Mean_SHAP'
climate_shap_df = climate_shap_df.rename(columns=shap_col_map)

feature_imp_df = pd.read_csv('results/metrics/feature_importance_comparison.csv')

# Compute climate means and stds for feature contribution calculations
climate_stats = {}
for feat in ['Temperature_Avg', 'Temperature_Min', 'Temperature_Max',
             'Rainfall_Precipitation', 'Relative_Humidity', 'Solar_Radiation',
             'Wind_Speed', 'Soil_Moisture']:
    if feat in integrated_df.columns:
        climate_stats[feat] = {
            'min': round(float(integrated_df[feat].min()), 1),
            'max': round(float(integrated_df[feat].max()), 1),
            'mean': round(float(integrated_df[feat].mean()), 1),
            'std': round(float(integrated_df[feat].std()), 1) or 1.0,
        }

print(f'[API] Primary model: {primary_model_name}')
print(f'[API] Available crops: {len(available_crops)}, districts: {len(available_districts)}, seasons: {len(available_seasons)}')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ['Crop', 'District_Name', 'Season']
NUMERICAL_FEATURES = ['Area', 'Temperature_Avg', 'Temperature_Min', 'Temperature_Max',
                      'Rainfall_Precipitation', 'Relative_Humidity', 'Solar_Radiation',
                      'Wind_Speed', 'Soil_Moisture']
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def build_row(data):
    """Build a DataFrame row from the incoming JSON."""
    return {
        'Crop': data['crop'],
        'District_Name': data['district'],
        'Season': data['season'],
        'Area': float(data.get('area', 5000)),
        'Temperature_Avg': float(data['temperature_avg']),
        'Temperature_Min': float(data['temperature_min']),
        'Temperature_Max': float(data['temperature_max']),
        'Rainfall_Precipitation': float(data['rainfall']),
        'Relative_Humidity': float(data['humidity']),
        'Solar_Radiation': float(data['solar_radiation']),
        'Wind_Speed': float(data['wind_speed']),
        'Soil_Moisture': float(data['soil_moisture'])
    }


def compute_input_feature_contributions(row, prediction):
    """
    Compute input-specific climate feature contributions
    based on SHAP importance weights and user input values.
    """
    shap_weights = dict(zip(climate_shap_df['Feature'], climate_shap_df['Mean_SHAP']))
    contributions = []

    for feat in ['Soil_Moisture', 'Rainfall_Precipitation', 'Temperature_Avg',
                 'Temperature_Max', 'Temperature_Min', 'Relative_Humidity',
                 'Solar_Radiation', 'Wind_Speed']:
        if feat in row and feat in climate_stats:
            val = row[feat]
            mean_val = climate_stats[feat]['mean']
            std_val = climate_stats[feat]['std']
            weight = shap_weights.get(feat, 500.0)

            # Z-score deviation from historical average
            z_score = (val - mean_val) / std_val
            # Impact estimation in kg/ha
            impact = round(z_score * (weight * 0.35), 1)

            contributions.append({
                'feature': feat,
                'user_value': val,
                'mean_value': mean_val,
                'impact_kg_ha': impact,
                'direction': 'positive' if impact >= 0 else 'negative',
                'weight': round(weight, 1)
            })

    # Sort by absolute impact descending
    contributions.sort(key=lambda x: abs(x['impact_kg_ha']), reverse=True)
    return contributions


def compute_feature_comparison(row, feat_a='Soil_Moisture', feat_b='Rainfall_Precipitation'):
    """
    Compute pairwise feature comparison and interaction effect between two climate factors.
    """
    shap_weights = dict(zip(climate_shap_df['Feature'], climate_shap_df['Mean_SHAP']))

    val_a = float(row.get(feat_a, 25.0))
    val_b = float(row.get(feat_b, 400.0))

    stats_a = climate_stats.get(feat_a, {'mean': 25.0, 'std': 1.0})
    mean_a, std_a = stats_a['mean'], stats_a['std']
    weight_a = shap_weights.get(feat_a, 500.0)
    impact_a = round(((val_a - mean_a) / std_a) * (weight_a * 0.35), 1)

    stats_b = climate_stats.get(feat_b, {'mean': 500.0, 'std': 1.0})
    mean_b, std_b = stats_b['mean'], stats_b['std']
    weight_b = shap_weights.get(feat_b, 500.0)
    impact_b = round(((val_b - mean_b) / std_b) * (weight_b * 0.35), 1)

    # Combined interaction effect
    coupled_impact = round(impact_a + impact_b + (0.12 * (impact_a * impact_b) / 1000.0), 1)
    dominant = feat_a if abs(impact_a) >= abs(impact_b) else feat_b

    if impact_a < 0 and impact_b < 0:
        stress_type = "High Dual Climate Stress"
    elif impact_a < 0 or impact_b < 0:
        stress_type = f"Moderate Stress ({dominant.replace('_', ' ')} Limiting)"
    else:
        stress_type = "Favorable Climate Synergy"

    return {
        'feature_a': feat_a,
        'feature_a_label': feat_a.replace('_', ' '),
        'feature_a_val': val_a,
        'feature_a_mean': mean_a,
        'feature_a_impact': impact_a,
        'feature_b': feat_b,
        'feature_b_label': feat_b.replace('_', ' '),
        'feature_b_val': val_b,
        'feature_b_mean': mean_b,
        'feature_b_impact': impact_b,
        'coupled_impact': coupled_impact,
        'dominant_feature': dominant.replace('_', ' '),
        'stress_type': stress_type
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Return available options for the prediction form."""
    return jsonify({
        'crops': available_crops,
        'districts': available_districts,
        'seasons': available_seasons,
        'model_name': primary_model_name,
        'available_models': list(available_models.keys()),
        'climate_features': climate_features,
        'climate_ranges': climate_stats,
        'crop_stats': crop_stats_cache,
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return climate feature metrics and SHAP importance."""
    return jsonify({
        'model_comparison': metrics_df.to_dict(orient='records'),
        'climate_shap': climate_shap_df.to_dict(orient='records'),
        'feature_importance': feature_imp_df.to_dict(orient='records')
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Accept climate + crop parameters and return predicted yield based strictly on user inputs."""
    data = request.get_json()

    required = ['crop', 'district', 'season', 'temperature_avg', 'temperature_min',
                'temperature_max', 'rainfall', 'humidity', 'solar_radiation',
                'wind_speed', 'soil_moisture']
    missing = [f for f in required if data.get(f) is None or data.get(f) == '']
    if missing:
        return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing)}'}), 400

    try:
        row = build_row(data)
        input_df = pd.DataFrame([row])
        X_input = preprocessor.transform(input_df[FEATURE_COLS])

        # Choose model
        model_choice = data.get('model', primary_model_name)
        model = available_models.get(model_choice, primary_model)
        used_name = model_choice if model_choice in available_models else primary_model_name

        prediction = float(model.predict(X_input)[0])
        # Non-negative prediction guard
        prediction = max(0.0, prediction)
        prediction_tonnes = prediction / 1000.0

        # Feature Comparison Pair Analysis
        feat_a = data.get('feature_a', 'Soil_Moisture')
        feat_b = data.get('feature_b', 'Rainfall_Precipitation')
        feature_comp_result = compute_feature_comparison(row, feat_a, feat_b)

        # Historical context for the requested crop
        crop_stats = crop_stats_cache.get(data['crop'])

        # Input-specific feature breakdown
        feature_contributions = compute_input_feature_contributions(row, prediction)

        return jsonify({
            'success': True,
            'predicted_yield_kg_per_ha': round(prediction, 2),
            'predicted_yield_tonnes_per_ha': round(prediction_tonnes, 3),
            'model_used': used_name,
            'input_summary': row,
            'crop_historical_stats': crop_stats,
            'feature_contributions': feature_contributions,
            'feature_comparison': feature_comp_result,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/dataset-stats', methods=['GET'])
def dataset_stats():
    """Return dataset overview stats."""
    return jsonify({
        'total_records': int(len(integrated_df)),
        'num_crops': len(available_crops),
        'num_districts': len(available_districts),
        'num_seasons': len(available_seasons),
        'year_range': {
            'min': int(integrated_df['Crop_Year'].min()) if 'Crop_Year' in integrated_df.columns else 1997,
            'max': int(integrated_df['Crop_Year'].max()) if 'Crop_Year' in integrated_df.columns else 2013,
        },
        'yield_stats': {
            'mean': round(float(integrated_df['Yield_kg_per_ha'].mean()), 2),
            'min': round(float(integrated_df['Yield_kg_per_ha'].min()), 2),
            'max': round(float(integrated_df['Yield_kg_per_ha'].max()), 2),
        },
        'models_count': len(available_models),
        'primary_model': primary_model_name,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
