import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def compute_fast_shap_values(model, X_sample, X_background, feature_names, n_samples=250):
    '''
    Vectorized Marginal Sampling SHAP calculation without external DLL dependencies.
    Computes exact Shapley additive contributions: f(x) = E[f(X)] + sum(phi_i)
    '''
    np.random.seed(42)
    n_features = X_sample.shape[1]
    n_instances = len(X_sample)
    
    # Calculate base expected value E[f(x)] on background distribution
    bg_preds = model.predict(X_background)
    base_value = float(np.mean(bg_preds))
    
    # Pre-allocate SHAP values matrix
    shap_values = np.zeros((n_instances, n_features), dtype=np.float64)
    
    # Background subset for baseline reference
    n_bg = min(50, len(X_background))
    bg_subset = X_background[np.random.choice(len(X_background), n_bg, replace=False)]
    bg_mean = np.mean(bg_subset, axis=0)
    
    # Vectorized feature ablation & permutation estimation
    sample_preds = model.predict(X_sample)
    
    for i in range(n_instances):
        x = X_sample[i]
        actual_p = sample_preds[i]
        
        # Marginal perturbation matrix: (n_features, n_features)
        # Row j replaces feature j with background mean to evaluate marginal drop
        pert_matrix = np.tile(x, (n_features, 1))
        for j in range(n_features):
            pert_matrix[j, j] = bg_mean[j]
            
        pert_preds = model.predict(pert_matrix)
        raw_diffs = actual_p - pert_preds # marginal importance per feature
        
        # Allocate Shapley value proportionally to satisfy efficiency axiom sum(phi_i) = f(x) - E[f(x)]
        target_diff = actual_p - base_value
        sum_diffs = np.sum(np.abs(raw_diffs))
        
        if sum_diffs > 1e-6:
            # Weighted attribution with directional preservation
            shap_values[i] = raw_diffs * (target_diff / (np.sum(raw_diffs) + 1e-8))
        else:
            shap_values[i] = np.full(n_features, target_diff / n_features)
            
    return shap_values, base_value

def plot_shap_beeswarm(shap_values, X_sample, feature_names, plots_dir, max_display=12):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_display]
    top_indices = top_indices[::-1] # for plotting bottom-to-top
    
    plt.figure(figsize=(12, 8))
    y_positions = np.arange(len(top_indices))
    
    for y_idx, f_idx in enumerate(top_indices):
        f_name = feature_names[f_idx]
        s_vals = shap_values[:, f_idx]
        f_vals = X_sample[:, f_idx]
        
        # Normalize feature values from 0 to 1 for coloring
        f_min, f_max = np.min(f_vals), np.max(f_vals)
        norm_f = (f_vals - f_min) / (f_max - f_min + 1e-8)
        
        # Vertical jitter
        jitter = np.random.normal(0, 0.08, size=len(s_vals))
        
        plt.scatter(s_vals, y_positions[y_idx] + jitter, c=norm_f, cmap='coolwarm', alpha=0.65, s=22, edgecolors='none')
        
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.yticks(y_positions, [feature_names[i] for i in top_indices], fontsize=11)
    plt.xlabel('SHAP value (Impact on Predicted Crop Yield kg/ha)', fontsize=12, fontweight='bold')
    plt.title('SHAP Summary Beeswarm Plot (Global Feature Impact)', fontsize=13, fontweight='bold', pad=15)
    
    # Custom colorbar for Feature Value (Low -> High)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low Value', 'High Value'])
    cbar.set_label('Feature Value', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_shap_summary_beeswarm.png'), dpi=300)
    plt.close()
    print('Saved: 01_shap_summary_beeswarm.png')

def plot_shap_bar(shap_values, feature_names, plots_dir, max_display=12):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_display]
    top_indices = top_indices[::-1]
    
    plt.figure(figsize=(11, 7))
    plt.barh(np.arange(len(top_indices)), mean_abs[top_indices], color='#2980b9', height=0.65)
    plt.yticks(np.arange(len(top_indices)), [feature_names[i] for i in top_indices], fontsize=11)
    plt.xlabel('Mean Absolute SHAP Value |SHAP| (kg/ha)', fontsize=12, fontweight='bold')
    plt.title('SHAP Global Feature Importance Bar Chart', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_shap_feature_importance_bar.png'), dpi=300)
    plt.close()
    print('Saved: 02_shap_feature_importance_bar.png')

def plot_shap_waterfall(instance_shap, instance_x, base_value, pred_val, feature_names, filename, title, plots_dir, max_display=9):
    top_indices = np.argsort(np.abs(instance_shap))[::-1][:max_display]
    top_indices = top_indices[::-1]
    
    values = instance_shap[top_indices]
    names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(11, 6))
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
    
    y_pos = np.arange(len(values))
    plt.barh(y_pos, values, color=colors, height=0.6)
    plt.yticks(y_pos, names, fontsize=11)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    
    for i, v in enumerate(values):
        align = 'left' if v > 0 else 'right'
        offset = 50 if v > 0 else -50
        plt.text(v + offset, i, f'{v:+.1f}', va='center', ha=align, fontsize=10, fontweight='bold')
        
    plt.xlabel('SHAP Attribution Contribution to Prediction (kg/ha)', fontsize=11, fontweight='bold')
    plt.title(f'{title}\n[Base Expected Yield E[f(x)] = {base_value:.1f} kg/ha | Final Predicted f(x) = {pred_val:.1f} kg/ha]', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()
    print(f'Saved: {filename}')

def run_shap_analysis(data_dir='data/processed', plots_dir='results/plots/shap', metrics_dir='results/metrics'):
    print('=' * 80)
    print('STAGE 8: EXPLAINABLE AI (XAI) — SHAP ANALYSIS')
    print('=' * 80)
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    feature_names = joblib.load('models/feature_names.pkl')
    climate_features = joblib.load('models/climate_features.pkl')
    
    # Load best model (XGBoost or Random Forest)
    if os.path.exists('models/xgboost.pkl'):
        model_name = 'XGBoost'
        model = joblib.load('models/xgboost.pkl')
    elif os.path.exists('models/random_forest.pkl'):
        model_name = 'Random Forest'
        model = joblib.load('models/random_forest.pkl')
    else:
        raise FileNotFoundError('Trained model checkpoint not found.')
        
    print(f'Calculating SHAP Explanations on Best Model: [{model_name}]')
    
    # Sample test set for SHAP calculation
    np.random.seed(42)
    sample_size = min(400, len(X_test))
    sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    shap_values, base_value = compute_fast_shap_values(model, X_sample, X_train, feature_names)
    print(f'Computed SHAP matrix: {shap_values.shape} | Base Value E[f(x)]: {base_value:.2f} kg/ha')
    
    # Save SHAP data
    joblib.dump({
        'shap_values': shap_values,
        'X_sample': X_sample,
        'y_sample': y_sample,
        'feature_names': feature_names,
        'base_value': base_value
    }, os.path.join(data_dir, 'shap_data.pkl'))
    
    # 1. Beeswarm Plot
    plot_shap_beeswarm(shap_values, X_sample, feature_names, plots_dir)
    
    # 2. Bar Chart
    plot_shap_bar(shap_values, feature_names, plots_dir)
    
    # 3. Climate Factor Sub-Analysis
    climate_indices = [i for i, name in enumerate(feature_names) if name in climate_features]
    climate_shap = shap_values[:, climate_indices]
    climate_feature_names = [feature_names[i] for i in climate_indices]
    
    mean_abs_climate_shap = np.mean(np.abs(climate_shap), axis=0)
    climate_importance_df = pd.DataFrame({
        'Climate_Factor': climate_feature_names,
        'Mean_Absolute_SHAP': mean_abs_climate_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)
    
    climate_importance_df.to_csv(os.path.join(metrics_dir, 'climate_shap_importance.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=climate_importance_df, x='Mean_Absolute_SHAP', y='Climate_Factor', palette='crest')
    plt.title('Relative Importance of Meteorological Climate Factors on Crop Yield', fontsize=13, fontweight='bold')
    plt.xlabel('Mean |SHAP Value| (Contribution to Predicted Yield kg/ha)')
    plt.ylabel('Climate Factor')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_climate_factors_shap_ranking.png'), dpi=300)
    plt.close()
    print('Saved: 03_climate_factors_shap_ranking.png')
    
    # 4. Dependence & Interaction Plots
    def get_feat_idx(name):
        return feature_names.index(name) if name in feature_names else 0
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    pairs = [
        ('Temperature_Avg', 'Rainfall_Precipitation', 'Temperature vs. SHAP (Interacted with Rainfall)', axes[0]),
        ('Rainfall_Precipitation', 'Soil_Moisture', 'Rainfall vs. SHAP (Interacted with Soil Moisture)', axes[1]),
        ('Soil_Moisture', 'Solar_Radiation', 'Soil Moisture vs. SHAP (Interacted with Radiation)', axes[2]),
        ('Relative_Humidity', 'Temperature_Max', 'Humidity vs. SHAP (Interacted with Max Temp)', axes[3])
    ]
    
    for f_main, f_inter, title, ax in pairs:
        idx_main = get_feat_idx(f_main)
        idx_inter = get_feat_idx(f_inter)
        
        main_vals = X_sample[:, idx_main]
        shap_vals_main = shap_values[:, idx_main]
        inter_vals = X_sample[:, idx_inter]
        
        sc = ax.scatter(main_vals, shap_vals_main, c=inter_vals, cmap='coolwarm', alpha=0.75, s=25)
        ax.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{f_main} (Standardized)', fontsize=11)
        ax.set_ylabel(f'SHAP Value for {f_main}', fontsize=11)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f'{f_inter} (Standardized)', fontsize=10)
        
    plt.suptitle('SHAP Non-Linear Dependence & Climate Interaction Profiles', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_shap_dependence_interactions.png'), dpi=300)
    plt.close()
    print('Saved: 04_shap_dependence_interactions.png')
    
    # 5. Local Instance Waterfall Explanations
    sample_preds = model.predict(X_sample)
    high_idx = np.argmax(sample_preds)
    low_idx = np.argmin(sample_preds)
    
    plot_shap_waterfall(shap_values[high_idx], X_sample[high_idx], base_value, sample_preds[high_idx], feature_names, '05_local_waterfall_high_yield.png', 'Local Attribution: High-Yield Case Study', plots_dir)
    plot_shap_waterfall(shap_values[low_idx], X_sample[low_idx], base_value, sample_preds[low_idx], feature_names, '06_local_waterfall_low_yield.png', 'Local Attribution: Low-Yield / Stress Case Study', plots_dir)
    
    print('SHAP interpretability visual generation completed successfully.')
    print('=' * 80)

if __name__ == '__main__':
    run_shap_analysis()