import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(data_dir='data/processed', metrics_dir='results/metrics', plots_dir='results/plots/models'):
    print('=' * 80)
    print('STAGE 7: MODEL EVALUATION & PERFORMANCE VISUALIZATION')
    print('=' * 80)
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    predictions_test = joblib.load(os.path.join(data_dir, 'predictions_test.pkl'))
    predictions_val = joblib.load(os.path.join(data_dir, 'predictions_val.pkl'))
    training_times = joblib.load(os.path.join('models', 'training_times.pkl'))
    
    metrics_list = []
    
    # 1. Compute Regression Metrics
    for name in predictions_test.keys():
        preds = predictions_test[name]
        
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        train_time = training_times.get(name, 0.0)
        
        metrics_list.append({
            'Model': name,
            'MAE (kg/ha)': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE (kg/ha)': round(rmse, 2),
            'R² Score': round(r2, 4),
            'Training Time (s)': round(train_time, 2)
        })
        
    metrics_df = pd.DataFrame(metrics_list).sort_values(by='R² Score', ascending=False).reset_index(drop=True)
    
    # Save comparison table
    table_path = os.path.join(metrics_dir, 'model_comparison_table.csv')
    metrics_df.to_csv(table_path, index=False)
    
    print('\n' + '=' * 80)
    print('FINAL MODEL PERFORMANCE BENCHMARK COMPARISON TABLE')
    print('=' * 80)
    print(metrics_df.to_string(index=False))
    print('=' * 80)
    
    best_model_name = metrics_df.iloc[0]['Model']
    print(f'\n* Best Performing Architecture: [{best_model_name}] with Test R² = {metrics_df.iloc[0]["R² Score"]:.4f} and RMSE = {metrics_df.iloc[0]["RMSE (kg/ha)"]:.2f} kg/ha')
    
    # 2. Performance Visualizations
    sns.set_theme(style='whitegrid', palette='muted')
    
    # A. Model Comparison Bar Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(data=metrics_df, x='Model', y='R² Score', ax=axes[0], palette='crest')
    axes[0].set_title('Test R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
    axes[0].set_ylim(0, 1.0)
    
    sns.barplot(data=metrics_df, x='Model', y='RMSE (kg/ha)', ax=axes[1], palette='rocket')
    axes[1].set_title('Test RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
    
    sns.barplot(data=metrics_df, x='Model', y='MAE (kg/ha)', ax=axes[2], palette='mako')
    axes[2].set_title('Test MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_model_metrics_comparison.png'), dpi=300)
    plt.close()
    print('Saved: 01_model_metrics_comparison.png')
    
    # B. Actual vs Predicted Scatter Plots (for top models)
    top_models = metrics_df['Model'].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(top_models):
        preds = predictions_test[name]
        ax = axes[i]
        
        ax.scatter(y_test, preds, alpha=0.35, color='#2980b9', s=18)
        # Identity y = x line
        min_v = min(y_test.min(), preds.min())
        max_v = max(y_test.max(), preds.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal Fit (y = x)')
        
        r2_val = metrics_df[metrics_df['Model'] == name]['R² Score'].values[0]
        rmse_val = metrics_df[metrics_df['Model'] == name]['RMSE (kg/ha)'].values[0]
        ax.set_title(f'{name} (R² = {r2_val:.4f}, RMSE = {rmse_val:.1f} kg/ha)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Yield (kg/ha)')
        ax.set_ylabel('Predicted Yield (kg/ha)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        
    plt.suptitle('Actual vs. Predicted Crop Yield (Test Set 2011-2013)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_actual_vs_predicted_top_models.png'), dpi=300)
    plt.close()
    print('Saved: 02_actual_vs_predicted_top_models.png')
    
    # C. Residual Plot (Residuals vs Predicted)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    best_preds = predictions_test[best_model_name]
    residuals = y_test - best_preds
    
    axes[0].scatter(best_preds, residuals, alpha=0.35, color='#8e44ad', s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title(f'Residuals vs. Predicted ({best_model_name})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Yield (kg/ha)')
    axes[0].set_ylabel('Residual (Actual - Predicted)')
    axes[0].set_xscale('log')
    
    sns.histplot(residuals, kde=True, ax=axes[1], color='#e67e22', bins=35)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_title(f'Prediction Error Distribution ({best_model_name})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Prediction Error (kg/ha)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_residuals_and_error_distribution.png'), dpi=300)
    plt.close()
    print('Saved: 03_residuals_and_error_distribution.png')
    
    # D. Training vs Validation Learning Curves for Deep Learning Models
    if os.path.exists('models/ann_history.pkl') and os.path.exists('models/cnn_1d_history.pkl'):
        ann_h = joblib.load('models/ann_history.pkl')
        cnn_h = joblib.load('models/cnn_1d_history.pkl')
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(ann_h['train_loss'], label='Train Loss (MSE)', color='#2980b9', linewidth=2)
        axes[0].plot(ann_h['val_loss'], label='Val Loss (MSE)', color='#e74c3c', linewidth=2, linestyle='--')
        axes[0].set_title('ANN / MLP Learning Curves (Loss vs Epochs)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Normalized MSE Loss')
        axes[0].legend()
        
        axes[1].plot(cnn_h['train_loss'], label='Train Loss (MSE)', color='#27ae60', linewidth=2)
        axes[1].plot(cnn_h['val_loss'], label='Val Loss (MSE)', color='#f39c12', linewidth=2, linestyle='--')
        axes[1].set_title('1D CNN Learning Curves (Loss vs Epochs)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Normalized MSE Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '04_deep_learning_loss_curves.png'), dpi=300)
        plt.close()
        print('Saved: 04_deep_learning_loss_curves.png')
        
    print('Model evaluation and visualization completed successfully.')
    print('=' * 80)
    return metrics_df, best_model_name

if __name__ == '__main__':
    evaluate_models()