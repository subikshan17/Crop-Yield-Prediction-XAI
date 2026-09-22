import sys
import os
import time

# Ensure src is in python path
sys.path.append(os.path.abspath('src'))

from data_inspection import inspect_all_datasets
from data_merge import integrate_datasets
from preprocessing import preprocess_data
from eda import run_eda
from train_models import train_and_save_all_models
from evaluate import evaluate_models
from explainability_shap import run_shap_analysis
from feature_importance_comparison import generate_feature_importance_comparison

def main():
    total_start = time.time()
    print('#' * 80)
    print('# PROJECT PIPELINE: EXPLAINABLE ML FOR CLIMATE-CROP YIELD ANALYSIS')
    print('#' * 80)
    
    # Step 1: Inspect raw datasets
    inspect_all_datasets()
    
    # Step 2: Merge and create integrated dataset
    integrate_datasets()
    
    # Step 3: Preprocess, scale, and time-aware split
    preprocess_data()
    
    # Step 4: Exploratory Data Analysis (EDA)
    run_eda()
    
    # Step 5 & 6: Train all models
    train_and_save_all_models()
    
    # Step 7: Evaluate regression metrics and plot diagnostics
    evaluate_models()
    
    # Step 8: Compute SHAP explanations and visualizations
    run_shap_analysis()
    
    # Step 9: Triangulate feature importances
    generate_feature_importance_comparison()
    
    total_elapsed = time.time() - total_start
    print('\n' + '#' * 80)
    print(f'# FULL PIPELINE EXECUTION FINISHED SUCCESSFULLY IN {total_elapsed:.1f} SECONDS')
    print('#' * 80)

if __name__ == '__main__':
    main()