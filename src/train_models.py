import os
import time
import joblib
import torch
import numpy as np
import pandas as pd

from models_ml import get_ml_models
from models_dl import ANNRegressor, CNN1DRegressor, train_torch_model, predict_torch_model

def train_and_save_all_models(data_dir='data/processed', models_dir='models'):
    print('=' * 80)
    print('STAGE 5 & 6: MODEL TRAINING & CROSS-ARCHITECTURE BENCHMARKING')
    print('=' * 80)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load processed datasets
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}')
    
    trained_models = {}
    training_times = {}
    predictions_val = {}
    predictions_test = {}
    histories = {}
    
    # 1. Train Traditional ML Models
    ml_models = get_ml_models()
    for name, model in ml_models.items():
        print(f'\n>>> Training [{name}]...')
        start_t = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_t
        
        training_times[name] = elapsed
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        
        predictions_val[name] = val_preds
        predictions_test[name] = test_preds
        trained_models[name] = model
        
        # Save model
        clean_name = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(models_dir, f'{clean_name}.pkl'))
        print(f'    Trained in {elapsed:.2f}s | Saved: models/{clean_name}.pkl')
        
    # 2. Train Deep Learning ANN / MLP
    print('\n>>> Training [Artificial Neural Network (ANN / MLP)]...')
    input_dim = X_train.shape[1]
    ann_model = ANNRegressor(input_dim)
    start_t = time.time()
    ann_trained, ann_history = train_torch_model(ann_model, X_train, y_train, X_val, y_val, epochs=120, batch_size=64, lr=0.003, patience=20)
    elapsed = time.time() - start_t
    
    training_times['ANN'] = elapsed
    predictions_val['ANN'] = predict_torch_model(ann_trained, X_val, ann_history)
    predictions_test['ANN'] = predict_torch_model(ann_trained, X_test, ann_history)
    trained_models['ANN'] = ann_trained
    histories['ANN'] = ann_history
    
    torch.save(ann_trained.state_dict(), os.path.join(models_dir, 'ann_model.pt'))
    joblib.dump(ann_history, os.path.join(models_dir, 'ann_history.pkl'))
    print(f'    Trained in {elapsed:.2f}s | Saved: models/ann_model.pt')
    
    # 3. Train Deep Learning 1D CNN
    print('\n>>> Training [1D Convolutional Neural Network (1D CNN)]...')
    cnn_model = CNN1DRegressor(input_dim)
    start_t = time.time()
    cnn_trained, cnn_history = train_torch_model(cnn_model, X_train, y_train, X_val, y_val, epochs=120, batch_size=64, lr=0.003, patience=20)
    elapsed = time.time() - start_t
    
    training_times['1D CNN'] = elapsed
    predictions_val['1D CNN'] = predict_torch_model(cnn_trained, X_val, cnn_history)
    predictions_test['1D CNN'] = predict_torch_model(cnn_trained, X_test, cnn_history)
    trained_models['1D CNN'] = cnn_trained
    histories['1D CNN'] = cnn_history
    
    torch.save(cnn_trained.state_dict(), os.path.join(models_dir, 'cnn_1d_model.pt'))
    joblib.dump(cnn_history, os.path.join(models_dir, 'cnn_1d_history.pkl'))
    print(f'    Trained in {elapsed:.2f}s | Saved: models/cnn_1d_model.pt')
    
    # Save all test predictions & metadata
    joblib.dump(predictions_val, os.path.join(data_dir, 'predictions_val.pkl'))
    joblib.dump(predictions_test, os.path.join(data_dir, 'predictions_test.pkl'))
    joblib.dump(training_times, os.path.join(models_dir, 'training_times.pkl'))
    
    print('\nAll models trained and checkpoints saved successfully.')
    print('=' * 80)
    return trained_models, predictions_test, training_times, histories

if __name__ == '__main__':
    train_and_save_all_models()