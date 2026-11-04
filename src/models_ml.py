from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor

def get_ml_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0
        ),
        'EBM': ExplainableBoostingRegressor(
            interactions=10,
            random_state=42,
            n_jobs=-1
        )
    }
    return models