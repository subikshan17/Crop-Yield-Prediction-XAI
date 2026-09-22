import os
import pandas as pd
import numpy as np

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('src', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/plots/eda', exist_ok=True)
os.makedirs('results/plots/models', exist_ok=True)
os.makedirs('results/plots/shap', exist_ok=True)

print("Directories created.")
