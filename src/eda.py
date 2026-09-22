import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_eda(data_path='data/integrated_crop_climate_dataset.csv', output_dir='results/plots/eda'):
    print('=' * 80)
    print('STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)')
    print('=' * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    
    # 1. Target Yield Distribution (Linear & Log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['Yield_kg_per_ha'], kde=True, ax=axes[0], color='#2b5c8f', bins=40)
    axes[0].set_title('Crop Yield Distribution (Linear Scale)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Yield (kg/ha)')
    axes[0].set_ylabel('Frequency')
    
    sns.histplot(np.log1p(df['Yield_kg_per_ha']), kde=True, ax=axes[1], color='#27ae60', bins=40)
    axes[1].set_title('Log-Transformed Crop Yield Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Log(1 + Yield (kg/ha))')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_yield_distribution.png'), dpi=300)
    plt.close()
    print('Saved: 01_yield_distribution.png')
    
    # 2. Yield by Major Crop
    top_crops = df.groupby('Crop')['Yield_kg_per_ha'].median().sort_values(ascending=False).index[:12]
    df_top_crops = df[df['Crop'].isin(top_crops)].copy()
    
    plt.figure(figsize=(14, 6))
    order = df_top_crops.groupby('Crop')['Yield_kg_per_ha'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_top_crops, x='Crop', y='Yield_kg_per_ha', order=order, palette='viridis')
    plt.title('Yield Variability across Top Crops (Log-Scale Y)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Crop Type')
    plt.ylabel('Yield (kg/ha) [Log Scale]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_yield_by_crop.png'), dpi=300)
    plt.close()
    print('Saved: 02_yield_by_crop.png')
    
    # 3. Yield by District (Top Agricultural Districts)
    plt.figure(figsize=(16, 6))
    d_order = df.groupby('District_Name')['Yield_kg_per_ha'].median().sort_values(ascending=False).index
    sns.barplot(data=df, x='District_Name', y='Yield_kg_per_ha', order=d_order, ci=None, palette='mako')
    plt.title('Average Crop Productivity across Tamil Nadu Districts', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60, ha='right')
    plt.xlabel('District')
    plt.ylabel('Mean Yield (kg/ha)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_yield_by_district.png'), dpi=300)
    plt.close()
    print('Saved: 03_yield_by_district.png')
    
    # 4. Multi-Year Yield Progression (Temporal Trend)
    plt.figure(figsize=(12, 5))
    yearly_stats = df.groupby('Crop_Year')['Yield_kg_per_ha'].agg(['mean', 'median', 'std']).reset_index()
    plt.plot(yearly_stats['Crop_Year'], yearly_stats['mean'], marker='o', color='#2980b9', linewidth=2.5, label='Mean Yield')
    plt.plot(yearly_stats['Crop_Year'], yearly_stats['median'], marker='s', color='#e67e22', linewidth=2.0, linestyle='--', label='Median Yield')
    plt.fill_between(yearly_stats['Crop_Year'], yearly_stats['mean'] - 0.5*yearly_stats['std'], yearly_stats['mean'] + 0.5*yearly_stats['std'], color='#2980b9', alpha=0.15)
    plt.title('Annual Crop Yield Progression across Tamil Nadu (1997 - 2013)', fontsize=14, fontweight='bold')
    plt.xlabel('Crop Year')
    plt.ylabel('Yield (kg/ha)')
    plt.xticks(yearly_stats['Crop_Year'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_yield_over_years.png'), dpi=300)
    plt.close()
    print('Saved: 04_yield_over_years.png')
    
    # 5. Climate Factors vs Yield (Bivariate Scatter with Trendlines)
    climate_vars = [
        ('Temperature_Avg', 'Average Temperature (°C)', '#e74c3c'),
        ('Rainfall_Precipitation', 'Rainfall / Precipitation (mm)', '#3498db'),
        ('Relative_Humidity', 'Relative Humidity (%)', '#1abc9c'),
        ('Soil_Moisture', 'Soil Moisture (% Volumetric)', '#8e44ad')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    sample_df = df.sample(n=min(3000, len(df)), random_state=42)
    
    for idx, (col, label, color) in enumerate(climate_vars):
        sns.regplot(data=sample_df, x=col, y='Yield_kg_per_ha', ax=axes[idx],
                    scatter_kws={'alpha': 0.3, 'color': color, 's': 20},
                    line_kws={'color': '#2c3e50', 'linewidth': 2})
        axes[idx].set_title(f'{label} vs. Crop Yield', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel(label)
        axes[idx].set_ylabel('Yield (kg/ha)')
        axes[idx].set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_climate_vs_yield_bivariate.png'), dpi=300)
    plt.close()
    print('Saved: 05_climate_vs_yield_bivariate.png')
    
    # 6. Correlation Heatmap (Pearson & Spearman)
    num_cols = [
        'Area', 'Temperature_Avg', 'Temperature_Min', 'Temperature_Max',
        'Rainfall_Precipitation', 'Relative_Humidity', 'Solar_Radiation',
        'Wind_Speed', 'Soil_Moisture', 'Yield_kg_per_ha'
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    corr_pearson = df[num_cols].corr(method='pearson')
    corr_spearman = df[num_cols].corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    
    sns.heatmap(corr_pearson, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Pearson Linear Correlation Matrix', fontsize=13, fontweight='bold')
    
    sns.heatmap(corr_spearman, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title('Spearman Monotonic Rank Correlation Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_correlation_heatmaps.png'), dpi=300)
    plt.close()
    print('Saved: 06_correlation_heatmaps.png')
    
    # 7. Climate Factor Distributions & Outlier Boxplots
    climate_cols = [
        'Temperature_Avg', 'Temperature_Min', 'Temperature_Max',
        'Rainfall_Precipitation', 'Relative_Humidity', 'Solar_Radiation',
        'Wind_Speed', 'Soil_Moisture'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(climate_cols):
        sns.boxplot(y=df[col], ax=axes[i], color='#34495e')
        axes[i].set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('')
        
    plt.suptitle('Environmental & Climate Feature Distribution & Outlier Audit', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_climate_distributions_outliers.png'), dpi=300)
    plt.close()
    print('Saved: 07_climate_distributions_outliers.png')
    
    print('\nEDA visual generation completed successfully. All plots saved to results/plots/eda/.')
    print('=' * 80)

if __name__ == '__main__':
    run_eda()