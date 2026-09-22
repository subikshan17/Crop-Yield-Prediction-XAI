# 🌾 CropYield AI — Climate-Aware Crop Yield Predictor

An **Explainable Machine Learning (XAI) System** for predicting agricultural crop yields based on micro-climate parameters and regional agricultural variables, featuring real-time feature interaction analysis and SHAP feature attribution.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-000000?style=flat&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green?style=flat)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20Explainability-blue?style=flat)

---

## 🌟 Key Features

* **⚡ Real-Time Yield Prediction**: Calculates crop yield outputs ($kg/ha$ & $tonnes/ha$) dynamically from live user inputs across 23 major crops, 31 districts, and 4 seasons.
* **🔍 Dual SHAP Explainability**:
  * **Global Dataset Baseline**: Ranks overall climate drivers across 18,445 integrated records.
  * **Live Input Impact Breakdown**: Computes exact $+ / - \text{ kg/ha}$ contribution for each specific weather variable.
* **🌿 Climate Feature Interaction & Comparison**: Evaluates pairwise climate interactions (e.g. *Soil Moisture vs Rainfall*, *Soil Moisture vs Max Temperature*) to identify coupled climate stress leverage.
* **📊 Historical Range Context**: Contextualizes yield outputs against historical minimum, average, and maximum bounds for the selected crop.
* **🎨 Pastel Nature UI**: Modern, responsive dashboard design system built with custom CSS tokens.

---

## ⚙️ Tech Stack

* **Machine Learning**: XGBoost ($R^2 \approx 0.952$), Random Forest ($R^2 \approx 0.948$), LightGBM ($R^2 \approx 0.935$), Scikit-Learn.
* **Explainable AI**: SHAP (SHapley Additive exPlanations).
* **Backend**: Python 3.9+, Flask WSGI Server.
* **Frontend**: HTML5, Custom CSS3 (Pastel Nature Theme), JavaScript (ES6+).

---

## 📁 Dataset Characteristics

* **Integrated Records**: 18,445 clean agricultural records (1997–2013).
* **Regional Scope**: 31 Districts (Tamil Nadu region).
* **Crops Covered**: 23 crops (Rice, Sugarcane, Cotton, Groundnut, Maize, Banana, etc.).
* **8 Micro-Climate Parameters**: Avg Temp, Min Temp, Max Temp, Rainfall, Relative Humidity, Solar Radiation, Wind Speed, Soil Moisture.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 📄 License & Attribution
Developed for Explainable Machine Learning Based Analysis of Climate Factors Affecting Crop Yield.
