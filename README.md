# EV Range Prediction — Machine Learning Project

**Predicting electric vehicle driving range (km) from vehicle specifications.**

---

## Project Overview

This repository contains a complete machine-learning pipeline that predicts an electric vehicle’s driving **range (km)** using technical specifications (battery capacity, efficiency, motor power, weight, etc.).  
The goal is to demonstrate data cleaning, exploratory data analysis, model training, evaluation, and model optimization.

---

## Problem Statement

> Given an EV’s specifications and contextual features, predict its expected driving range in kilometres.

---

## Dataset

- **Source:** Kaggle — *Electric Vehicle Specifications Dataset (2025)*  
- **Filename (in this repo):** `electric_vehicles_spec_2025.csv`  
- **License:** CC BY 4.0

> The dataset includes vehicle specs such as `battery_capacity_kWh`, `efficiency_wh_per_km`, `top_speed_kmh`, `acceleration_0_100_s`, `torque_nm`, `car_body_type`, and the target `range_km`.

---

## Key Steps Performed

1. Data loading and cleaning (handling missing values, type conversions)  
2. Encoding categorical features and feature selection (correlation + variance checks)  
3. Exploratory Data Analysis (visualizations, correlation heatmaps)  
4. Model training: Linear Regression, Decision Tree, Random Forest  
5. Model evaluation: R², MAE, RMSE  
6. Model optimization with `GridSearchCV` (Random Forest)  
7. Saved optimized model for deployment

---

## Results (summary)

- **Best model:** Random Forest Regressor (after tuning)  
- **Performance (example)**  
  - R² ≈ 0.93  
  - MAE ≈ 20 km (raw units — depends on whether features are scaled)  
  - RMSE ≈ 27 km

(Full metrics, plots, and model parameter details are available in the notebook.)

---

## How to run this project (local / Colab)

### Option A — Run in Google Colab (recommended)
1. Open `EV_Range_Prediction_Notebook.ipynb` in Google Colab.  
2. Mount your Google Drive (if needed) and update the dataset path if you placed it in Drive.  
3. Run cells top-to-bottom.  
4. The notebook contains EDA, model training, evaluation, and model saving steps.

### Option B — Run locally
1. Create and activate a Python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows




## 📈 Week 2 — Model Optimization and Persistence

**Objective:**  
Enhance the Week 1 baseline model by performing hyperparameter tuning, analyzing feature importance, and saving the optimized model for deployment in the upcoming interface.

### 🔧 Improvements Implemented

1. **Hyperparameter Tuning:**  
   Utilized `GridSearchCV` to identify the best combination of parameters for the Random Forest model, improving the R² score and reducing overfitting.

2. **Feature Importance Analysis:**  
   Generated a horizontal bar plot visualizing each feature’s contribution to the final prediction.

3. **Model Persistence:**  
   Saved the optimized model and the corresponding feature list as `.pkl` files using `joblib`, ensuring that the trained model can be reused later without retraining.

4. **Drive Integration:**  
   Configured the notebook to automatically save the model artifacts both locally and inside Google Drive (`/content/drive/MyDrive/EV_Range_Prediction`).

### 🧩 Output Files

| File Name | Description |
|------------|-------------|
| `optimized_rf_model.pkl` | Serialized optimized Random Forest model |
| `model_features.pkl` | List of features used during model training |

### 📊 Next Steps (Week 3 Preview)

- Build a **Streamlit interface** for real-time predictions.  
- Integrate a **Generative AI chatbot** to assist users with queries about the dataset and model performance.  
- Deploy the complete project on Streamlit Cloud.

---



