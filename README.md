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
