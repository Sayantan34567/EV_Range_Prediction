import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("electric_vehicles_spec_2025.csv.csv")   # <-- change filename if needed

# ---------------------------
# 2. Feature / Target split
# ---------------------------
X = df.drop("range_km", axis=1)
y = df["range_km"]   # UNTOUCHED (no scaling)

# ---------------------------
# 3. Preprocessing
# ---------------------------
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ---------------------------
# 4. Model Pipeline
# ---------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# ---------------------------
# 5. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 6. Train the model
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# 7. Evaluation
# ---------------------------
preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))

# ---------------------------
# 8. Save the model
# ---------------------------
with open("ev_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as ev_model.pkl")




# --- Add this at the end of train_model.py ---
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_save(csv_path="electric_vehicles_spec_2025.csv.csv",
                   model_out="final_ev_model.pkl",
                   features_out="model_features.pkl",
                   scaler_out="scaler.pkl",
                   quick=True):
    """
    Train a RandomForest using the project CSV (assumes CSV is in repo root).
    Saves model, feature list, and scaler into working directory.
    quick=True => smaller grid (faster)
    """
    # 1) Load data
    df = pd.read_csv(csv_path)
    # Basic cleaning/impute (match what you used before)
    df['number_of_cells'] = df['number_of_cells'].fillna(df['number_of_cells'].median())
    df['towing_capacity_kg'] = df['towing_capacity_kg'].fillna(df['towing_capacity_kg'].median())
    df['torque_nm'] = df['torque_nm'].fillna(df['torque_nm'].median())
    df['fast_charging_power_kw_dc'] = df['fast_charging_power_kw_dc'].fillna(df['fast_charging_power_kw_dc'].median())
    df['cargo_volume_l'] = pd.to_numeric(df.get('cargo_volume_l', pd.Series()), errors='coerce').fillna(df.get('cargo_volume_l', pd.Series()).median() if 'cargo_volume_l' in df.columns else 0)
    # drop not needed or difficult columns if present
    drop_list = ['model', 'source_url', 'brand', 'battery_type']
    for c in drop_list:
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors=True)

    # handle simple dummies if present (keeps numeric only if not)
    df = pd.get_dummies(df, drop_first=True)

    # ensure target exists
    if 'range_km' not in df.columns:
        raise RuntimeError("Target 'range_km' not found in CSV.")

    # Keep numeric columns only (safe)
    df = df.select_dtypes(include=[np.number])
    # make sure target is present
    if 'range_km' not in df.columns:
        raise RuntimeError("After conversions, 'range_km' missing.")

    # 2) Train/test split
    X = df.drop(columns=['range_km'])
    y = df['range_km']

    # Fill any remaining NaNs with column medians (safe fallback)
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3) scale numeric features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 4) quick hyperparam grid (small so it runs on streamlit cloud)
    if quick:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
        }
    else:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_

    # 5) evaluate (print to logs)
    preds = best.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)  # sklearn >=1.0 supports squared param
    print(f"TRAIN DONE. R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # 6) Save model, scaler and feature list
    joblib.dump(best, model_out)
    joblib.dump(list(X.columns), features_out)
    joblib.dump(scaler, scaler_out)
    print(f"Saved: {model_out}, {features_out}, {scaler_out}")

# If run directly, allow testing locally
if __name__ == "__main__":
    train_and_save(quick=True)

