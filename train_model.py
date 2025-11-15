import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------
df = pd.read_csv("electric_vehicles_spec_2025.csv.csv")

print("Dataset loaded. Shape:", df.shape)

# ---------------------------------------------------
# 2) Keep ONLY the 9 numeric features + target
# ---------------------------------------------------
FEATURES = [
    "top_speed_kmh",
    "battery_capacity_kWh",
    "number_of_cells",
    "torque_nm",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "length_mm",
    "width_mm"
]

TARGET = "range_km"

df = df[FEATURES + [TARGET]]

print("Columns used:", df.columns.tolist())

# ---------------------------------------------------
# 3) Train-test split
# ---------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 4) Build preprocessing + model pipeline
# ---------------------------------------------------
numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, FEATURES)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42
    ))
])

# ---------------------------------------------------
# 5) Train model
# ---------------------------------------------------
model.fit(X_train, y_train)

print("Model training completed.")

# ---------------------------------------------------
# 6) Evaluate
# ---------------------------------------------------
preds = model.predict(X_test)

print("R² Score:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds)**0.5)

# ---------------------------------------------------
# 7) Save model
# ---------------------------------------------------
MODEL_NAME = "final_ev_model.pkl"
joblib.dump(model, MODEL_NAME)

print(f"Model saved as: {MODEL_NAME}")
