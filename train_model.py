import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =====================================================
# TRAINING FUNCTION CALLED BY STREAMLIT
# =====================================================
def train_and_save(quick=False):
    """
    quick=True  → For Streamlit Cloud (fast training)
    quick=False → For full local training
    """

    print("Loading dataset...")
    df = pd.read_csv("electric_vehicles_spec_2025.csv.csv")

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

    print("Training columns:", df.columns.tolist())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Faster RandomForest for Streamlit Cloud
    if quick:
        n_estimators = 80
        max_depth = 12
    else:
        n_estimators = 300
        max_depth = 20

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
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("R2:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)

    MODEL_NAME = "final_ev_model.pkl"
    joblib.dump(model, MODEL_NAME)

    print(f"Model saved successfully at: {MODEL_NAME}")
    return MODEL_NAME


# =====================================================
# DEFAULT ENTRY POINT FOR LOCAL RUNNING
# =====================================================
def main():
    print("Starting full training...")
    train_and_save(quick=False)


if __name__ == "__main__":
    main()
