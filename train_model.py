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
