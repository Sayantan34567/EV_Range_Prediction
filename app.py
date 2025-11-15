import streamlit as st
import pandas as pd
import joblib
import re
import random
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="EV Range Predictor", page_icon="🔋")

st.title("EV Range Prediction System")

MODEL_PATH = "final_ev_model.pkl"
SCALER_PATH = "scaler.pkl"

# ---------------------------------------------------------
# TRAIN MODEL BUTTON (RUNS ON STREAMLIT CLOUD)
# ---------------------------------------------------------
with st.sidebar:
    st.header("Admin Controls")
    if st.button("Train Model on Server"):
        try:
            import train_model
            st.info("Training model… Please wait 20–30 seconds ⏳")

            # Call the main() function inside train_model.py
            train_model.main()

            st.success("Model retrained & saved on server!")
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------------------------------------------------------
# LOAD TRAINED MODEL + SCALER
# ---------------------------------------------------------
model, scaler = None, None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
    else:
        st.warning("Model not found. Train it from the sidebar.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        st.warning("Scaler not found. Train model again.")

except Exception as e:
    st.error(f"Failed to load model: {e}")


# The 9 features your model uses
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

DEFAULTS = {
    "number_of_cells": 400,
    "torque_nm": 300,
    "fast_charging_power_kw_dc": 120,
    "towing_capacity_kg": 0,
    "width_mm": 1820
}

# ---------------------------------------------------------
# STRUCTURED INPUT UI
# ---------------------------------------------------------
st.header("Enter EV Specifications to Predict Range")

with st.form("ev_form"):
    battery_capacity = st.number_input(
        "Battery Capacity (kWh)", 20.0, 150.0, 60.0, step=1.0
    )
    top_speed = st.number_input(
        "Top Speed (km/h)", 80.0, 350.0, 150.0, step=5.0
    )
    vehicle_length = st.number_input(
        "Vehicle Length (mm)", 3000.0, 6000.0, 4500.0, step=50.0
    )
    acceleration = st.number_input(
        "0–100 km/h Acceleration (seconds)", 2.0, 15.0, 8.0, step=0.1
    )

    submitted = st.form_submit_button("🔮 Predict Range")

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
# ------------------- PREDICT FROM FORM (robust) -------------------
if submitted:
    # Build a safe input row that exactly matches FEATURES
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

    # defaults for any features we don't ask the user to input
    DEFAULTS_ROW = {
        "top_speed_kmh": float(top_speed),
        "battery_capacity_kWh": float(battery_capacity),
        "number_of_cells": DEFAULTS["number_of_cells"],
        "torque_nm": DEFAULTS["torque_nm"],
        "acceleration_0_100_s": float(acceleration),
        "fast_charging_power_kw_dc": DEFAULTS["fast_charging_power_kw_dc"],
        "towing_capacity_kg": DEFAULTS["towing_capacity_kg"],
        "length_mm": float(vehicle_length),
        "width_mm": DEFAULTS["width_mm"]
    }

    # Create DataFrame with EXACT column order and no extras
    input_df = pd.DataFrame([DEFAULTS_ROW])[FEATURES]

    try:
        pred = model.predict(input_df)[0]
        st.success(f"Predicted EV Range: **{pred:.2f} km**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")

# ---------------------------------------------------------
# CHATBOT SECTION
# ---------------------------------------------------------
st.header("Range Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "Me", "text": "Hi! Type 'predict 60,4500,180,7' or ask anything about EVs!"}
    ]

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")

user_input = st.text_input("Message:", key="input_box")

def extract_numbers(text):
    nums = re.findall(r"\d+\.?\d*", text)
    return [float(n) for n in nums]

if st.button("Send", key="send_btn"):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        msg = user_input.lower()

        if "predict" in msg:
            nums = extract_numbers(user_input)

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

            # Start with defaults
            row = {
                "top_speed_kmh": DEFAULTS["number_of_cells"],   
                "battery_capacity_kWh": DEFAULTS["number_of_cells"],
                "number_of_cells": DEFAULTS["number_of_cells"],
                "torque_nm": DEFAULTS["torque_nm"],
                "acceleration_0_100_s": DEFAULTS["number_of_cells"],
                "fast_charging_power_kw_dc": DEFAULTS["fast_charging_power_kw_dc"],
                "towing_capacity_kg": DEFAULTS["towing_capacity_kg"],
                "length_mm": DEFAULTS["width_mm"],
                "width_mm": DEFAULTS["width_mm"]
            }

            # Overwrite defaults with extracted values
            if len(nums) >= 4:
                row["battery_capacity_kWh"] = nums[0]
                row["length_mm"] = nums[1]
                row["top_speed_kmh"] = nums[2]
                row["acceleration_0_100_s"] = nums[3]
            else:
                bot_reply = "Please provide 4 values like: predict 60,4500,180,8"
                st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
                st.rerun()

            # Final dataframe with correct order
            df_input = pd.DataFrame([row])[FEATURES]

            try:
                pred = model.predict(df_input)[0]
                bot_reply = f"Estimated range: **{pred:.2f} km**"
            except Exception as e:
                bot_reply = f"Prediction failed: {e}"
        else:
            reply = random.choice([
                "Try: predict 60,4500,180,7!",
                "Ask me about battery capacity, top speed, or acceleration."
            ])

        st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
        st.rerun()



