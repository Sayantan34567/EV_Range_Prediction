import streamlit as st
import pandas as pd
import joblib
import re
import random

# ======================================================================================
# PAGE CONFIG
# ======================================================================================
st.set_page_config(page_title="EV Range Predictor", page_icon="🔋")

st.title("🔋 EV Range Prediction System")

# ======================================================================================
# LOAD MODEL FIRST (fixes NameError)
# ======================================================================================

MODEL_PATH = "final_ev_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = e


# ======================================================================================
# SIDEBAR — ADMIN TRAIN BUTTON
# ======================================================================================
with st.sidebar:
    st.header("⚙️ Admin Controls")

    if st.button("📘 Train Model on Server"):
        try:
            from train_model import train_and_save
            st.info("Training model... Please wait ⏳")
            train_and_save()       # this will retrain + save
            st.success("Model retrained successfully! Reloading app...")
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {e}")


# ======================================================================================
# UI FORM INPUT
# ======================================================================================
st.header("Enter EV Specifications to Predict Range")

# Features used by the model
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

# Default fallback values
DEFAULTS = {
    "number_of_cells": 400,
    "torque_nm": 300,
    "fast_charging_power_kw_dc": 120,
    "towing_capacity_kg": 0,
    "width_mm": 1820
}

# ---------------------------
# FORM UI
# ---------------------------
with st.form("ev_form"):
    battery_capacity = st.number_input(
        "Battery Capacity (kWh)", min_value=20.0, max_value=150.0, value=60.0, step=1.0
    )
    top_speed = st.number_input(
        "Top Speed (km/h)", min_value=80.0, max_value=350.0, value=150.0, step=5.0
    )
    vehicle_length = st.number_input(
        "Vehicle Length (mm)", min_value=3000.0, max_value=6000.0, value=4500.0, step=50.0
    )
    acceleration = st.number_input(
        "0–100 km/h Acceleration (seconds)", min_value=2.0, max_value=15.0, value=8.0, step=0.1
    )

    submitted = st.form_submit_button("🔮 Predict Range")

# ======================================================================================
# PREDICTION
# ======================================================================================
if submitted:
    if not model_loaded:
        st.error(f"Model is not loaded: {model_error}")
    else:
        input_data = pd.DataFrame([{
            "top_speed_kmh": top_speed,
            "battery_capacity_kWh": battery_capacity,
            "number_of_cells": DEFAULTS["number_of_cells"],
            "torque_nm": DEFAULTS["torque_nm"],
            "acceleration_0_100_s": acceleration,
            "fast_charging_power_kw_dc": DEFAULTS["fast_charging_power_kw_dc"],
            "towing_capacity_kg": DEFAULTS["towing_capacity_kg"],
            "length_mm": vehicle_length,
            "width_mm": DEFAULTS["width_mm"]
        }])

        try:
            pred = model.predict(input_data)[0]
            st.success(f"Estimated Range: **{pred:.2f} km**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")

# ======================================================================================
# CHATBOT SECTION
# ======================================================================================
st.header("Range Assistant Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "text": "Hello! You can type something like: predict 60,4500,180,7"}
    ]

# Display chat messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")

# User text input
user_input = st.text_input("Message:", key="chat_input")

def extract_numbers(text):
    nums = re.findall(r"\d+\.?\d*", text)
    return [float(n) for n in nums]

def build_row(nums):
    row = DEFAULTS.copy()
    if len(nums) >= 4:
        row.update({
            "battery_capacity_kWh": nums[0],
            "length_mm": nums[1],
            "top_speed_kmh": nums[2],
            "acceleration_0_100_s": nums[3]
        })
    return row

# Send message
if st.button("Send"):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        msg = user_input.lower()
        reply = ""

        if "predict" in msg:
            nums = extract_numbers(user_input)
            if len(nums) >= 4:
                row = build_row(nums)
                df_input = pd.DataFrame([row])

                if model_loaded:
                    try:
                        pred = model.predict(df_input)[0]
                        reply = f"Estimated range: **{pred:.2f} km**"
                    except Exception:
                        reply = "Prediction failed — please recheck your values."
                else:
                    reply = "Model is not loaded on the server."
            else:
                reply = "Please provide 4 values like: predict 60,4500,180,8"

        else:
            reply = random.choice([
                "Try: predict 60,4500,180,7",
                "You can ask about battery, speed, acceleration etc."
            ])

        st.session_state.chat_history.append({"role": "bot", "text": reply})
        st.rerun()
