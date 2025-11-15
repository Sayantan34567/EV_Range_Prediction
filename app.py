import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="EV Range Predictor", page_icon="🔋")
st.title("EV Range Prediction System")

MODEL_PATH = "final_ev_model.pkl"

# ----------------------------
# TRAIN MODEL (ONLY IF NEEDED)
# ----------------------------
def auto_train_if_needed():
    """
    If model file does NOT exist, train it automatically.
    If it exists, do nothing.
    """
    if os.path.exists(MODEL_PATH):
        return  # Model already exists

    try:
        import train_model
        st.info("Training model for the first time… please wait ⏳")
        train_model.main()    # call the function inside train_model.py
        st.success("Model trained automatically!")
    except Exception as e:
        st.error(f"Automatic training failed: {e}")

# Ensure model exists
auto_train_if_needed()

# ----------------------------
# LOAD MODEL
# ----------------------------
model = None
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ----------------------------
# FEATURES USED BY MODEL
# ----------------------------
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

# ----------------------------
# FORM UI
# ----------------------------
st.header("Enter EV Specifications")

with st.form("ev_form"):
    battery_capacity = st.number_input("Battery Capacity (kWh)", 20.0, 150.0, 60.0)
    top_speed = st.number_input("Top Speed (km/h)", 80.0, 350.0, 150.0)
    vehicle_length = st.number_input("Vehicle Length (mm)", 3000.0, 6000.0, 4500.0)
    acceleration = st.number_input("0–100 km/h Acceleration (seconds)", 2.0, 15.0, 8.0)

    submitted = st.form_submit_button("🔮 Predict Range")

if submitted:
    if model is None:
        st.error("Model is not ready. Please refresh the page.")
    else:
        input_row = pd.DataFrame([{
            "top_speed_kmh": top_speed,
            "battery_capacity_kWh": battery_capacity,
            "number_of_cells": DEFAULTS["number_of_cells"],
            "torque_nm": DEFAULTS["torque_nm"],
            "acceleration_0_100_s": acceleration,
            "fast_charging_power_kw_dc": DEFAULTS["fast_charging_power_kw_dc"],
            "towing_capacity_kg": DEFAULTS["towing_capacity_kg"],
            "length_mm": vehicle_length,
            "width_mm": DEFAULTS["width_mm"],
        }])

        try:
            pred = model.predict(input_row)[0]
            st.success(f"Predicted EV Range: **{pred:.2f} km**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ----------------------------
# CHATBOT SECTION
# ----------------------------
st.markdown("---")
st.header("Range Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "text": "Hello! Type 'predict 60,4500,180,7' or ask me EV questions!"}
    ]

# Display chat
for entry in st.session_state.chat_history:
    role = "You" if entry["role"] == "user" else "Bot"
    st.write(f"**{role}:** {entry['text']}")

user_input = st.text_input("Your message:")

import re
import random

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
            "acceleration_0_100_s": nums[3],
        })
    return row

if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        lower = user_input.lower()
        reply = ""

        if "predict" in lower:
            nums = extract_numbers(lower)
            if len(nums) >= 4:
                row = build_row(nums)
                df_input = pd.DataFrame([row])
                try:
                    pred = model.predict(df_input)[0]
                    reply = f"Estimated range: **{pred:.2f} km**"
                except Exception as e:
                    reply = f"Prediction failed: {e}"
            else:
                reply = "Please provide 4 numbers: predict 60,4500,180,7"
        else:
            reply = random.choice([
                "Try 'predict 60,4500,180,7'!",
                "I can estimate EV ranges. Ask me something!",
            ])

        st.session_state.chat_history.append({"role": "bot", "text": reply})
        st.rerun()
