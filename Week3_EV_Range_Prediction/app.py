import streamlit as st
import pandas as pd
import joblib
import re
import random

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="EV Range Predictor", page_icon="🔋")

st.title("EV Range Prediction System")

# ---------------------------------------------------------
# LOAD TRAINED MODEL (PIPELINE)
# ---------------------------------------------------------
MODEL_PATH = "final_ev_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
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

# Default values for fields not shown to users:
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
# PREDICT FROM FORM
# ---------------------------------------------------------
if submitted:
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
        {"role": "Me", "text": "Hi! Type 'predict 60,4500,180,7' or ask any EV question!"}
    ]

# Show chat messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")

user_input = st.text_input("Message:", key="input_box")

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

if st.button("Send", key="send_btn"):
    if user_input.strip() != "":
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        msg = user_input.lower()
        bot_reply = ""

        if "predict" in msg:
            nums = extract_numbers(user_input)
            if len(nums) >= 4:
                row = build_row(nums)
                df_input = pd.DataFrame([row])
                try:
                    pred = model.predict(df_input)[0]
                    bot_reply = f"Estimated range: **{pred:.2f} km**"
                except Exception as e:
                    bot_reply = f"Prediction failed: {e}"
            else:
                bot_reply = "Please provide 4 values like: predict 60,4500,180,8"
        else:
            bot_reply = random.choice([
                "Try typing something like: predict 60,4500,180,7!",
                "Ask me about battery capacity, top speed, acceleration etc."
            ])

        st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
        st.rerun()
