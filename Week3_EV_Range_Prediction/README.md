Week 3 – EV Range Prediction (UI + Backend Integration)

This week focuses on building a fully functional web application for the Electric Vehicle Range Prediction model developed earlier. The goal was to connect the trained ML model with an interactive user interface using Streamlit, while ensuring that the backend preprocessing pipeline (scaling, feature selection, etc.) is preserved during inference.

Work Completed (Week 3)

1. Frontend Development (Streamlit UI)

Designed a simple and clean Streamlit interface where users can enter EV specifications:

Battery Capacity

Top Speed

Vehicle Length

Acceleration

Added real-time prediction display.

Ensured correct input validation and user-friendly ranges.


2. Backend Integration

Built a single pipeline that:

Loads the trained Random Forest model (final_ev_model.pkl)

Loads the preprocessing scaler (scaler.pkl)

Applies the exact same scaling as training

Ensures correct feature ordering before prediction


3. Debugging & Issue Resolution

During integration, several issues were encountered and resolved:

Model predicting the same range for all inputs

Missing scaler mismatch

Feature-order mismatch

Training vs inference pipeline inconsistencies

CSV loading errors

Handling NaN values

Path mismatches in local vs Colab environments

Each issue was fixed by:

Re-training the model with a clean, reduced set of features

Saving a dedicated scaler

Creating a stable, minimal inference pipeline

Ensuring column order matches the training pipeline


4. Final Submission Files

The following files form the complete Week-3 implementation:

Week3_EV_Range_Prediction/
│── app.py
│── train_model.py
│── final_ev_model.pkl
│── scaler.pkl
│── requirements.txt
│── README.md


How to Run the App
pip install -r requirements.txt
streamlit run app.py


This launches the web application locally on http://localhost:8501.

Summary

Week-3 successfully delivers a working, end-to-end ML web application.
The final system integrates:

A trained EV range prediction model

The correct preprocessing pipeline

A functional, user-friendly web UI

This completes the transition from model development (Week 1–2) to real-world usability (Week 3).
