import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("../models/best_model.pkl")

# Get exact feature names from model
feature_columns = model.feature_names_in_

st.title("🔧 Equipment Failure Prediction System")

# User Inputs
air_temp = st.number_input("Air Temperature (K)", value=300.0)
process_temp = st.number_input("Process Temperature (K)", value=310.0)
rot_speed = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=0)

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

# Create dictionary with ALL required features initialized to 0
input_data = {col: 0 for col in feature_columns}

# Fill numeric features (only if they exist in model)
if "Air temperature [K]" in input_data:
    input_data["Air temperature [K]"] = air_temp

if "Process temperature [K]" in input_data:
    input_data["Process temperature [K]"] = process_temp

if "Rotational speed [rpm]" in input_data:
    input_data["Rotational speed [rpm]"] = rot_speed

if "Torque [Nm]" in input_data:
    input_data["Torque [Nm]"] = torque

if "Tool wear [min]" in input_data:
    input_data["Tool wear [min]"] = tool_wear

# Handle machine type dynamically
type_column = f"Type_{machine_type}"
if type_column in input_data:
    input_data[type_column] = 1

# Convert to DataFrame using exact order
input_df = pd.DataFrame([input_data], columns=feature_columns)

# Predict
if st.button("Predict Failure"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠️ Machine Will Fail")
    else:
        st.success("✅ Machine Operating Normally")

    st.write(f"Failure Probability: {probability:.2f}")