
import streamlit as st
import pandas as pd
import joblib
import sklearn

# Load the trained model
try:
    pipeline = joblib.load("random_forest_model.joblib")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_model.joblib' is in the same directory.")
    st.stop()

st.title("Heart Disease Prediction")

st.write("Enter the patient's information to predict the likelihood of heart disease.")

# Define input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain_type = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Create a button to predict
if st.button("Predict"):
    # Prepare the input data as a DataFrame with the correct column order and names
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain_type],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": ["Normal"], # Assuming 'Normal' as a default since it's not in input fields
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    # Make prediction
    prediction = pipeline.predict(input_data)

    # Display the prediction
    if prediction[0] == "Yes":
        st.error("Prediction: High likelihood of Heart Disease")
    else:
        st.success("Prediction: Low likelihood of Heart Disease")

