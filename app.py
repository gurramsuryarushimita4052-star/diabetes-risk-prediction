
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction")
st.write("Machine Learning based educational prediction demo.")
st.warning(
    "This tool is for educational purposes only. "
    "It cannot diagnose diabetes or replace medical advice."
)

model_path = Path("artifacts/diabetes_model.joblib")
features_path = Path("artifacts/feature_names.joblib")

if not model_path.exists() or not features_path.exists():
    st.error("Model files not found. First run: python train.py")
    st.stop()

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

st.subheader("Enter Patient Details")

with st.form("diabetes_form"):
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=0
    )
    glucose = st.number_input(
        "Glucose", min_value=0.0, max_value=300.0, value=120.0
    )
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0.0, max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0
    )
    insulin = st.number_input(
        "Insulin", min_value=0.0, max_value=900.0, value=80.0
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=80.0, value=25.0
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
        value=0.5
    )
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=30
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    input_df = pd.DataFrame([values])[feature_names]

    prediction = int(model.predict(input_df)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
        st.metric("Model-estimated probability", f"{probability:.1%}")

    if prediction == 1:
        st.error("The model predicts the positive class.")
    else:
        st.success("The model predicts the negative class.")

    st.caption(
        "This is a model output, not a medical diagnosis. "
        "Consult a qualified healthcare professional for health concerns."
    )
    