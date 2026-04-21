import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

st.title("🧪 Diabetes Risk Prediction System")
st.markdown("AI-powered early screening tool for diabetes risk assessment")

# -----------------------
# SAFE MODEL LOADING
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model not found at: {model_path}")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------
# SIDEBAR INPUT
# -----------------------
st.sidebar.header("📋 Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

# -----------------------
# PREDICTION
# -----------------------
if st.button("🔍 Predict Risk"):

    try:
        # MUST MATCH TRAINING FEATURES EXACTLY
        input_data = pd.DataFrame([[
            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]], columns=[
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ])

        prob = model.predict_proba(input_data)[0][1]

        st.subheader(f"🧪 Diabetes Probability: {prob:.2f}")

        # -----------------------
        # VISUALIZATION
        # -----------------------
        fig, ax = plt.subplots()
        ax.bar(["No Diabetes", "Diabetes"], [1 - prob, prob])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        st.progress(int(prob * 100))

        # -----------------------
        # RISK LEVEL
        # -----------------------
        if prob < 0.3:
            st.success("🟢 Low Risk")
            st.info("Maintain healthy lifestyle.")

        elif prob < 0.6:
            st.warning("🟡 Medium Risk")
            st.info("Consult doctor if needed.")

        else:
            st.error("🔴 High Risk")
            st.info("Seek medical advice.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning & Streamlit")