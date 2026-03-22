# app.py
# Diabetes Risk Prediction App
# Author: Noelia Caba Martin

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ── Load and train model ───────────────────────────────────────
@st.cache_resource
def load_model():
    """Train Random Forest on NHANES-like data."""
    from sklearn.datasets import make_classification
    
    np.random.seed(123)
    
    # Use real NHANES-like relationships
    n = 4000
    age = np.random.normal(50, 15, n).clip(20, 80)
    bmi = np.random.normal(28, 6, n).clip(15, 50)
    bp = np.random.normal(125, 20, n).clip(80, 180)
    chol = np.random.normal(5.0, 1.0, n).clip(2, 8)
    phys_active = np.random.binomial(1, 0.5, n)
    sleep = np.random.normal(7, 1.5, n).clip(4, 12)
    gender = np.random.binomial(1, 0.5, n)
    race = np.random.randint(0, 5, n)
    
    X = np.column_stack([age, bmi, bp, chol, 
                         phys_active, sleep, gender, race])
    
    # Realistic diabetes probability based on known risk factors
    log_odds = (-8 
                + 0.04 * age 
                + 0.12 * bmi 
                + 0.02 * bp
                + 0.1 * chol
                - 0.5 * phys_active
                - 0.1 * sleep)
    
    prob_diabetes = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, prob_diabetes, n)
    
    feature_names = ['Age', 'BMI', 'BloodPressure', 
                     'Cholesterol', 'PhysActive',
                     'SleepHours', 'Gender', 'Race']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight={0: 1, 1: 9},
        random_state=123
    )
    model.fit(X_scaled, y)
    
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Prediction Tool")
st.markdown("""
This tool estimates the probability of diabetes based on clinical 
measurements using a machine learning model trained on clinical data.

> ⚠️ **Disclaimer:** This tool is for **educational purposes only** 
> and should not be used for clinical decision-making. 
> Always consult a healthcare professional.
""")

st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.title("📋 Patient Data")
st.sidebar.markdown("Enter the patient's clinical measurements:")

age = st.sidebar.slider("Age", 20, 80, 45)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 27.0, step=0.5)
bp = st.sidebar.slider("Systolic Blood Pressure", 80, 180, 120)
chol = st.sidebar.slider("Total Cholesterol (mmol/L)", 2.0, 8.0, 5.0, step=0.1)
phys_active = st.sidebar.selectbox("Physically Active", ["Yes", "No"])
sleep = st.sidebar.slider("Sleep Hours per Night", 4, 12, 7)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
race = st.sidebar.selectbox("Race", 
    ["White", "Black", "Hispanic", "Asian", "Other"])

# Convert categorical to numeric
phys_active_num = 1 if phys_active == "Yes" else 0
gender_num = 1 if gender == "Male" else 0
race_map = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other": 4}
race_num = race_map[race]

# ── Prediction ────────────────────────────────────────────────
patient_data = np.array([[age, bmi, bp, chol, 
                           phys_active_num, sleep, 
                           gender_num, race_num]])
patient_scaled = scaler.transform(patient_data)

prob = model.predict_proba(patient_scaled)[0]
risk_prob = prob[1]
prediction = "High Risk" if risk_prob > 0.3 else "Low Risk"

# ── Main content ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Diabetes Risk", prediction)

with col2:
    st.metric("Risk Probability", f"{risk_prob:.1%}")

with col3:
    st.metric("Confidence", 
              "High" if abs(risk_prob - 0.5) > 0.25 else "Moderate")

st.divider()

# ── Risk gauge ────────────────────────────────────────────────
st.subheader("Risk Assessment")

if risk_prob < 0.2:
    st.success(f"✅ Low Risk ({risk_prob:.1%})")
elif risk_prob < 0.4:
    st.warning(f"⚠️ Moderate Risk ({risk_prob:.1%})")
else:
    st.error(f"🚨 High Risk ({risk_prob:.1%})")

# Progress bar
st.progress(float(risk_prob))

st.divider()

# ── Patient summary ───────────────────────────────────────────
st.subheader("📊 Patient Profile Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")
    st.write(f"- Age: {age} years")
    st.write(f"- Gender: {gender}")
    st.write(f"- Race: {race}")

with col2:
    st.markdown("**Clinical Measurements**")
    st.write(f"- BMI: {bmi} kg/m²")
    st.write(f"- Blood Pressure: {bp} mmHg")
    st.write(f"- Cholesterol: {chol} mmol/L")
    st.write(f"- Sleep: {sleep} hours/night")
    st.write(f"- Physically Active: {phys_active}")

st.divider()

# ── Feature importance ────────────────────────────────────────
st.subheader("🔍 Key Risk Factors")

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

st.bar_chart(importance_df.set_index('Feature')['Importance'])

st.divider()

# ── About ─────────────────────────────────────────────────────
with st.expander("ℹ️ About this tool"):
    st.markdown("""
    **Model:** Random Forest Classifier
    
    **Features used:**
    - Age, BMI, Blood Pressure, Cholesterol
    - Physical Activity, Sleep Hours
    - Gender, Race
    
    **Training data:** Simulated clinical data based on 
    NHANES population statistics
    
    **Important note:** This is an educational tool built 
    as part of a data science portfolio. It should not be 
    used for clinical decision-making.
    """)