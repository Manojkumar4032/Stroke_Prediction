import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('stroke_prediction_model.sav')

# App title
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Stroke Risk Prediction")
st.markdown("Enter patient details to assess the risk of stroke.")

# Sidebar info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=100)
    st.markdown("## ℹ️ About")
    st.write(
        """
        This app uses a trained machine learning model to predict the risk of stroke based on user input.
        """
    )
    st.markdown("---")
    st.markdown("📍 Developed for Health Screening")

# Form layout using columns
with st.form("prediction_form"):
    st.markdown("### 📝 Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        age = st.slider("🎂 Age", 0, 100, 30)
        bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0)

    with col2:
        hypertension = st.selectbox("💉 Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("❤️ Heart Disease", ["No", "Yes"])
        residence_type = st.selectbox("🏡 Residence Type", ["Urban", "Rural"])

    work_type = st.selectbox("💼 Work Type", ["Private", "Self-employed", "Never_worked", "children"])
    smoking_status = st.selectbox("🚬 Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

# Prediction logic
if submitted:
    # Binary encoding
    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    residence_type = 1 if residence_type == "Urban" else 0

    # One-hot encoding for work_type
    work_type_dict = {
        "Never_worked": [1, 0, 0, 0],
        "Private": [0, 1, 0, 0],
        "Self-employed": [0, 0, 1, 0],
        "children": [0, 0, 0, 1],
    }
    work_type_encoded = work_type_dict[work_type]

    # One-hot encoding for smoking_status
    smoking_dict = {
        "formerly smoked": [1, 0, 0],
        "never smoked": [0, 1, 0],
        "smokes": [0, 0, 1],
    }
    smoking_encoded = smoking_dict[smoking_status]

    # Final input array
    input_data = np.array([
        gender, age, hypertension, heart_disease, bmi,
        *work_type_encoded,
        residence_type,
        *smoking_encoded
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    st.markdown("---")
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Stroke!**\n\nRisk Score: **{probability:.2f}**")
    else:
        st.success(f"✅ **Low Risk of Stroke**\n\nRisk Score: **{probability:.2f}**")

    st.markdown("---")
    st.markdown("🧬 _This is a prediction tool, not a medical diagnosis._ Please consult a doctor for clinical advice.")
