import streamlit as st
import numpy as np
import  joblib

# Load model
model = joblib.load("gridSCV_salPred.pkl")
scaler = joblib.load("scaler_sal.pkl")

st.title("💼 Salary Prediction App")

st.write("Enter details to predict salary")

# Inputs
age = st.slider("Age", 18, 60)
experience = st.slider("Experience (Years)", 0, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
degree = st.selectbox("Degree", ["Bachelors", "Masters", "PhD"])
job = st.selectbox("Job Title", ["Engineer", "Manager", "Analyst"])

# Encoding manually (same as training)
gender_val = 1 if gender == "Male" else 0

degree_map = {"Bachelors":0, "Masters":1, "PhD":2}
job_map = {"Engineer":0, "Manager":1, "Analyst":2}

degree_val = degree_map[degree]
job_val = job_map[job]

# Prediction
if st.button("Predict Salary"):
    data = np.array([[age, gender_val, degree_val, job_val, experience]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.success(f"💰 Predicted Salary: ₹{int(prediction[0])}")