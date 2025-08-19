import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load saved model & scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🩺 Diabetes Prediction App")
st.write("Enter the details below to check if a person is likely to have diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of having diabetes

    # Text output
    if prediction == 1:
        st.error(f"⚠️ The person is likely to have **Diabetes**.\n\n🔹 Probability: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ The person is **Not likely diabetic**.\n\n🔹 Probability: **{probability*100:.2f}%**")

    # -----------------------------
    # Probability Bar Chart
    # -----------------------------
    st.subheader("📊 Prediction Probability")
    probs = model.predict_proba(input_scaled)[0]
    labels = ["Not Diabetic", "Diabetic"]

    fig, ax = plt.subplots()
    ax.bar(labels, probs, color=["green", "red"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v*100:.2f}%", ha="center", fontweight="bold")
    st.pyplot(fig)

    # -----------------------------
    # Gauge Meter (Progress Bar)
    # -----------------------------
    st.subheader("🎯 Risk Level Indicator")
    st.progress(int(probability*100))
