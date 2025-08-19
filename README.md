# 🩺 Diabetes Prediction App

A Machine Learning project to predict whether a person is likely to have diabetes based on medical details such as glucose level, BMI, age, etc.  
The app is built using **Python, Scikit-Learn, and Streamlit** and deployed on **Streamlit Cloud**.

---

## 🚀 Features
- Predicts diabetes risk using Random Forest Classifier.
- Shows probability (%) of having diabetes.
- Displays:
  - ✅ Prediction result (Diabetic / Not Diabetic)
  - 📊 Bar chart of probabilities
  - 🎯 Risk indicator (progress bar)

---

## 📂 Dataset
We use the **Pima Indians Diabetes Dataset** from Kaggle.  
It includes 768 records of female patients aged 21+, with the following features:

| Feature | Meaning | Normal Range | High Risk |
|---------|----------|--------------|-----------|
| Pregnancies | Number of times pregnant (0 for males) | 0–3 | > 6 |
| Glucose | Plasma glucose concentration (mg/dL) | < 110 | > 150 |
| Blood Pressure | Diastolic BP (mmHg) | ~80 | > 90 |
| Skin Thickness | Triceps skin fold (mm) | 10–30 | > 40 |
| Insulin | 2-hour serum insulin (µU/mL) | 15–276 | > 300 |
| BMI | Body Mass Index | 18.5–24.9 | ≥ 30 |
| Diabetes Pedigree Function | Genetic diabetes risk | 0.2–0.6 | > 1.0 |
| Age | Patient’s age (years) | < 40 | > 50 |

---

## ⚡ How to Run Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-app.git
   cd diabetes-prediction-app
