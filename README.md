# 🩺 Diabetes Risk Prediction App
## Interactive ML-powered Clinical Decision Support Tool

## 🔗 Live App
👉 **[Open App](https://app-diabetes-predictor-phuzoiujjf3evpsekifqdw.streamlit.app/)**

---

## 📌 Overview

Interactive web application for diabetes risk prediction built with 
Streamlit and Random Forest. Users can input clinical measurements 
and receive an instant risk assessment.

This project demonstrates the ability to **deploy machine learning 
models as user-facing applications** — bridging the gap between 
data science and real-world clinical use.

---

## ⚠️ Disclaimer

This tool is for **educational and portfolio purposes only** and 
should not be used for clinical decision-making. Always consult 
a qualified healthcare professional.

---

## ✨ Features

- 🎛️ **Interactive sliders** for clinical input
- 📊 **Real-time risk prediction** with probability score
- 🚦 **Risk classification** — Low / Moderate / High
- 🔍 **Feature importance** visualization
- 📋 **Patient profile summary**

---

## 🤖 Model

- **Algorithm:** Random Forest Classifier
- **Features:** Age, BMI, Blood Pressure, Cholesterol, 
Physical Activity, Sleep Hours, Gender, Race
- **Class weights:** Adjusted for clinical context 
(minimize false negatives)
- **Training data:** Simulated clinical data based on 
NHANES population statistics and known diabetes risk factors

---

## 🏗️ How It Works
```
User inputs clinical values
        ↓
StandardScaler normalizes inputs
        ↓
Random Forest predicts probability
        ↓
App displays risk level + recommendations
```

---

## 🛠️ Tech Stack

- Python 3.11
- `streamlit` — web application
- `scikit-learn` — Random Forest model
- `pandas`, `numpy` — data processing
- Deployed on **Streamlit Community Cloud**

## 📁 Files

- `app.py` — Complete Streamlit application
- `requirements.txt` — Package dependencies

---

## 💼 Why This Project Matters

✔ End-to-end ML deployment
✔ User-facing clinical tool
✔ Real-time predictions
✔ Responsible AI — clear disclaimers
✔ Demonstrates Python + ML + deployment skills

---

## 👩‍💻 Author

**Noelia Caba Martin**
Junior Data Scientist / Bioinformatician
Interested in health data, bioinformatics and applied ML

⭐ Feel free to star the repo or connect!
