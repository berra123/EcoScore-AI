# 🌱 EcoScore AI

AI-powered Carbon Footprint Prediction and Green Manufacturing Decision Support System developed for Dentaş Kağıt Sanayi.

## 🚀 Live Demo

👉 https://ecoscore-ai-5yfy9atbmocuebtg6xtjul.streamlit.app

---

## Project Overview

EcoScore AI predicts the energy consumption and carbon footprint of production orders using Machine Learning models.

The application also provides:

- Carbon Footprint Estimation
- EcoScore Classification
- Green Production Scenario Comparison
- AI-based Green Manufacturing Recommendations using Google Gemini
- Interactive Streamlit Dashboard

---

## Features

- 🔥 Natural Gas Consumption Prediction
- ⚡ Electricity Consumption Prediction
- 🌍 Carbon Footprint Calculation
- 📊 Emission Comparison Charts
- 🤖 Gemini AI Recommendation System
- ♻️ EcoScore Classification

---

## Technologies

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Google Gemini API
- Joblib

---

## Machine Learning

Two XGBoost regression models were trained using synthetic manufacturing data.

Models:

- Natural Gas Consumption Model
- Electricity Consumption Model

Performance:

- Gas Model R² ≈ 0.98
- Electricity Model R² ≈ 0.97

---

## Project Structure

```text
EcoScore-AI/
│
├── app.py
├── requirements.txt
├── data/
├── models/
├── src/
│   ├── calculator.py
│   ├── llm_agent.py
│   └── model_train.py
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/berra123/EcoScore-AI.git

cd EcoScore-AI

pip install -r requirements.txt

streamlit run app.py
```

---

## Live Application

https://ecoscore-ai-5yfy9atbmocuebtg6xtjul.streamlit.app

---

Developed as an AI-powered Sustainable Manufacturing Decision Support System.