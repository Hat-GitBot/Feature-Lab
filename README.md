# 🔬 Feature Engineering & Model Benchmark Lab

An interactive Streamlit web application for rapid machine learning experimentation.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## 🎯 Features
- Upload CSV or generate sample datasets
- Feature Engineering: scaling, encoding, polynomials, date extraction
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Visualizations: ROC curves, confusion matrices, feature importance, SHAP plots
- Experiment tracking with SQLite

## 🛠️ Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/feature-lab.git
cd feature-lab
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Tech Stack
- Streamlit · Plotly · pandas · scikit-learn · XGBoost · LightGBM · SHAP

## 📁 Structure
```
feature-lab/
├── app.py               ← Main Streamlit app
├── requirements.txt     ← Python dependencies
├── .streamlit/
│   └── config.toml      ← Theme & server config
└── src/
    ├── data_utils.py    ← Preprocessing utilities
    ├── modeling.py      ← ML model training
    ├── plotting.py      ← Plotly visualizations
    └── database.py      ← SQLite experiment tracking
```

## 🌐 Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your repo → set `app.py` → Deploy
