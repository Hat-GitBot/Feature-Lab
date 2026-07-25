
# 🔬 Feature Engineering & Model Benchmark Lab

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://feature-lab.streamlit.app/) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-0086D4)](https://xgboost.readthedocs.io/) [![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-2D9E6B)](https://lightgbm.readthedocs.io/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An interactive end-to-end machine learning experimentation platform built for data scientists.**  
Upload any CSV, engineer features, benchmark models, and interpret results — all in one place.

[🚀 **Live Demo**](https://feature-lab.streamlit.app/) · [📖 Report Bug](https://github.com/Hat-GitBot/Feature-Lab/issues) 

## 📋 Table of Contents

-   Overview
-   Live Demo
-   Features
-   Tech Stack
-   Project Structure
-   Getting Started
    -   Prerequisites
    -   Local Installation
    -   Google Colab
-   Usage Guide
-   Models Supported
-   Feature Engineering Options
-   Experiment Tracking
-   Contributing

----------

## 🧠 Overview

**Feature Engineering & Model Benchmark Lab** is a production-ready Streamlit application designed for rapid ML prototyping. It removes the friction between raw data and actionable model results — letting you focus on experimentation rather than boilerplate code.

Whether you're a data scientist validating a new dataset, a researcher benchmarking algorithms, or a student learning the ML workflow, this tool provides an interactive, code-free environment to:

-   Upload any CSV and instantly profile your data
-   Apply a full suite of feature engineering transformations
-   Train and compare multiple ML models side by side
-   Visualize results with interactive Plotly charts
-   Interpret predictions with SHAP explainability
-   Save and revisit experiments over time

----------

## 🚀 Live Demo

**Try it now — no installation required:**

> 🔗 **[https://feature-lab.streamlit.app/](https://feature-lab.streamlit.app/)**

The live demo includes pre-loaded sample datasets for classification and regression tasks so you can explore all features immediately.

----------

## ✨ Features

### 📊 Data Management

-   **CSV Upload** — drag-and-drop any CSV file (up to 200 MB)
-   **Sample Data Generator** — built-in classification and regression datasets
-   **Data Validation** — automatic detection of missing values, data types, cardinality, and memory usage
-   **Data Preview** — interactive table with type-aware rendering

### 🛠️ Feature Engineering Pipeline

**Missing Values** : Drop rows, or fill with mean / median / mode

**Standard Scaling** : Zero mean, unit variance (StandardScaler)

**Min-Max Scaling** : Scale to [0, 1] range (MinMaxScaler)

**One-Hot Encoding** : Expand categoricals into binary columns

**Label Encoding** : Ordinal integer mapping for categoricals

**Polynomial Features** : Degree-2 or degree-3 interaction terms

**DateTime Extraction** : Year, month, day, weekday, quarter, is_weekend

All steps are applied deterministically and tracked — the pipeline can be reset or replayed at any time.

### 🤖 Model Training

-   Train multiple models simultaneously with a progress bar
-   Configurable train/test split ratio and random seed
-   Automatic numeric enforcement and schema validation before training
-   Training time recorded per model

### 📈 Evaluation & Visualization

**Classification:**

-   Accuracy, Precision, Recall, F1 Score, ROC AUC
-   Interactive Confusion Matrix (count + percentage)
-   ROC Curve and Precision-Recall Curve
-   Feature Importance bar chart

**Regression:**

-   RMSE, MAE, R² Score
-   Actual vs. Predicted scatter plot
-   Residual analysis plot
-   Feature Importance bar chart

### 🔍 SHAP Explainability

-   **SHAP Summary Plot** — global feature importance ranked by mean |SHAP value|
-   **SHAP Waterfall Plot** — per-prediction breakdown showing how each feature contributes
-   Instance selector slider to explain any prediction in the test set
-   Supports TreeExplainer for tree-based models (fast) and KernelExplainer for linear models

### 💾 Experiment Tracking

-   Optional SQLite-backed experiment history
-   Stores preprocessing steps, model metrics, and feature importance per run
-   Auto-deduplicates experiment names
-   Delete individual experiments
-   Relative timestamp display (e.g. "3 min ago")

----------

## 🧰 Tech Stack

**Frontend / UI** : [Streamlit](https://streamlit.io/)

**Visualizations** : [Plotly](https://plotly.com/python/)

**Data Processing** : [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

**ML — Core** : [scikit-learn](https://scikit-learn.org/)

**ML — Boosting** : [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/)

**Explainability** : [SHAP](https://shap.readthedocs.io/)

**Persistence** : SQLite (via Python `sqlite3`)

**Deployment** : Streamlit Cloud

----------

## 📁 Project Structure

```
feature-lab/
│
├── app.py                      # Main Streamlit application (all 6 pages)
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── .streamlit/
│   └── config.toml             # Streamlit theme & server settings
│
├── src/                        # Core modules
│   ├── data_utils.py           # Data loading, validation, feature engineering
│   ├── modeling.py             # Model training, evaluation, SHAP generation
│   ├── plotting.py             # All Plotly chart functions (incl. SHAP plots)
│   └── database.py             # SQLite experiment CRUD operations
│
├── assets/                     # Static assets (images, icons)
└── sample_data/                # Sample CSV files for testing

```

----------

## 🏁 Getting Started

### Prerequisites

-   Python 3.8 or higher
-   pip

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/Hat-GitBot/Feature-Lab.git
cd feature-lab

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py

```

The app opens automatically at **http://localhost:8501**

----------

### Google Colab

Run the entire app in the cloud without any local setup:

```python
# Cell 1 — Install dependencies
!pip install streamlit plotly pandas numpy scikit-learn xgboost lightgbm shap -q

# Cell 2 — Clone repo (replace with your GitHub URL)
!git clone https://github.com/Hat-GitBot/Feature-Lab.git
%cd feature-lab

# Cell 3 — Launch Streamlit
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Cell 4 — Get the public URL
from google.colab.output import eval_js
import time
time.sleep(3)
print(eval_js("google.colab.kernel.proxyPort(8501)"))

```

----------

## 📖 Usage Guide

The app is organized into **6 pages** accessible from the sidebar:

### 🏠 Home

-   Overview of features and quick start guide
-   Generate sample **Classification** or **Regression** datasets with a single click
-   Instant dataset preview and metrics displayed in the main area

### 📊 Data Upload

-   Upload your own CSV file
-   Automatic data validation: row/column counts, missing value detection, type inference
-   Configure **task type** (Classification or Regression) and **target column**
-   View target distribution chart

### 🛠️ Feature Engineering

Five tabs, each covering a different transformation step:

🧹 Preprocessing : Handle missing values
🔢 Scaling : Normalize numeric features
🏷️ Encoding : Encode categorical variables
✖️ Interactions : Create polynomial / interaction features
📅 Date Features : Extract components from datetime columns

After each step, an expandable **dataset preview** shows exactly what changed, and an animated **"Next Step"** prompt guides you to the next tab.

### 🤖 Model Training

1.  Adjust the train/test split slider (default 80/20)
2.  Set a random seed for reproducibility
3.  Select one or more models from the list
4.  Click **🚀 Train Models** — a progress bar tracks each model
5.  View the summary table and grouped bar chart comparing all models

### 📈 Evaluation

Select any trained model from the dropdown to see:

-   Full metric scorecard
-   Classification: Confusion Matrix, ROC Curve, PR Curve
-   Regression: Actual vs. Predicted, Residuals
-   Feature Importance ranked bar chart
-   **SHAP Explanations** (check the box):
    -   SHAP Summary: which features matter most globally
    -   SHAP Waterfall: why the model made a specific prediction

### 💾 Experiments

_(Enable "Experiment Tracking" in the sidebar first)_

-   Name and save the current run to SQLite
-   Browse all saved experiments in a table with relative timestamps
-   Load any past experiment to inspect its preprocessing pipeline and model results
-   Delete one or more experiments

----------

## 🤖 Models Supported

### Classification

Logistic Regression : scikit-learn, `max_iter=1000`

Random Forest : scikit-learn, 100 estimators, parallel jobs

XGBoost : xgboost, `eval_metric='logloss'`

LightGBM : lightgbm, `verbose=-1` (silent)

### Regression

Linear Regression : scikit-learn, OLS baseline

Ridge Regression : scikit-learn, L2 regularization

Lasso Regression : scikit-learn, L1 regularization

Random Forest : scikit-learn, 100 estimators

XGBoost : xgboost, Gradient boosting

LightGBM : lightgbm, Fast gradient boosting

----------

## 🛠️ Feature Engineering Options

### Scaling Methods

-   **StandardScaler** — transforms to zero mean and unit variance; best for models sensitive to feature magnitude (e.g., logistic regression)
-   **MinMaxScaler** — scales to [0, 1]; good when bounded range is needed

### Encoding Methods

-   **One-Hot Encoding** — creates a binary column per category (drop_first=True to avoid multicollinearity)
-   **Label Encoding** — maps categories to integers; use for tree-based models

### Polynomial Features

-   Generates all degree-N interaction terms (e.g. x₁², x₁·x₂, x₂²)
-   Column names are sanitized for XGBoost compatibility
-   Degrees 2 and 3 supported

### DateTime Extraction

Automatically extracts from any datetime or date-like string column: `year`, `month`, `day`, `dayofweek`, `quarter`, `is_weekend`

----------

## 💾 Experiment Tracking

When **Experiment Tracking** is enabled (sidebar toggle), every run is saved to a local `experiments.db` SQLite file containing:

-   Experiment name, task type, timestamp
-   Dataset metadata (rows, columns, target column)
-   Full preprocessing pipeline (step type + config)
-   Per-model metrics and feature importance

Experiments are auto-renamed if the name already exists (e.g. `my_exp (1)`, `my_exp (2)`).

> **Note:** `experiments.db` is listed in `.gitignore` — it stays local and is never committed to GitHub.

## 🔧 Troubleshooting

`ModuleNotFoundError` : Run `pip install -r requirements.txt` again

`Port 8501 already in use` : `pkill streamlit` then restart

SHAP taking too long : Reduce dataset size or use Random Forest

Encoding error before training : Encode all categorical columns before clicking Train

`experiments.db` locked : Close other connections; restart the app

Colab URL not working : Re-run the `eval_js` cell after a 3-second wait

----------

## 🤝 Contributing

Contributions are welcome! To contribute:

1.  Fork this repository
2.  Create a feature branch: `git checkout -b feature/your-feature-name`
3.  Commit your changes: `git commit -m "Add: your feature description"`
4.  Push to your fork: `git push origin feature/your-feature-name`
5.  Open a Pull Request

### Ideas for Contributions

-   Add neural network support (TensorFlow / PyTorch)
-   Hyperparameter tuning (Optuna integration)
-   Cross-validation support
-   PDF / CSV report export
-   Multi-file dataset merging

<div align="center">

**Built with ❤️ using Streamlit**

[🔗 Live App](https://feature-lab.streamlit.app/) · [⭐ Star on GitHub](https://github.com/Hat-GitBot/Feature-Lab) · [🐛 Report Issues](https://github.com/Hat-GitBot/Feature-Lab/issues)

</div>
