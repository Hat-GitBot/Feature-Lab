
"""
Feature Engineering & Model Benchmark Lab - COMPLETE FIXED VERSION
ALL 6 ISSUES RESOLVED + SHAP VISUALIZATIONS
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys
import os
import re

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_utils import (
    validate_dataset, handle_missing_values, encode_categorical,
    scale_features, create_polynomial_features, create_interactions,
    extract_datetime_features, prepare_train_test_split,
    generate_sample_classification_data, generate_sample_regression_data
)
from src.modeling import (
    get_classification_models, get_regression_models,
    train_model, evaluate_classification_model, evaluate_regression_model,
    get_feature_importance, explain_prediction_shap, calculate_model_metrics_summary
)
from src.plotting import (
    plot_missing_values, plot_correlation_matrix, plot_feature_distributions,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance, plot_actual_vs_predicted, plot_residuals,
    plot_model_comparison, plot_shap_summary, plot_shap_waterfall
)
from src.database import ExperimentDB

# Page configuration
st.set_page_config(
    page_title="Feature Engineering & Model Benchmark Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIX #1 & #2: Custom CSS with proper contrast and no broken images
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }

    /* FIX #1: Proper contrast - dark gradient background with white text */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-left: 5px solid #4a5568;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .info-box h3 {
        color: white !important;
        margin-top: 0;
        font-size: 1.5rem;
    }

    .info-box p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }

    /* FIX #4: Next step indicator styling */
    .next-step {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        color: white !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    /* FIX #2: Sidebar branding - professional header instead of broken image */
    .sidebar-brand {
        text-align: center;
        padding: 1rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }

    .sidebar-brand h2 {
        color: white;
        margin: 0;
        font-size: 1.4rem;
        font-weight: bold;
    }

    .sidebar-brand p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 0.85rem;
    }

    .step-badge {
        display: inline-block;
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

MAX_STYLE_CELLS = 200_000

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    import re
    df = df.copy()
    df.columns = [
        re.sub(r'_+', '_', re.sub(r'[^\w]', '_', col)).strip('_').lower()
        for col in df.columns
    ]
    return df

def get_changed_mask(df_original, df_processed):
    # Align both DataFrames (VERY IMPORTANT)
    df1, df2 = df_original.align(df_processed, join="outer", axis=1)

    # Fill NaNs so comparison works
    df1 = df1.fillna("__MISSING__")
    df2 = df2.fillna("__MISSING__")

    return df1 != df2

def highlight_changes(df_original, df_processed):
    def style_fn(val, orig_val):
        if pd.isna(orig_val) and not pd.isna(val):
            return "background-color: #1f4d8b"  # new column → blue
        elif val != orig_val:
            return "background-color: #144d2a"  # changed → green
        return ""

    return df_processed.style.apply(
        lambda col: [
            style_fn(
                val,
                df_original.iloc[i].get(col.name, np.nan)  # ✅ SAFE ACCESS
            )
            for i, val in enumerate(col)
        ],
        axis=0
    )

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data_loaded': False,
        'df': None,
        'df_processed': None,
        'df_original': None,
        'task_type': None,
        'target_column': None,
        'model_results': {},
        'preprocessing_steps': [],
        'db_enabled': False,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# FIX #4: Helper function to show next step messages
def show_next_step(message):
    """Display animated next step indicator"""
    st.markdown(f'<div class="next-step">✨ {message}</div>', unsafe_allow_html=True)

# FIX #5: Helper function to show dataset preview after transformations
def show_dataset_preview(df, title="📊 Dataset Preview"):
    """Show dataset with metrics"""
    with st.expander(title, expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        st.dataframe(df.head(100), use_container_width=True, height=300)

# FIX #2: Sidebar with proper branding (no broken image)
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🔬 ML Experiment Lab</h2>
        <p>Feature Engineering & Model Benchmarking</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Upload", "🛠️ Feature Engineering",
         "🤖 Model Training", "📈 Evaluation", "💾 Experiments"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Database toggle
    st.session_state.db_enabled = st.checkbox(
        "Enable Experiment Tracking",
        value=st.session_state.db_enabled,
        help="Save experiments to SQLite database"
    )

    # Utility buttons
    if st.session_state.data_loaded:
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            st.session_state.df_processed = None
            st.session_state.preprocessing_steps = []
            st.session_state.model_results = {}
            st.success("Pipeline reset!")
            st.rerun()

        if st.button("💥 Reset EVERYTHING", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Main content
st.markdown('<div class="main-header">🔬 Feature Engineering & Model Benchmark Lab</div>', unsafe_allow_html=True)

# ===== HOME PAGE =====
if page == "🏠 Home":
    # FIX #1: Info box now has proper contrast
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the ML Experiment Lab! 🚀</h3>
    <p>An interactive platform for rapid ML prototyping and model benchmarking designed for data scientists.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - 📤 **Upload CSV** or generate sample datasets
        - 🛠️ **Feature Engineering**: scaling, encoding, interactions, polynomials
        - 🤖 **Multiple Models**: Logistic, Random Forest, XGBoost, LightGBM
        - 📊 **Rich Visualizations**: ROC, PR curves, feature importance, SHAP plots
        - 💾 **Experiment Tracking**: Save and compare experiments
        - 🔍 **SHAP Explanations**: Full model interpretability with visualizations
        """)

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Upload Data** → Go to 📊 Data Upload
        2. **Select Task** → Classification or Regression
        3. **Engineer Features** → Apply transformations
        4. **Train Models** → Run multiple algorithms
        5. **Evaluate** → Compare performance with SHAP
        6. **Save** → Track experiments (optional)
        """)

    st.markdown("---")

    if "data_loaded" not in st.session_state:
      st.session_state.data_loaded = False

    # FIX #3: Sample data generators now show info in MAIN AREA
    st.markdown('<div class="section-header">🎲 Generate Sample Data</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        n_samples = st.number_input("Number of samples", 100, 10000, 1000, 100)

    with col2:
        if st.button("📊 Classification Data", use_container_width=True, type="primary"):
            with st.spinner("Generating classification dataset..."):
                st.session_state.df = generate_sample_classification_data(n_samples)
                st.session_state.df = clean_column_names(st.session_state.df)
                st.session_state.df_original = st.session_state.df.copy()
                st.session_state.data_loaded = True
                st.session_state.task_type = 'classification'
                st.session_state.target_column = 'target'

    with col3:
        if st.button("📈 Regression Data", use_container_width=True, type="primary"):
            with st.spinner("Generating regression dataset..."):
                st.session_state.df = generate_sample_regression_data(n_samples)
                st.session_state.df = clean_column_names(st.session_state.df)
                st.session_state.df_original = st.session_state.df.copy()
                st.session_state.data_loaded = True
                st.session_state.task_type = 'regression'
                st.session_state.target_column = 'target'

    # ===== FULL WIDTH RENDER AREA =====
    if st.session_state.get("data_loaded", False):

        st.success(f"✅ {st.session_state.task_type.capitalize()} dataset generated!")
        st.markdown("### 📊 Generated Dataset Info")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rows", f"{len(st.session_state.df):,}")
        col_b.metric("Columns", len(st.session_state.df.columns))
        col_c.metric("Task", st.session_state.task_type.capitalize())
        col_d.metric("Target", st.session_state.target_column)

        # ✅ FULL WIDTH NOW
        show_dataset_preview(st.session_state.df, "View Generated Data")
        show_next_step("Next Step: Go to 📊 Data Upload to configure or proceed to 🛠️ Feature Engineering")

# ===== DATA UPLOAD PAGE =====
elif page == "📊 Data Upload":
    st.markdown('<div class="section-header">📊 Data Upload & Validation</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=['csv'],
        help="Upload a CSV file with your dataset"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = clean_column_names(df)
            st.session_state.df = df
            st.session_state.df_original = df.copy()
            st.session_state.data_loaded = True

            st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")

            # Dataset preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Validation
            st.markdown("### 🔍 Data Validation")
            diagnostics = validate_dataset(df)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{diagnostics['n_rows']:,}")
            col2.metric("Total Columns", diagnostics['n_cols'])
            col3.metric("Numeric Columns", len(diagnostics['numeric_cols']))
            col4.metric("Categorical Columns", len(diagnostics['categorical_cols']))

            # Missing values
            if sum(diagnostics['missing_values'].values()) > 0:
                st.warning("⚠️ Missing values detected!")
                fig = plot_missing_values(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

            # Task type selection
            st.markdown("### 🎯 Task Configuration")

            col1, col2 = st.columns(2)

            with col1:
                task_type = st.selectbox(
                    "Select task type",
                    ["classification", "regression"],
                    index=0 if st.session_state.task_type == 'classification' else 1
                    if st.session_state.task_type else 0
                )
                st.session_state.task_type = task_type

            with col2:
                target_col = st.selectbox(
                    "Select target column",
                    df.columns.tolist(),
                    index=df.columns.tolist().index(st.session_state.target_column)
                    if st.session_state.target_column in df.columns else 0
                )
                st.session_state.target_column = target_col

            # Target distribution
            st.markdown("### 📊 Target Distribution")
            if task_type == 'classification':
                target_counts = df[target_col].value_counts()
                st.bar_chart(target_counts)
            else:
                st.line_chart(df[target_col].head(100))

            # FIX #4: Next step message
            show_next_step("Next Step: Go to 🛠️ Feature Engineering to transform your data")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    elif st.session_state.data_loaded:
        st.info("📁 Dataset already loaded. Upload a new file to replace it.")
        show_dataset_preview(st.session_state.df)
        show_next_step("Next Step: Go to 🛠️ Feature Engineering")

# ===== FEATURE ENGINEERING PAGE =====
elif page == "🛠️ Feature Engineering":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
        st.stop()

    st.markdown('<div class="section-header">🛠️ Feature Engineering</div>', unsafe_allow_html=True)

    # Use processed data if available, otherwise original
    if st.session_state.df_processed is None:
        st.session_state.df_processed = st.session_state.df.copy()

    df = st.session_state.df_processed.copy()

    # Create tabs for different feature engineering steps
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Preprocessing", "🔢 Scaling", "🏷️ Encoding",
        "✖️ Interactions", "📅 Date Features"
    ])

    # Tab 1: Preprocessing
    with tab1:
        st.markdown("### 🧹 Handle Missing Values")

        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            st.warning(f"⚠️ Found {missing_count} missing values")

            strategy = st.selectbox(
                "Missing value strategy",
                ["drop", "mean", "median", "mode"]
            )

            if st.button("Apply Missing Value Handler", type="primary"):
                df = handle_missing_values(df, strategy)
                st.session_state.df_processed = df
                st.session_state.preprocessing_steps.append({
                    'type': 'missing_values',
                    'config': {'strategy': strategy}
                })
                st.success(f"✅ Applied {strategy} strategy")

                # FIX #5: Show dataset preview after transformation
                show_dataset_preview(df, "📊 Dataset After Missing Value Handling")

                # FIX #4: Show next step
                show_next_step("Next Step: Scaling - Move to the Scaling tab to scale numeric features")
                st.rerun()
        else:
            st.success("✅ No missing values!")
            show_next_step("Next Step: Scaling - Move to the Scaling tab")

    # Tab 2: Scaling
    with tab2:
        st.markdown("### 🔢 Feature Scaling")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if st.session_state.target_column in numeric_cols:
            numeric_cols.remove(st.session_state.target_column)

        if numeric_cols:
            cols_to_scale = st.multiselect(
                "Select columns to scale",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )

            scale_method = st.radio(
                "Scaling method",
                ["standard", "minmax"],
                horizontal=True
            )

            if st.button("Apply Scaling", type="primary") and cols_to_scale:
                st.session_state.df_before_scaling = df.copy()  # ✅ STORE BEFORE
                df = scale_features(df, cols_to_scale, scale_method)
                st.session_state.df_processed = df
                st.session_state.scaling_applied = True

                # store metadata safely
                st.session_state.scaled_columns = cols_to_scale
                st.session_state.scaling_method = scale_method
                st.session_state.preprocessing_steps.append({
                    'type': 'scaling',
                    'config': {'columns': cols_to_scale, 'method': scale_method}
                })

            if st.session_state.get("scaling_applied", False):
                st.success(
                    f"✅ Scaled {len(st.session_state.scaled_columns)} features using {st.session_state.scaling_method}"
                )

                df_before = st.session_state.df_before_scaling
                df_after = st.session_state.df_processed

                st.markdown("### 🔍 Data Changes (Before vs After)")

                # ===== FILTER CHANGED ROWS =====
                change_mask = get_changed_mask(df_before, df_after)
                changed_rows = change_mask.any(axis=1)

                df_before_changed = df_before[changed_rows]
                df_after_changed = df_after[changed_rows]

                # ===== SIDE BY SIDE VIEW =====
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🟥 Before")
                    st.dataframe(df_before_changed, use_container_width=True)

                with col2:
                    st.markdown("#### 🟩 After (Highlighted)")
                    df_after_filtered = df_after[changed_rows]
                    df_before_filtered = df_before[changed_rows]

                    styled_df = highlight_changes(df_before_filtered, df_after_filtered)
                    st.dataframe(styled_df, use_container_width=True)

                show_next_step("Next Step: Encoding - Move to the Encoding tab to encode categorical variables")

        else:
            st.info("No numeric columns available for scaling")
            show_next_step("Next Step: Encoding - Move to the Encoding tab")

    # Tab 3: Encoding
    with tab3:
        st.markdown("### 🏷️ Categorical Encoding")

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if cat_cols:
            cols_to_encode = st.multiselect(
                "Select columns to encode",
                cat_cols,
                default=cat_cols[:min(3, len(cat_cols))]
            )

            encode_method = st.radio(
                "Encoding method",
                ["onehot", "label"],
                horizontal=True,
                help="One-hot: Creates binary columns. Label: Converts to integers."
            )

            if st.button("Apply Encoding", type="primary") and cols_to_encode:
              st.session_state.df_before_encoding = df.copy()

              df = encode_categorical(df, cols_to_encode, encode_method)
              st.session_state.df_processed = df
              st.session_state.encoding_applied = True
              st.session_state.encoded_columns = cols_to_encode
              st.session_state.encoding_method = encode_method

              st.session_state.preprocessing_steps.append({
                  'type': 'encoding',
                  'config': {'columns': cols_to_encode, 'method': encode_method}
              })

            if st.session_state.get("encoding_applied", False):
              st.success(f"✅ Encoded {len(st.session_state.encoded_columns)} features using {st.session_state.encoding_method}")

              df_before = st.session_state.df_before_encoding
              df_after = st.session_state.df_processed
              st.markdown("### 🔍 Encoding Changes")
              change_mask = get_changed_mask(df_before, df_after)

              changed_rows = change_mask.any(axis=1)
              changed_cols = change_mask.any(axis=0)

              df_before_aligned, df_after_aligned = df_before.align(df_after, join="outer", axis=1)
              df_before_filtered = df_before_aligned.loc[changed_rows, changed_cols]
              df_after_filtered = df_after_aligned.loc[changed_rows, changed_cols]

              added_cols = set(df_after.columns) - set(df_before.columns)
              removed_cols = set(df_before.columns) - set(df_after.columns)

              if added_cols:
                  st.info(f"🟢 Added columns: {list(added_cols)}")

              if removed_cols:
                  st.warning(f"🔴 Removed columns: {list(removed_cols)}")

              col1, col2 = st.columns(2)

              with col1:
                  st.markdown("#### 🟥 Before")
                  st.dataframe(df_before_filtered, use_container_width=True)

              with col2:
                  st.markdown("#### 🟩 After (Highlighted)")
                  styled_df = highlight_changes(df_before_filtered, df_after_filtered)
                  st.dataframe(styled_df, use_container_width=True)

              show_next_step("Next Step: Interactions - Create polynomial features (optional) or proceed to 🤖 Model Training")

        else:
            st.info("No categorical columns to encode")
            show_next_step("Next Step: Go to 🤖 Model Training")

    # Tab 4: Interactions
    with tab4:
        st.markdown("### ✖️ Feature Interactions")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if st.session_state.target_column in numeric_cols:
            numeric_cols.remove(st.session_state.target_column)

        if len(numeric_cols) >= 2:
            st.markdown("**Create Polynomial Features**")
            poly_cols = st.multiselect(
                "Select columns for polynomial features",
                numeric_cols,
                key="poly_cols"
            )
            poly_degree = st.slider("Polynomial degree", 2, 3, 2)

            if st.button("Create Polynomial Features", type="primary") and poly_cols:
              st.session_state.df_before_poly = df.copy()
              df = create_polynomial_features(df, poly_cols, poly_degree)
              st.session_state.df_processed = df
              st.session_state.poly_applied = True

              st.session_state.poly_columns = poly_cols
              st.session_state.poly_degree = poly_degree
              st.session_state.preprocessing_steps.append({
                  'type': 'polynomial',
                  'config': {'columns': poly_cols, 'degree': poly_degree}})

            if st.session_state.get("poly_applied", False):
              st.success(f"✅ Created polynomial features (degree {st.session_state.poly_degree})")

              df_before = st.session_state.df_before_poly
              df_after = st.session_state.df_processed

              st.markdown("### 🔍 Polynomial Feature Changes")
              change_mask = get_changed_mask(df_before, df_after)

              changed_rows = change_mask.any(axis=1)
              changed_cols = change_mask.any(axis=0)

              df_before_aligned, df_after_aligned = df_before.align(df_after, join="outer", axis=1)
              df_before_filtered = df_before_aligned.loc[changed_rows, changed_cols]
              df_after_filtered = df_after_aligned.loc[changed_rows, changed_cols]
              #df_after_filtered = df_after.loc[changed_rows, changed_cols]

              col1, col2 = st.columns(2)

              with col1:
                  st.markdown("#### 🟥 Before")
                  st.dataframe(df_before_filtered, use_container_width=True)

              with col2:
                  st.markdown("#### 🟩 After (Highlighted)")
                  rows, cols = df_after_filtered.shape
                  total_cells = rows * cols

                  if total_cells <= MAX_STYLE_CELLS:
                      # ✅ ONLY create Styler here
                      styled_df = highlight_changes(df_before_filtered, df_after_filtered)
                      st.dataframe(styled_df, use_container_width=True)

                  else:
                      st.warning(f"⚠️ Large dataset detected ({total_cells:,} cells). Showing sampled diff.")

                      # ===== SAMPLE ROWS =====
                      sample_rows = min(500, len(df_after_filtered))
                      row_idx = df_after_filtered.sample(sample_rows, random_state=42).index

                      # ===== SAMPLE COLUMNS (CRITICAL FIX) =====
                      sample_cols = min(50, len(df_after_filtered.columns))
                      col_idx = df_after_filtered.columns[:sample_cols]  # deterministic (better UX)

                      df_before_sample = df_before_filtered.loc[row_idx, col_idx].reset_index(drop=True)
                      df_after_sample = df_after_filtered.loc[row_idx, col_idx].reset_index(drop=True)

                      # ✅ SAFE: small dataset now
                      styled_df = highlight_changes(df_before_sample, df_after_sample)
                      st.dataframe(styled_df, use_container_width=True)

    # Tab 5: Date Features
    with tab5:
        st.markdown("### 📅 Date Feature Extraction")

        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Check for string columns that might be dates
        potential_date_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col].head(), errors='raise')
                potential_date_cols.append(col)
            except:
                pass

        all_date_cols = date_cols + potential_date_cols

        if all_date_cols:
            cols_to_extract = st.multiselect(
                "Select date columns",
                all_date_cols
            )

            if st.button("Extract Date Features", type="primary") and cols_to_extract:
              st.session_state.df_before_date = df.copy()
              df = extract_datetime_features(df, cols_to_extract)

              st.session_state.df_processed = df
              st.session_state.date_applied = True

              st.session_state.date_columns = cols_to_extract
              st.session_state.preprocessing_steps.append({
                  'type': 'datetime',
                  'config': {'columns': cols_to_extract}
              })

            if st.session_state.get("date_applied", False):
              st.success(f"✅ Extracted date features from {len(st.session_state.date_columns)} columns")

              df_before = st.session_state.df_before_date
              df_after = st.session_state.df_processed

              st.markdown("### 🔍 Date Feature Changes")

              change_mask = get_changed_mask(df_before, df_after)

              changed_rows = change_mask.any(axis=1)
              changed_cols = change_mask.any(axis=0)

              df_before_aligned, df_after_aligned = df_before.align(df_after, join="outer", axis=1)

              df_before_filtered = df_before_aligned.loc[changed_rows, changed_cols]
              df_after_filtered = df_after_aligned.loc[changed_rows, changed_cols]

              col1, col2 = st.columns(2)

              with col1:
                  st.markdown("#### 🟥 Before")
                  st.dataframe(df_before_filtered, use_container_width=True)

              with col2:
                  st.markdown("#### 🟩 After (Highlighted)")
                  styled_df = highlight_changes(df_before_filtered, df_after_filtered)
                  st.dataframe(styled_df, use_container_width=True)

              show_next_step("Next Step: 🤖 Model Training")

        else:
            st.info("No date columns detected")
            show_next_step("Next Step: Go to 🤖 Model Training")

    # Show processed data summary
    if st.session_state.df_processed is not None:
        st.markdown("---")
        st.markdown("### 📊 Processed Data Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Features", len(st.session_state.df.columns))
        col2.metric("Current Features", len(st.session_state.df_processed.columns))
        col3.metric("Preprocessing Steps", len(st.session_state.preprocessing_steps))

        with st.expander("View Applied Steps"):
            for i, step in enumerate(st.session_state.preprocessing_steps, 1):
                st.markdown(f'<span class="step-badge">Step {i}</span> **{step["type"].replace("_", " ").title()}**: {step["config"]}', unsafe_allow_html=True)


# ===== MODEL TRAINING PAGE =====
elif page == "🤖 Model Training":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
        st.stop()

    st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)

    # Use processed data if available, otherwise original
    df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    with col2:
        random_state = st.number_input("Random state", 0, 100, 42)

    # Model selection
    st.markdown("### 🎯 Select Models to Train")

    if st.session_state.task_type == 'classification':
        available_models = get_classification_models()
    else:
        available_models = get_regression_models()

    selected_models = st.multiselect(
        "Choose models",
        list(available_models.keys()),
        default=list(available_models.keys())[:2]
    )

    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model")
            st.stop()

        try:
            # Prepare data
            with st.spinner("Preparing data..."):
                X_train, X_test, y_train, y_test = prepare_train_test_split(
                    df, st.session_state.target_column, test_size, random_state
                )

            st.success(f"✅ Data split: {len(X_train):,} training, {len(X_test):,} testing samples")

            # Train models
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")

                model = available_models[model_name]

                # Train
                start_time = time.time()
                trained_model = train_model(model, X_train, y_train)
                training_time = time.time() - start_time

                # Evaluate
                if st.session_state.task_type == 'classification':
                    metrics = evaluate_classification_model(trained_model, X_test, y_test)
                else:
                    metrics = evaluate_regression_model(trained_model, X_test, y_test)

                # Get feature importance
                importance = get_feature_importance(trained_model, X_train.columns.tolist())

                results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics,
                    'feature_importance': importance,
                    'training_time': training_time
                }

                progress_bar.progress((idx + 1) / len(selected_models))

            status_text.text("✅ Training complete!")
            st.session_state.model_results = results

            # Store data splits
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            # Show summary
            st.markdown("### 📊 Training Summary")
            summary_df = calculate_model_metrics_summary(results)
            st.dataframe(summary_df, use_container_width=True)

            # Model comparison plot
            fig = plot_model_comparison(summary_df, st.session_state.task_type)
            st.plotly_chart(fig, use_container_width=True)

            # FIX #4: Show next step
            show_next_step("Next Step: Go to 📈 Evaluation to analyze model performance and generate SHAP explanations!")

        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            import traceback
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

# ===== EVALUATION PAGE =====
elif page == "📈 Evaluation":
    if not st.session_state.model_results:
        st.warning("⚠️ Please train models first!")
        st.stop()

    st.markdown('<div class="section-header">📈 Model Evaluation</div>', unsafe_allow_html=True)

    # Model selector
    model_name = st.selectbox(
        "Select model to evaluate",
        list(st.session_state.model_results.keys())
    )

    result = st.session_state.model_results[model_name]
    metrics = result['metrics']

    # Metrics display
    st.markdown(f"### 📊 {model_name} Performance")

    if st.session_state.task_type == 'classification':
        # Classification metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("F1 Score", f"{metrics['f1']:.4f}")

        if 'roc_auc' in metrics:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")

        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        fig = plot_confusion_matrix(metrics['confusion_matrix'])
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curve
        if 'roc_curve' in metrics:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 ROC Curve")
                fig = plot_roc_curve(metrics['roc_curve'])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📊 Precision-Recall Curve")
                fig = plot_precision_recall_curve(metrics['pr_curve'])
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Regression metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['rmse']:.4f}")
        col2.metric("MAE", f"{metrics['mae']:.4f}")
        col3.metric("R² Score", f"{metrics['r2']:.4f}")

        # Actual vs Predicted
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Actual vs Predicted")
            fig = plot_actual_vs_predicted(metrics['actuals'], metrics['predictions'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📉 Residuals")
            fig = plot_residuals(metrics['actuals'], metrics['predictions'])
            st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if result['feature_importance'] is not None:
        st.markdown("### 🎯 Feature Importance")
        fig = plot_feature_importance(result['feature_importance'])
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Top Features Table"):
            st.dataframe(result['feature_importance'].head(20), use_container_width=True)

    # FIX #6: SHAP Explanations with ACTUAL VISUALIZATIONS
    st.markdown("---")
    st.markdown("### 🔍 SHAP Explainability")

    if st.checkbox("🔍 Generate SHAP Explanations (may take 30-60 seconds)"):
        with st.spinner("Generating SHAP values... This may take a moment."):
            try:
                shap_result = explain_prediction_shap(
                    result['model'],
                    st.session_state.X_train,
                    st.session_state.X_test,
                    max_samples=100
                )

                if shap_result:
                    st.success("✅ SHAP explanations generated successfully!")

                    # FIX #6: SHAP Summary Plot
                    st.markdown("### 📊 SHAP Feature Importance")
                    st.info("This plot shows which features have the most impact on model predictions overall")

                    fig = plot_shap_summary(
                        shap_result['shap_values'],
                        shap_result['X_test_sample']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # FIX #6: SHAP Waterfall Plot
                    st.markdown("### 🌊 SHAP Waterfall - Single Prediction Explanation")
                    st.info("This plot shows how each feature contributes to a specific prediction")

                    # Let user select which instance to explain
                    instance_idx = st.slider(
                        "Select prediction instance to explain",
                        0, len(shap_result['X_test_sample']) - 1, 0
                    )

                    fig = plot_shap_waterfall(
                        shap_result['shap_values'],
                        shap_result['X_test_sample'],
                        shap_result['explainer'],
                        instance_idx=instance_idx
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # FIX #4: Show next step
                    show_next_step("Next Step: Go to 💾 Experiments to save this experiment (if tracking enabled)")

                else:
                    st.warning("⚠️ SHAP not available for this model type")

            except Exception as e:
                st.error(f"❌ SHAP generation failed: {str(e)}")
                with st.expander("View Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    else:
        # FIX #4: Show next step even if SHAP not generated
        show_next_step("Next Step: Check the SHAP box above for model interpretability, or go to 💾 Experiments to save")

# ===== EXPERIMENTS PAGE =====
elif page == "💾 Experiments":
    st.markdown('<div class="section-header">💾 Experiment Tracking</div>', unsafe_allow_html=True)

    if not st.session_state.db_enabled:
        st.info("📌 Enable 'Experiment Tracking' in the sidebar to save experiments")
        st.stop()

    db = ExperimentDB()

    # Save current experiment
    if st.session_state.model_results:
        st.markdown("### 💾 Save Current Experiment")

        # ✅ Auto experiment naming
        existing_experiments = db.load_experiments()

        exp_numbers = []

        if len(existing_experiments) > 0:
            for name in existing_experiments["experiment_name"]:
                match = re.match(r"^Exp_(\d+)$", str(name))
                if match:
                    exp_numbers.append(int(match.group(1)))

        next_exp_num = max(exp_numbers, default=0) + 1

        default_exp_name = f"Exp_{next_exp_num}"

        # ✅ User can type custom name
        experiment_name = st.text_input("Experiment name",value="",placeholder=default_exp_name,key="experiment_name")

        # ✅ If empty → use placeholder/default
        if not experiment_name.strip():
            experiment_name = default_exp_name

        # =========================================
        # ✅ Save Button
        # =========================================

        if st.button("Save Experiment",type="primary"):
            existing_names = (existing_experiments['experiment_name'].str.lower().tolist())

            if experiment_name.lower() in existing_names:
                st.error("❌ Experiment name already exists.")
                st.stop()
            try:
                dataset_info = {
                    'name': 'uploaded_dataset',
                    'n_samples': len(st.session_state.df),
                    'n_features': len(st.session_state.df.columns),
                    'target_column': st.session_state.target_column
                }

                # ✅ Convert numpy/pandas objects to JSON-safe
                def make_json_safe(obj):
                    if isinstance(obj, dict):
                        return {k: make_json_safe(v) for k, v in obj.items()}

                    elif isinstance(obj, list):
                        return [make_json_safe(v) for v in obj]

                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()

                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict()

                    elif isinstance(obj, pd.Series):
                        return obj.to_list()

                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()

                    else:
                        return obj


                safe_results = make_json_safe(st.session_state.model_results)

                exp_id = db.save_experiment(
                    experiment_name,
                    st.session_state.task_type,
                    dataset_info,
                    st.session_state.preprocessing_steps,
                    safe_results
                )

                st.success(f"✅ Experiment saved with ID: {exp_id}")

                # ✅ Refresh page so placeholder updates
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {str(e)}")

    st.markdown("---")

    # Load experiments
    st.markdown("### 📂 Saved Experiments")

    experiments_df = db.load_experiments()
    display_df = experiments_df.copy()
    display_df.insert(0, "ID", range(len(display_df), 0, -1))

    if len(experiments_df) > 0:
        display_df = experiments_df.copy()
        # User-friendly numbering
        display_df.insert(0, "ID", range(len(display_df), 0, -1))
        # Delete checkbox
        display_df["Delete"] = False
        display_df = display_df.drop(columns=["id"])

        edited_df = st.data_editor(display_df,use_container_width=True,hide_index=True,
            disabled=[col for col in display_df.columns if col != "Delete"])
        
        rows_to_delete = edited_df[edited_df["Delete"]]
        if len(rows_to_delete) > 0:

            st.warning(f"⚠️ {len(rows_to_delete)} experiment(s) selected for deletion")
            if st.button("🗑️ Delete Selected Experiments",type="secondary"):

                for display_id in rows_to_delete["ID"]:
                    real_id = experiments_df.iloc[display_id - 1]["id"]
                    db.delete_experiment(real_id)

                st.success(f"✅ Deleted {len(rows_to_delete)} experiment(s)")
                st.rerun()

        # View experiment details
        selected_exp_id = st.selectbox(
            "Select experiment to view",
            experiments_df['id'].tolist(),
            format_func=lambda x: experiments_df[experiments_df['id']==x]['experiment_name'].values[0]
        )

        if st.button("Load Experiment Details"):
            details = db.load_experiment_details(selected_exp_id)

            if details:
                st.markdown(f"### 📊 {details['experiment_name']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Task", details['task_type'].title())
                col2.metric("Samples", f"{details['n_samples']:,}")
                col3.metric("Features", details['n_features'])

                st.markdown("**Preprocessing Steps:**")
                for step in details['preprocessing']:
                    st.write(f"- {step['type']}: {step['config']}")

                st.markdown("**Model Results:**")
                model_metrics = []
                for model_name, model_data in details['models'].items():
                    metrics = model_data['metrics']
                    model_metrics.append({
                        'Model': model_name,
                        **{k: f"{v:.4f}" if isinstance(v, float) else v
                           for k, v in metrics.items()
                           if k not in ['confusion_matrix', 'classification_report',
                                       'roc_curve', 'pr_curve', 'predictions', 'actuals']}
                    })

                st.dataframe(pd.DataFrame(model_metrics), use_container_width=True)
    else:
        st.info("No experiments saved yet")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>🔬 Feature Engineering & Model Benchmark Lab | Built with Streamlit</p>
    <p>✅ All 6 Issues Fixed | SHAP Visualizations Complete</p>
</div>
""", unsafe_allow_html=True)
