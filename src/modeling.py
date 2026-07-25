"""
Machine learning modeling utilities
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def get_classification_models() -> Dict[str, Any]:
    """
    Return dictionary of classification models
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }


def get_regression_models() -> Dict[str, Any]:
    """
    Return dictionary of regression models
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a model
    """

    # 🔥 CRITICAL FIX: Flatten any nested structures
    X_train = X_train.copy()

    # Remove any columns that are DataFrames
    bad_cols = [col for col in X_train.columns if isinstance(X_train[col], pd.DataFrame)]
    if bad_cols:
        print(f"⚠️ Dropping invalid columns: {bad_cols}")
        X_train = X_train.drop(columns=bad_cols)

    # Ensure all columns are numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    # Optional: fill NaNs created during coercion
    X_train = X_train.fillna(0)

    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate classification model and return metrics
    """
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_pred_proba = None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    
    # Add ROC AUC if binary classification and probabilities available
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            metrics['pr_curve'] = {'precision': precision, 'recall': recall}
        except:
            pass
    
    return metrics


def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate regression model and return metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred[:100],  # Store sample predictions
        'actuals': y_test.values[:100]
    }
    
    return metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from model
    """
    try:
        # Try different attribute names
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:
                importances = importances[0]
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return None


def explain_prediction_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           max_samples: int = 100) -> Any:
    """
    Generate SHAP explanations for model predictions
    """
    try:
        import shap
        
        # Limit samples for performance
        X_train_sample = X_train.sample(min(50, len(X_train)), random_state=42)
        X_test_sample = X_test.sample(min(20, len(X_test)), random_state=42)
        
        # Create explainer based on model type
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor,
                      RandomForestClassifier, RandomForestRegressor,
                      lgb.LGBMClassifier, lgb.LGBMRegressor)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train_sample)
        
        shap_values = explainer.shap_values(X_test_sample)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_test_sample': X_test_sample
        }
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None


def calculate_model_metrics_summary(models_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create summary DataFrame of model metrics
    """
    summary_data = []
    
    for model_name, results in models_results.items():
        metrics = results.get('metrics', {})
        
        if 'accuracy' in metrics:  # Classification
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC AUC': f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else 'N/A'
            })
        else:  # Regression
            summary_data.append({
                'Model': model_name,
                'RMSE': f"{metrics.get('rmse', 0):.4f}",
                'MAE': f"{metrics.get('mae', 0):.4f}",
                'R²': f"{metrics.get('r2', 0):.4f}"
            })
    
    return pd.DataFrame(summary_data)
