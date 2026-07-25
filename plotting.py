"""
Plotting utilities using Plotly
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """
    Plot missing values heatmap
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values found!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = go.Figure(data=[
        go.Bar(
            x=missing_data.index,
            y=missing_data.values,
            marker=dict(color='indianred')
        )
    ])
    
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Column",
        yaxis_title="Number of Missing Values",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Plot correlation matrix heatmap
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numeric columns for correlation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_white",
        height=600,
        width=700
    )
    
    return fig


def plot_feature_distributions(df: pd.DataFrame, n_cols: int = 3) -> go.Figure:
    """
    Plot distributions of numeric features
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric columns to plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    n_features = min(len(numeric_cols), 9)  # Limit to 9 features
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=numeric_cols[:n_features]
    )
    
    for idx, col in enumerate(numeric_cols[:n_features]):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title="Feature Distributions",
        template="plotly_white",
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None) -> go.Figure:
    """
    Plot confusion matrix
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    text = [[f"{cm[i][j]}<br>({cm_normalized[i][j]:.1f}%)" 
             for j in range(len(cm[0]))] for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_roc_curve(roc_data: Dict[str, np.ndarray]) -> go.Figure:
    """
    Plot ROC curve
    """
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=500,
        width=600
    )
    
    return fig


def plot_precision_recall_curve(pr_data: Dict[str, np.ndarray]) -> go.Figure:
    """
    Plot Precision-Recall curve
    """
    precision = pr_data['precision']
    recall = pr_data['recall']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='PR Curve',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=500,
        width=600
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Plot feature importance
    """
    if importance_df is None or len(importance_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Feature importance not available for this model",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    top_features = importance_df.head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker=dict(color='steelblue')
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_actual_vs_predicted(actuals: np.ndarray, predictions: np.ndarray) -> go.Figure:
    """
    Plot actual vs predicted for regression
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=actuals, 
        y=predictions,
        mode='markers',
        name='Predictions',
        marker=dict(color='steelblue', size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_white",
        height=500,
        width=600
    )
    
    return fig


def plot_residuals(actuals: np.ndarray, predictions: np.ndarray) -> go.Figure:
    """
    Plot residuals for regression
    """
    residuals = actuals - predictions
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        marker=dict(color='coral', size=8, opacity=0.6)
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        template="plotly_white",
        height=500,
        width=600
    )
    
    return fig


def plot_model_comparison(summary_df: pd.DataFrame, task_type: str = 'classification') -> go.Figure:
    """
    Plot model comparison metrics
    """
    if task_type == 'classification':
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        # Convert string metrics to float
        for metric in metrics:
            if metric in summary_df.columns:
                summary_df[metric] = summary_df[metric].astype(float)
    else:
        metrics = ['RMSE', 'MAE', 'R²']
        for metric in metrics:
            if metric in summary_df.columns:
                summary_df[metric] = summary_df[metric].astype(float)
    
    fig = go.Figure()
    
    for metric in metrics:
        if metric in summary_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=summary_df['Model'],
                y=summary_df[metric],
                text=summary_df[metric].round(3),
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_shap_summary(shap_values, X_test_sample, top_n: int = 20) -> go.Figure:
    """
    FIX #6: Plot SHAP feature importance summary
    Shows which features have the most impact on model predictions
    """
    try:
        # Handle different SHAP value formats (binary, multi-class, regression)
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # 🔥 HANDLE 3D CASE (RandomForest classification)
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 0]  # pick class 0
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_vals).mean(axis=0)
        
        # Create DataFrame for sorting
        feature_importance = pd.DataFrame({
            'feature': X_test_sample.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=feature_importance['feature'][::-1],  # Reverse for top-to-bottom
                x=feature_importance['importance'][::-1],
                orientation='h',
                marker=dict(
                    color=feature_importance['importance'][::-1],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Mean |SHAP|")
                ),
                text=[f"{x:.4f}" for x in feature_importance['importance'][::-1]],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"SHAP Feature Importance - Top {top_n} Features",
            xaxis_title="Mean Absolute SHAP Value (Impact on Predictions)",
            yaxis_title="Feature",
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating SHAP summary: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig


def plot_shap_waterfall(shap_values, X_test_sample, explainer, instance_idx: int = 0) -> go.Figure:
    """
    FIX #6: Plot SHAP waterfall chart for a single prediction explanation
    Shows how each feature pushes the prediction higher or lower
    """
    try:
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # 🔥 HANDLE 3D CASE
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 0]
        
        # Get SHAP values for the specified instance
        instance_shap = shap_vals[instance_idx]
        feature_names = X_test_sample.columns.tolist()
        feature_values = X_test_sample.iloc[instance_idx].values
        
        # Create DataFrame and sort by absolute SHAP value
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': instance_shap,
            'feature_value': feature_values
        })
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False).head(15)
        
        # Get base value (expected value)
        base_value = 0
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        
        # Create labels with feature values
        labels = [f"{feat}<br>value = {val:.3f}" for feat, val in zip(shap_df['feature'], shap_df['feature_value'])]
        
        # Color code: red for negative impact (pushes down), blue for positive (pushes up)
        colors = ['rgba(244, 67, 54, 0.8)' if x < 0 else 'rgba(33, 150, 243, 0.8)' 
                  for x in shap_df['shap_value']]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=shap_df['shap_value'],
            y=labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{x:+.4f}" for x in shap_df['shap_value']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>SHAP Impact: %{x:.4f}<extra></extra>'
        ))
        
        # Add base value annotation
        fig.add_annotation(
            text=f"Base value (model average): {base_value:.4f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        # Add legend for colors
        fig.add_annotation(
            text="🔴 Red = Decreases prediction | 🔵 Blue = Increases prediction",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.update_layout(
            title=f"SHAP Waterfall - How Features Impact Prediction #{instance_idx}",
            xaxis_title="SHAP Value (Impact on Model Output)",
            yaxis_title="Feature (with actual value)",
            template="plotly_white",
            height=600,
            showlegend=False,
            margin=dict(b=100)  # Extra bottom margin for legend
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating SHAP waterfall: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig
