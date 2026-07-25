"""
Data utilities for preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
import warnings
import re
warnings.filterwarnings('ignore')


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate uploaded dataset and return diagnostics
    """
    diagnostics = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Check for high cardinality categoricals
    high_card = []
    for col in diagnostics['categorical_cols']:
        if df[col].nunique() > 50:
            high_card.append(col)
    diagnostics['high_cardinality_cols'] = high_card
    
    return diagnostics


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values
    """
    df = df.copy()
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].iloc[0], inplace=True)
    
    return df


def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical variables
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        elif method == 'onehot':
            # Get dummies and handle prefix
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    
    return df


def scale_features(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
    """
    Scale numerical features
    """
    df = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        return df
    
    if columns:
        df[columns] = scaler.fit_transform(df[columns])
    
    return df


def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create clean polynomial features (XGBoost-safe)
    """
    from sklearn.preprocessing import PolynomialFeatures

    df = df.copy()

    if not columns or degree < 2:
        return df

    # ✅ Keep only valid numeric columns
    valid_cols = [col for col in columns if col in df.columns]
    valid_cols = [col for col in valid_cols if pd.api.types.is_numeric_dtype(df[col])]

    if not valid_cols:
        return df

    poly = PolynomialFeatures(degree=degree, include_bias=False)

    poly_features = poly.fit_transform(df[valid_cols])

    # ✅ CLEAN feature names (CRITICAL)
    raw_names = poly.get_feature_names_out(valid_cols)

    clean_names = []
    for name in raw_names:
        # Replace special chars with underscore
        name = re.sub(r'[^\w]', '_', name)

        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)

        # Trim underscores
        name = name.strip('_')

        clean_names.append(name.lower())

    poly_df = pd.DataFrame(poly_features, columns=clean_names, index=df.index)

    # ✅ Drop originals
    df = df.drop(columns=valid_cols)

    # ✅ Merge safely
    df = pd.concat([df, poly_df], axis=1)

    return df


def create_interactions(df: pd.DataFrame, col_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between column pairs
    """
    df = df.copy()
    
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            interaction_name = f"{col1}_x_{col2}"
            df[interaction_name] = df[col1] * df[col2]
    
    return df


def extract_datetime_features(df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
    """
    Extract features from datetime columns
    """
    df = df.copy()
    
    for col in datetime_cols:
        if col not in df.columns:
            continue
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract features
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        df[f'{col}_quarter'] = df[col].dt.quarter
        df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Drop original datetime column
        df = df.drop(columns=[col])
    
    return df


def prepare_train_test_split(df: pd.DataFrame, 
                              target_col: str, 
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 20 else None
    )
    
    return X_train, X_test, y_train, y_test


def generate_sample_classification_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample classification dataset
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['cat1', 'cat2', 'cat3'], size=n_samples)
    df['category_B'] = np.random.choice(['low', 'medium', 'high'], size=n_samples)
    
    # Add date feature
    df['date'] = pd.date_range('2023-01-01', periods=n_samples, freq='h')
    
    return df


def generate_sample_regression_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample regression dataset
    """
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        noise=10.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
    df['target'] = y
    
    # Add some categorical features
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
    df['product_type'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    
    return df
