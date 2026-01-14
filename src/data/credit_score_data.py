"""Real Credit Score Dataset Loader and Preprocessor.

This module handles loading and preprocessing the Kaggle Credit Score dataset
for bias analysis in the Neural Microscope dashboard.

Dataset source: Kaggle Credit Score Classification
Features include: Age, Occupation, Income, Credit History, etc.
Target: Credit_Score (Good, Standard, Poor)
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Container for dataset metadata."""

    num_samples: int
    num_features: int
    feature_names: List[str]
    target_classes: List[str]
    protected_attributes: List[str]
    categorical_features: List[str]
    numerical_features: List[str]


# Features to use for the model
FEATURE_COLUMNS = [
    'Age',
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Num_Credit_Inquiries',
    'Credit_Utilization_Ratio',
    'Total_EMI_per_month',
    'Outstanding_Debt',
]

# Categorical features to encode
CATEGORICAL_FEATURES = [
    'Occupation',
    'Credit_Mix',
    'Payment_of_Min_Amount',
    'Payment_Behaviour',
]

# Features that could be protected attributes (for bias analysis)
PROTECTED_ATTRIBUTES = [
    'Occupation',
    'Age_Group',  # We'll create age groups
    'Credit_Mix',
    'Payment_of_Min_Amount',
]

# Target column
TARGET_COLUMN = 'Credit_Score'

# Target encoding
TARGET_ENCODING = {'Poor': 0, 'Standard': 1, 'Good': 2}


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw CSV data from the dataset.

    Args:
        data_path: Path to the data directory. If None, uses default.

    Returns:
        Raw DataFrame.
    """
    if data_path is None:
        # Default path relative to project root (src/data/ -> parents[2] = project root)
        data_path = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'CreditScore'

    train_path = data_path / 'train.csv'

    if not train_path.exists():
        raise FileNotFoundError(f"Dataset not found at {train_path}")

    df = pd.read_csv(train_path, low_memory=False)
    return df


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Clean a column that should be numeric but has garbage values.

    Handles cases like:
    - "_" placeholder values
    - Negative ages
    - String representations of numbers
    """
    # Convert to string first
    str_series = series.astype(str)

    # Remove underscores and other garbage
    str_series = str_series.str.replace('_', '', regex=False)
    str_series = str_series.str.replace(',', '', regex=False)

    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(str_series, errors='coerce')

    return numeric_series


def clean_categorical_column(series: pd.Series, valid_values: Optional[List[str]] = None) -> pd.Series:
    """Clean a categorical column by replacing garbage values.

    Args:
        series: The column to clean.
        valid_values: List of valid values. Others become 'Unknown'.
    """
    cleaned = series.astype(str).str.strip()

    # Replace common garbage patterns
    garbage_patterns = ['_', '_______', '!@9#%8', 'nan', 'NaN', '']
    for pattern in garbage_patterns:
        cleaned = cleaned.replace(pattern, 'Unknown')

    if valid_values is not None:
        cleaned = cleaned.apply(lambda x: x if x in valid_values else 'Unknown')

    return cleaned


def create_age_groups(ages: pd.Series) -> pd.Series:
    """Create age group bins for bias analysis.

    Groups:
    - Young: 18-30
    - Middle: 31-50
    - Senior: 51+
    """
    def categorize(age):
        if pd.isna(age) or age < 18 or age > 120:
            return 'Unknown'
        elif age <= 30:
            return 'Young (18-30)'
        elif age <= 50:
            return 'Middle (31-50)'
        else:
            return 'Senior (51+)'

    return ages.apply(categorize)


def preprocess_data(
    df: pd.DataFrame,
    max_samples: int = 10000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, DatasetInfo]:
    """Preprocess the raw data for model training and bias analysis.

    Args:
        df: Raw DataFrame.
        max_samples: Maximum number of samples to use (for speed).
        random_state: Random seed for sampling.

    Returns:
        Tuple of (processed DataFrame, DatasetInfo).
    """
    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    processed = pd.DataFrame()

    # Clean numeric features
    processed['Age'] = clean_numeric_column(df['Age'])
    processed['Age'] = processed['Age'].clip(18, 100)  # Reasonable age range

    processed['Annual_Income'] = clean_numeric_column(df['Annual_Income'])
    processed['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(0)
    processed['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].clip(0, 20)
    processed['Num_Credit_Card'] = df['Num_Credit_Card'].clip(0, 20)
    processed['Interest_Rate'] = df['Interest_Rate'].clip(0, 50)
    processed['Num_of_Loan'] = clean_numeric_column(df['Num_of_Loan']).fillna(0).clip(0, 20)
    processed['Delay_from_due_date'] = df['Delay_from_due_date'].clip(-10, 60)
    processed['Num_of_Delayed_Payment'] = clean_numeric_column(df['Num_of_Delayed_Payment']).fillna(0)
    processed['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].fillna(0).clip(0, 20)
    processed['Credit_Utilization_Ratio'] = df['Credit_Utilization_Ratio'].clip(0, 100)
    processed['Total_EMI_per_month'] = df['Total_EMI_per_month'].fillna(0)
    processed['Outstanding_Debt'] = clean_numeric_column(df['Outstanding_Debt']).fillna(0)

    # Clean categorical features
    valid_occupations = [
        'Lawyer', 'Architect', 'Engineer', 'Scientist', 'Mechanic',
        'Accountant', 'Developer', 'Media_Manager', 'Teacher',
        'Entrepreneur', 'Doctor', 'Journalist', 'Manager', 'Musician', 'Writer'
    ]
    processed['Occupation'] = clean_categorical_column(df['Occupation'], valid_occupations)

    valid_credit_mix = ['Good', 'Standard', 'Bad']
    processed['Credit_Mix'] = clean_categorical_column(df['Credit_Mix'], valid_credit_mix)

    valid_payment = ['Yes', 'No']
    processed['Payment_of_Min_Amount'] = clean_categorical_column(
        df['Payment_of_Min_Amount'], valid_payment
    )

    valid_behaviour = [
        'Low_spent_Small_value_payments',
        'High_spent_Medium_value_payments',
        'Low_spent_Medium_value_payments',
        'High_spent_Large_value_payments',
        'High_spent_Small_value_payments',
        'Low_spent_Large_value_payments'
    ]
    processed['Payment_Behaviour'] = clean_categorical_column(
        df['Payment_Behaviour'], valid_behaviour
    )

    # Create age groups for bias analysis
    processed['Age_Group'] = create_age_groups(processed['Age'])

    # Target
    processed['Credit_Score'] = df['Credit_Score'].map(TARGET_ENCODING)

    # Drop rows with missing target
    processed = processed.dropna(subset=['Credit_Score'])
    processed['Credit_Score'] = processed['Credit_Score'].astype(int)

    # Fill remaining NaN with median/mode
    for col in processed.select_dtypes(include=[np.number]).columns:
        if col != 'Credit_Score':
            processed[col] = processed[col].fillna(processed[col].median())

    # Get info
    numerical_features = [col for col in processed.columns
                         if processed[col].dtype in ['int64', 'float64']
                         and col != 'Credit_Score']

    info = DatasetInfo(
        num_samples=len(processed),
        num_features=len(numerical_features) + len(CATEGORICAL_FEATURES),
        feature_names=numerical_features + CATEGORICAL_FEATURES,
        target_classes=['Poor', 'Standard', 'Good'],
        protected_attributes=PROTECTED_ATTRIBUTES,
        categorical_features=CATEGORICAL_FEATURES,
        numerical_features=numerical_features,
    )

    return processed, info


def encode_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> Tuple[np.ndarray, Dict[str, Dict]]:
    """Encode features for neural network input.

    Args:
        df: Preprocessed DataFrame.
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.

    Returns:
        Tuple of (feature array, encoding dictionaries).
    """
    encodings = {}
    encoded_parts = []

    # Normalize numerical features
    for col in numerical_cols:
        values = df[col].values.astype(float)
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
        encoded_parts.append(normalized.reshape(-1, 1))
        encodings[col] = {'type': 'numerical', 'min': min_val, 'max': max_val}

    # One-hot encode categorical features
    for col in categorical_cols:
        unique_values = sorted(df[col].unique())
        value_to_idx = {v: i for i, v in enumerate(unique_values)}

        one_hot = np.zeros((len(df), len(unique_values)))
        for i, v in enumerate(df[col].values):
            if v in value_to_idx:
                one_hot[i, value_to_idx[v]] = 1

        encoded_parts.append(one_hot)
        encodings[col] = {'type': 'categorical', 'mapping': value_to_idx}

    X = np.hstack(encoded_parts)
    return X, encodings


def prepare_tensors(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Prepare PyTorch tensors for training.

    Args:
        df: Preprocessed DataFrame.
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.

    Returns:
        Tuple of (X tensor, y tensor, encodings dict).
    """
    X, encodings = encode_features(df, categorical_cols, numerical_cols)
    y = df['Credit_Score'].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor, encodings


def get_protected_attribute_values(
    df: pd.DataFrame,
    attribute: str
) -> List[str]:
    """Get unique values for a protected attribute.

    Args:
        df: Preprocessed DataFrame.
        attribute: Name of the protected attribute.

    Returns:
        List of unique values (excluding 'Unknown').
    """
    values = df[attribute].unique().tolist()
    values = [v for v in values if v != 'Unknown']
    return sorted(values)


def create_contrastive_samples(
    df: pd.DataFrame,
    X_tensor: torch.Tensor,
    protected_attribute: str,
    value_a: str,
    value_b: str,
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Create contrastive samples for bias analysis.

    Takes samples with attribute=value_a and creates counterfactuals
    with attribute=value_b (keeping other features the same).

    Args:
        df: Preprocessed DataFrame.
        X_tensor: Encoded feature tensor.
        protected_attribute: Name of the protected attribute.
        value_a: Original value.
        value_b: Counterfactual value.
        numerical_cols: Numerical column names.
        categorical_cols: Categorical column names.

    Returns:
        Tuple of (original tensors, counterfactual tensors, sample info DataFrame).
    """
    # Find samples with value_a
    mask_a = df[protected_attribute] == value_a
    indices_a = np.where(mask_a)[0]

    if len(indices_a) == 0:
        raise ValueError(f"No samples found with {protected_attribute}={value_a}")

    # Take subset
    max_samples = min(100, len(indices_a))
    indices_a = indices_a[:max_samples]

    # Get original tensors
    X_original = X_tensor[indices_a]

    # Create counterfactual DataFrame
    df_cf = df.iloc[indices_a].copy()
    df_cf[protected_attribute] = value_b

    # Re-encode counterfactual
    X_cf, _ = encode_features(df_cf, categorical_cols, numerical_cols)
    X_counterfactual = torch.tensor(X_cf, dtype=torch.float32)

    return X_original, X_counterfactual, df.iloc[indices_a]


# Convenience function for Streamlit
def load_credit_score_dataset(
    max_samples: int = 5000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, Dict, DatasetInfo]:
    """Load and preprocess Credit Score dataset for Streamlit.

    Args:
        max_samples: Maximum samples to load.
        random_state: Random seed.

    Returns:
        Tuple of (DataFrame, X_tensor, y_tensor, encodings, info).
    """
    raw_df = load_raw_data()
    processed_df, info = preprocess_data(raw_df, max_samples, random_state)

    numerical_cols = info.numerical_features
    categorical_cols = info.categorical_features

    X, y, encodings = prepare_tensors(processed_df, numerical_cols, categorical_cols)

    return processed_df, X, y, encodings, info
