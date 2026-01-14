"""Data loading and preprocessing modules."""

from src.data.credit_score_data import (
    load_credit_score_dataset,
    get_protected_attribute_values,
    create_contrastive_samples,
    DatasetInfo,
    PROTECTED_ATTRIBUTES,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_ENCODING,
)

__all__ = [
    "load_credit_score_dataset",
    "get_protected_attribute_values",
    "create_contrastive_samples",
    "DatasetInfo",
    "PROTECTED_ATTRIBUTES",
    "FEATURE_COLUMNS",
    "CATEGORICAL_FEATURES",
    "TARGET_COLUMN",
    "TARGET_ENCODING",
]
