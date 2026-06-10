import logging

import numpy as np
import pandas as pd

from exceptions import ValidationError

logger = logging.getLogger(__name__)

VALID_IMPUTATION_STRATEGIES = {"mean", "median", "mode", "constant", "drop", "ffill", "bfill"}
VALID_OUTLIER_METHODS = {"iqr", "zscore"}
VALID_SCALING_METHODS = {"standard", "minmax", "robust"}
VALID_ENCODING_METHODS = {"onehot", "label", "ordinal", "target"}
VALID_MODEL_TYPES = {"random_forest", "xgboost", "logistic_regression", "linear_regression"}


def validate_dataframe(df, required_columns=None, allow_empty=False, check_inf=True):
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    if df.empty and not allow_empty:
        raise ValidationError("DataFrame is empty")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")

    if check_inf:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                continue
            if np.isinf(df[col]).any().any():
                raise ValidationError(f"Column '{col}' contains infinity values")

    logger.debug("DataFrame validation passed: shape=%s, cols=%d", df.shape, len(df.columns))
    return True


def validate_tool_inputs(tool_name, **kwargs):
    checks = {
        "handle_missing_values": lambda: _check_enum("strategy", kwargs.get("strategy"), VALID_IMPUTATION_STRATEGIES),
        "detect_outliers": lambda: _check_enum("method", kwargs.get("method"), VALID_OUTLIER_METHODS),
        "scale_features": lambda: _check_enum("method", kwargs.get("method"), VALID_SCALING_METHODS),
        "encode_categorical": lambda: _check_enum("method", kwargs.get("method"), VALID_ENCODING_METHODS),
        "create_polynomial_features": lambda: _check_int_ge("degree", kwargs.get("degree"), 2),
        "select_features_by_variance": lambda: _check_float_ge("threshold", kwargs.get("threshold"), 0.0),
        "split_data": lambda: _check_float_range("test_size", kwargs.get("test_size", 0.2), 0.0, 1.0),
        "train_model": lambda: _check_enum("model_type", kwargs.get("model_type"), VALID_MODEL_TYPES),
    }
    check = checks.get(tool_name)
    if check:
        check()
    logger.debug("Input validation passed for tool '%s'", tool_name)


def _check_enum(name, value, allowed):
    if value is not None and value not in allowed:
        raise ValidationError(f"{name} must be one of {allowed}, got '{value}'")


def _check_int_ge(name, value, minimum):
    if value is not None:
        if not isinstance(value, int):
            raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
        if value < minimum:
            raise ValidationError(f"{name} must be >= {minimum}, got {value}")


def _check_float_ge(name, value, minimum):
    if value is not None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
        if v < minimum:
            raise ValidationError(f"{name} must be >= {minimum}, got {v}")


def _check_float_range(name, value, low, high):
    if value is not None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
        if v < low or v > high:
            raise ValidationError(f"{name} must be between {low} and {high}, got {v}")
