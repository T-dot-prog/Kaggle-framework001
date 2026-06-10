import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from exceptions import ToolExecutionError, ValidationError
from tool_registry import register_tool
from tools.validators import validate_dataframe, validate_tool_inputs, VALID_IMPUTATION_STRATEGIES, VALID_OUTLIER_METHODS

logger = logging.getLogger(__name__)

METHOD_ALIASES = {
    "handle_missing_values": {
        "strategy": {"most_frequent": "mode", "frequent": "mode", "most_common": "mode"},
    },
    "detect_outliers": {
        "method": {"z-score": "zscore", "z_score": "zscore", "3sigma": "zscore", "3_sigma": "zscore",
                   "isolation_forest": "iqr", "isolation forest": "iqr"},
    },
    "scale_features": {
        "method": {"normalize": "minmax", "normalise": "minmax", "standardize": "standard", "standardise": "standard"},
    },
    "encode_categorical": {
        "method": {"one-hot": "onehot", "one_hot": "onehot", "dummy": "onehot", "get_dummies": "onehot"},
    },
    "train_model": {
        "model_type": {"rf": "random_forest", "randomforest": "random_forest", "random forest": "random_forest",
                       "xgb": "xgboost", "lr": "logistic_regression", "logistic": "logistic_regression",
                       "linear": "linear_regression", "linreg": "linear_regression"},
    },
}


def _normalize(tool_name, param_name, value):
    mapping = METHOD_ALIASES.get(tool_name, {}).get(param_name, {})
    if value in mapping:
        logger.debug("Normalized '%s' in %s.%s -> '%s'", value, tool_name, param_name, mapping[value])
        return mapping[value]
    return value


def handle_missing_values(df, strategy="mean", fill_value=None, columns=None):
    strategy = _normalize("handle_missing_values", "strategy", strategy)
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("handle_missing_values", strategy=strategy)

    result = df.copy()
    target_cols = list(columns) if columns is not None else list(result.columns)

    try:
        if strategy == "drop":
            before = len(result)
            result = result.dropna(subset=target_cols)
            logger.info("handle_missing_values: dropped %d rows with missing values", before - len(result))

        elif strategy == "ffill":
            result[target_cols] = result[target_cols].ffill()
            logger.info("handle_missing_values: applied forward fill")

        elif strategy == "bfill":
            result[target_cols] = result[target_cols].bfill()
            logger.info("handle_missing_values: applied backward fill")

        elif strategy == "constant":
            if fill_value is None:
                raise ValidationError("fill_value is required for 'constant' strategy")
            result[target_cols] = result[target_cols].fillna(fill_value)
            logger.info("handle_missing_values: filled with constant %s", fill_value)

        elif strategy in ("mean", "median", "mode"):
            for col in target_cols:
                if col not in result.columns:
                    continue
                if result[col].isnull().sum() == 0:
                    continue
                if not pd.api.types.is_numeric_dtype(result[col].dtype):
                    logger.info("handle_missing_values: skipping non-numeric column '%s'", col)
                    continue
                if strategy == "mean":
                    val = result[col].mean()
                elif strategy == "median":
                    val = result[col].median()
                else:
                    val = result[col].mode()
                    val = val.iloc[0] if not val.empty else None
                if val is not None:
                    result[col] = result[col].fillna(val)
                    filled = result[col].isnull().sum()
                    logger.info("handle_missing_values: filled '%s' using '%s' (remaining: %d)", col, strategy, filled)

        remaining = result.isnull().sum().sum()
        logger.info("handle_missing_values: completed with %d remaining missing values", remaining)
        return result, None

    except Exception as e:
        raise ToolExecutionError(f"handle_missing_values failed: {e}")


def detect_outliers(df, method="iqr", threshold=1.5, columns=None):
    method = _normalize("detect_outliers", "method", method)
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("detect_outliers", method=method)

    result = df.copy()
    target_cols = list(columns) if columns is not None else result.select_dtypes(include=[np.number]).columns.tolist()

    try:
        outlier_mask = pd.Series(False, index=result.index)

        for col in target_cols:
            if col not in result.columns:
                continue
            if result[col].dtype not in (np.float64, np.int64, float, int):
                continue
            if result[col].isnull().all():
                continue

            col_data = result[col].dropna()

            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_outliers = (result[col] < lower) | (result[col] > upper)
            elif method == "zscore":
                mean = col_data.mean()
                std = col_data.std()
                if std == 0:
                    continue
                z = (result[col] - mean) / std
                col_outliers = z.abs() > threshold
            else:
                raise ValidationError(f"Unknown outlier method: {method}")

            col_outliers = col_outliers.fillna(False)
            outlier_mask = outlier_mask | col_outliers

        logger.info("detect_outliers: found %d outlier rows across %d columns",
                    outlier_mask.sum(), len(target_cols))
        return result, outlier_mask

    except Exception as e:
        raise ToolExecutionError(f"detect_outliers failed: {e}")


def scale_features(df, method="standard", columns=None):
    method = _normalize("scale_features", "method", method)
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("scale_features", method=method)

    result = df.copy()
    if columns is not None:
        target_cols = list(columns)
    else:
        target_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in target_cols if c in result.columns and result[c].nunique() > 1]

    try:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValidationError(f"Unknown scaling method: {method}")

        if not target_cols:
            logger.warning("scale_features: no numeric columns with variance found")
            return result, None

        scaled_values = scaler.fit_transform(result[target_cols])
        result[target_cols] = scaled_values
        logger.info("scale_features: applied '%s' scaling to %d columns", method, len(target_cols))
        return result, scaler

    except Exception as e:
        raise ToolExecutionError(f"scale_features failed: {e}")


def encode_categorical(df, method="onehot", columns=None):
    method = _normalize("encode_categorical", "method", method)
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("encode_categorical", method=method)

    result = df.copy()
    target_cols = list(columns) if columns is not None else result.select_dtypes(include=["object", "category"]).columns.tolist()
    target_cols = [c for c in target_cols if c in result.columns]

    try:
        if method == "onehot":
            result = pd.get_dummies(result, columns=target_cols, drop_first=False)
            logger.info("encode_categorical: applied one-hot encoding to %d columns", len(target_cols))
            return result, None

        elif method in ("label", "ordinal"):
            mapping = {}
            for col in target_cols:
                if result[col].dtype.name == "category":
                    result[col] = result[col].cat.codes
                else:
                    codes, uniques = pd.factorize(result[col])
                    result[col] = codes
                    mapping[col] = list(uniques)
            logger.info("encode_categorical: applied label encoding to %d columns", len(target_cols))
            return result, mapping

        elif method == "target":
            logger.warning("encode_categorical: target encoding is a stub — using label encoding instead")
            for col in target_cols:
                codes, _ = pd.factorize(result[col])
                result[col] = codes
            return result, None

        else:
            raise ValidationError(f"Unknown encoding method: {method}")

    except Exception as e:
        raise ToolExecutionError(f"encode_categorical failed: {e}")


def create_polynomial_features(df, degree=2, columns=None):
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("create_polynomial_features", degree=degree)

    result = df.copy()
    target_cols = list(columns) if columns is not None else result.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in target_cols if c in result.columns]

    try:
        from itertools import combinations

        for col in target_cols:
            new_name = f"{col}^2"
            result[new_name] = result[col] ** 2

        if degree >= 2:
            for a, b in combinations(target_cols, 2):
                new_name = f"{a}_{b}_inter"
                result[new_name] = result[a] * result[b]

        if degree >= 3:
            for col in target_cols:
                new_name = f"{col}^3"
                result[new_name] = result[col] ** 3

        logger.info("create_polynomial_features: created polynomial features (degree=%d)", degree)
        return result, None

    except Exception as e:
        raise ToolExecutionError(f"create_polynomial_features failed: {e}")


def select_features_by_variance(df, threshold=0.0, columns=None):
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("select_features_by_variance", threshold=threshold)

    result = df.copy()
    target_cols = list(columns) if columns is not None else result.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in target_cols if c in result.columns]

    try:
        kept = []
        dropped = []
        for col in target_cols:
            var = result[col].var()
            if var >= threshold:
                kept.append(col)
            else:
                dropped.append(col)

        non_numeric_cols = [c for c in result.columns if c not in target_cols]
        result = result[non_numeric_cols + kept]

        logger.info("select_features_by_variance: kept %d features, dropped %d (threshold=%.4f)",
                    len(kept), len(dropped), threshold)
        return result, {"kept": kept, "dropped": dropped}

    except Exception as e:
        raise ToolExecutionError(f"select_features_by_variance failed: {e}")


def split_data(df, target_col, test_size=0.2, random_state=42):
    validate_dataframe(df, allow_empty=False)
    validate_tool_inputs("split_data", test_size=test_size)

    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in DataFrame")

    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info("split_data: train=%d rows, test=%d rows (test_size=%.2f)",
                    len(X_train), len(X_test), test_size)
        return (X_train, X_test, y_train, y_test), None

    except Exception as e:
        raise ToolExecutionError(f"split_data failed: {e}")


def train_model(X_train, y_train, model_type="random_forest", random_state=42, **kwargs):
    model_type = _normalize("train_model", "model_type", model_type)
    validate_tool_inputs("train_model", model_type=model_type)

    try:
        if model_type == "random_forest":
            n_estimators = kwargs.get("n_estimators", 100)
            y_unique = y_train.nunique() if hasattr(y_train, 'nunique') else len(y_train.unique())
            if y_train.dtype in (np.int64, int, bool) or y_unique <= 10:
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

        elif model_type == "xgboost":
            if not _XGB_AVAILABLE:
                raise ToolExecutionError("xgboost is not installed")
            y_unique = y_train.nunique() if hasattr(y_train, 'nunique') else len(y_train.unique())
            if y_train.dtype in (np.int64, int, bool) or y_unique <= 10:
                model = XGBClassifier(random_state=random_state, verbosity=0)
            else:
                model = XGBRegressor(random_state=random_state, verbosity=0)

        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=random_state, max_iter=kwargs.get("max_iter", 1000))

        elif model_type == "linear_regression":
            model = LinearRegression()

        else:
            raise ValidationError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        logger.info("train_model: trained '%s' on %d samples", model_type, len(X_train))
        return model, None

    except Exception as e:
        raise ToolExecutionError(f"train_model failed: {e}")


def evaluate_model(model, X_test, y_test, task_type=None):
    try:
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
            except Exception:
                y_prob = None
        else:
            y_prob = None

        # Handle NaN in nunique()
        y_unique = y_test.nunique() if hasattr(y_test, 'nunique') else len(y_test.unique())
        
        if task_type is None:
            # Guess task type if not provided
            is_classification = y_test.dtype in (np.int64, int, bool) or y_unique <= 10
            logger.warning("evaluate_model: task_type not provided, guessing from data (classification=%s)", is_classification)
        else:
            is_classification = task_type == "classification"

        metrics = {}
        if is_classification:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["f1"] = float(f1_score(y_test, y_pred, average="weighted"))
            if y_prob is not None and y_unique == 2:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                except Exception:
                    pass
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics["r2"] = float(r2_score(y_test, y_pred))

        logger.info("evaluate_model: metrics=%s", metrics)
        return metrics, {"y_pred": y_pred, "y_prob": y_prob}

    except Exception as e:
        raise ToolExecutionError(f"evaluate_model failed: {e}")


_TOOL_REGISTRATIONS = [
    ("handle_missing_values", ["missing_values", "imputation", "data_cleaning"]),
    ("detect_outliers", ["outliers", "anomaly_detection", "data_cleaning"]),
    ("scale_features", ["scaling", "normalization", "feature_engineering"]),
    ("encode_categorical", ["encoding", "categorical", "feature_engineering"]),
    ("create_polynomial_features", ["polynomial", "feature_interaction", "feature_engineering"]),
    ("select_features_by_variance", ["variance_threshold", "feature_selection", "feature_engineering"]),
    ("split_data", ["train_test_split", "data_splitting", "modeling"]),
    ("train_model", ["training", "model_fitting", "modeling"]),
    ("evaluate_model", ["evaluation", "metrics", "modeling"]),
]

for _name, _tags in _TOOL_REGISTRATIONS:
    register_tool(_name, _tags)
