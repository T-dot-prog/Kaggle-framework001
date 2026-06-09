import logging

from tool_registry import register_tool

logger = logging.getLogger(__name__)


def handle_missing_values(df, strategy="mean"):
    raise NotImplementedError("handle_missing_values: to be implemented in Layer 3")


def detect_outliers(df, method="iqr"):
    raise NotImplementedError("detect_outliers: to be implemented in Layer 3")


def scale_features(df, method="standard"):
    raise NotImplementedError("scale_features: to be implemented in Layer 3")


def encode_categoricals(df, method="onehot"):
    raise NotImplementedError("encode_categoricals: to be implemented in Layer 3")


def create_polynomial_features(df, degree=2):
    raise NotImplementedError("create_polynomial_features: to be implemented in Layer 3")


def select_features_by_variance(df, threshold=0.0):
    raise NotImplementedError("select_features_by_variance: to be implemented in Layer 3")


def split_data(df, target_col, test_size=0.2):
    raise NotImplementedError("split_data: to be implemented in Layer 3")


def train_model(X_train, y_train, model_type="random_forest"):
    raise NotImplementedError("train_model: to be implemented in Layer 3")


def evaluate_model(model, X_test, y_test):
    raise NotImplementedError("evaluate_model: to be implemented in Layer 3")


_TOOL_REGISTRATIONS = [
    ("handle_missing_values", ["missing_values", "imputation", "data_cleaning"]),
    ("detect_outliers", ["outliers", "anomaly_detection", "data_cleaning"]),
    ("scale_features", ["scaling", "normalization", "feature_engineering"]),
    ("encode_categoricals", ["encoding", "categorical", "feature_engineering"]),
    ("create_polynomial_features", ["polynomial", "feature_interaction", "feature_engineering"]),
    ("select_features_by_variance", ["variance_threshold", "feature_selection", "feature_engineering"]),
    ("split_data", ["train_test_split", "data_splitting", "modeling"]),
    ("train_model", ["training", "model_fitting", "modeling"]),
    ("evaluate_model", ["evaluation", "metrics", "modeling"]),
]

for _name, _tags in _TOOL_REGISTRATIONS:
    register_tool(_name, _tags)
