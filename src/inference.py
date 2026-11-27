import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_preprocessor(data_path):
    """Load dataset and fit the training scaler and expected dummy schema for inference."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found. Please run `python src/fetch_data.py`."
        )

    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    X = df.drop('Churn', axis=1)

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Match the exact train split used by load_and_preprocess_data so scaler statistics align
    if len(X) >= 2:
        X_train, _ = train_test_split(X, test_size=0.2, random_state=42)
    else:
        X_train = X

    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])

    X_dummy = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    expected_cols = X_dummy.columns.tolist()

    return scaler, categorical_cols, numerical_cols, expected_cols, X


def preprocess_input(
    input_df,
    scaler,
    categorical_cols,
    numerical_cols,
    expected_cols,
    template_df=None,
):
    """Preprocess inference rows without drop_first baseline drift."""
    _ = template_df  # Kept for backwards-compatible signature
    processed = input_df.copy()
    processed[numerical_cols] = scaler.transform(processed[numerical_cols])

    # Expand all present categories with drop_first=False, then reindex to exact training columns
    processed_dummy = pd.get_dummies(
        processed, columns=categorical_cols, drop_first=False
    )
    return processed_dummy.reindex(columns=expected_cols, fill_value=0)


def save_model_json(model: LogisticRegression, json_path: str) -> None:
    """Serialize fitted LogisticRegression parameters to JSON without pickle."""
    payload = {
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "n_features_in": int(model.n_features_in_),
    }
    if hasattr(model, "feature_names_in_"):
        payload["feature_names_in"] = model.feature_names_in_.tolist()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_model_json(json_path: str) -> LogisticRegression:
    """Reconstruct a fitted LogisticRegression from JSON parameters without pickle."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Trained model artifact not found. Please run `python src/train_model.py`."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.classes_ = np.array(payload["classes"])
    model.coef_ = np.array(payload["coef"], dtype=float)
    model.intercept_ = np.array(payload["intercept"], dtype=float)
    model.n_features_in_ = int(payload["n_features_in"])
    if "feature_names_in" in payload:
        model.feature_names_in_ = np.array(payload["feature_names_in"], dtype=object)
    return model


def ensure_runtime_artifacts(data_path: str, model_json_path: str) -> None:
    """Download dataset and train balanced Logistic Regression JSON artifact if missing."""
    try:
        from src.data_preprocessing import load_and_preprocess_data
        from src.fetch_data import main as fetch_data_main
    except ImportError:  # pragma: no cover
        from data_preprocessing import load_and_preprocess_data
        from fetch_data import main as fetch_data_main

    if not os.path.exists(data_path):
        fetch_data_main()

    if os.path.exists(data_path) and not os.path.exists(model_json_path):
        X_train, _, y_train, _ = load_and_preprocess_data(data_path)
        model = LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        )
        model.fit(X_train, y_train)
        os.makedirs(os.path.dirname(os.path.abspath(model_json_path)), exist_ok=True)
        save_model_json(model, model_json_path)
