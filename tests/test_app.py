import importlib
import os
import sys
import types
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import (
    load_and_prepare_preprocessor,
    load_model_json,
    preprocess_input,
    save_model_json,
)


def _make_sample_df():
    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003"],
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "No"],
            "tenure": [1, 20, 5],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["Yes", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [50.0, 80.0, 20.0],
            "TotalCharges": ["50.0", "1600.0", "100.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def _make_stub_streamlit(called: dict, sidebar_values: dict | None = None):
    vals = sidebar_values or {}
    stub_st = types.ModuleType("streamlit")
    stub_st.cache_data = lambda fn: fn
    stub_st.cache_resource = lambda fn: fn
    stub_st.set_page_config = lambda **kwargs: called.__setitem__(
        "set_page_config", True
    )
    stub_st.markdown = lambda *args, **kwargs: None
    stub_st.title = lambda *args, **kwargs: None
    stub_st.header = lambda *args, **kwargs: None
    stub_st.subheader = lambda *args, **kwargs: None
    stub_st.write = lambda *args, **kwargs: None
    stub_st.error = lambda msg, **kwargs: called.setdefault("errors", []).append(msg)
    stub_st.warning = lambda *args, **kwargs: None
    stub_st.success = lambda *args, **kwargs: None
    stub_st.image = lambda *args, **kwargs: None
    stub_st.stop = lambda: called.__setitem__("stopped", True)
    stub_st.tabs = lambda labels: [nullcontext() for _ in labels]
    stub_st.columns = lambda spec: [
        nullcontext() for _ in (range(spec) if isinstance(spec, int) else spec)
    ]
    stub_st.sidebar = nullcontext()
    stub_st.selectbox = lambda label, options, **kwargs: vals.get(label, options[0])
    stub_st.slider = lambda label, min_v, max_v, default, **kwargs: vals.get(
        label, default
    )
    stub_st.number_input = lambda label, min_v, max_v, default, **kwargs: vals.get(
        label, default
    )
    return stub_st


def test_load_and_prepare_preprocessor(tmp_path):
    """Test that the preprocessor loads data and extracts expected columns and scaler."""
    data_path = tmp_path / "telco_customer_churn.csv"
    _make_sample_df().to_csv(data_path, index=False)

    scaler, cat_cols, num_cols, expected_cols, template_X = (
        load_and_prepare_preprocessor(str(data_path))
    )

    assert scaler is not None
    assert "gender" in cat_cols
    assert "tenure" in num_cols
    assert len(expected_cols) > len(cat_cols)
    assert not template_X.empty


def test_load_and_prepare_preprocessor_single_row(tmp_path):
    """Verify len(X) < 2 fallback fits scaler without splitting."""
    data_path = tmp_path / "single.csv"
    _make_sample_df().iloc[[0]].to_csv(data_path, index=False)

    scaler, cat_cols, num_cols, expected_cols, template_X = (
        load_and_prepare_preprocessor(str(data_path))
    )
    assert scaler is not None
    assert len(template_X) == 1
    assert isinstance(expected_cols, list)


def test_load_and_prepare_preprocessor_missing_file_raises(tmp_path):
    """Verify FileNotFoundError is raised when dataset path does not exist."""
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_and_prepare_preprocessor(str(missing))


def test_preprocess_input(tmp_path):
    """Test that a single inference row is aligned to expected training columns."""
    data_path = tmp_path / "telco_customer_churn.csv"
    _make_sample_df().to_csv(data_path, index=False)

    scaler, cat_cols, num_cols, expected_cols, template_X = (
        load_and_prepare_preprocessor(str(data_path))
    )

    input_df = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 15,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0,
                "TotalCharges": 1425.0,
            }
        ]
    )

    X_processed = preprocess_input(
        input_df, scaler, cat_cols, num_cols, expected_cols, template_X
    )

    assert X_processed.shape == (1, len(expected_cols))
    assert list(X_processed.columns) == expected_cols
    assert X_processed["Contract_One year"].iloc[0] == 0
    assert X_processed["Contract_Two year"].iloc[0] == 0


def test_preprocess_input_no_drop_first_baseline_drift(tmp_path):
    """Regression: Contract encoding must not depend on template_df first row."""
    data_path = tmp_path / "telco_customer_churn.csv"
    _make_sample_df().to_csv(data_path, index=False)

    scaler, cat_cols, num_cols, expected_cols, template_X = (
        load_and_prepare_preprocessor(str(data_path))
    )

    base_row = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }

    alt_template = template_X.iloc[[2]].copy()

    row_one_year = pd.DataFrame([base_row])
    res_default = preprocess_input(
        row_one_year, scaler, cat_cols, num_cols, expected_cols, template_X
    )
    res_alt = preprocess_input(
        row_one_year, scaler, cat_cols, num_cols, expected_cols, alt_template
    )

    assert res_default["Contract_One year"].iloc[0] == 1
    assert res_default["Contract_Two year"].iloc[0] == 0
    pd.testing.assert_frame_equal(res_default, res_alt)

    row_two_year = pd.DataFrame([{**base_row, "Contract": "Two year"}])
    res_two = preprocess_input(
        row_two_year, scaler, cat_cols, num_cols, expected_cols, alt_template
    )
    assert res_two["Contract_One year"].iloc[0] == 0
    assert res_two["Contract_Two year"].iloc[0] == 1


def test_save_and_load_model_json_equivalence(tmp_path):
    """Verify JSON model serialization reproduces exact LogisticRegression predictions."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(40, 4), columns=["a", "b", "c", "d"])
    y = np.random.randint(0, 2, 40)
    model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=500)
    model.fit(X, y)

    json_path = tmp_path / "lr_model.json"
    save_model_json(model, str(json_path))
    loaded = load_model_json(str(json_path))

    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), atol=1e-12)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_load_model_json_missing_raises(tmp_path):
    """Verify missing JSON model artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_json(str(tmp_path / "missing.json"))


def test_app_import_has_no_ui_side_effects(monkeypatch):
    """Verify importing app.py does not execute Streamlit UI rendering at import time."""
    called = {"set_page_config": False}
    stub_st = _make_stub_streamlit(called)
    monkeypatch.setitem(sys.modules, "streamlit", stub_st)

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    assert callable(app_module.main)
    assert app_module.load_and_prepare_preprocessor is load_and_prepare_preprocessor
    assert app_module.preprocess_input is preprocess_input
    assert called["set_page_config"] is False


def test_app_classify_risk_and_main_execution(tmp_path, monkeypatch):
    """Verify classify_risk tiers and app.main() execution with mocked Streamlit."""
    called = {"set_page_config": False, "errors": [], "stopped": False}
    stub_st = _make_stub_streamlit(called)
    monkeypatch.setitem(sys.modules, "streamlit", stub_st)

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    assert app_module.classify_risk(0.75) == ("risk-high", "HIGH RISK")
    assert app_module.classify_risk(0.50) == ("risk-medium", "MEDIUM RISK")
    assert app_module.classify_risk(0.20) == ("risk-low", "LOW RISK")

    data_path = tmp_path / "telco_customer_churn.csv"
    _make_sample_df().to_csv(data_path, index=False)
    scaler, cat_cols, num_cols, expected_cols, _ = load_and_prepare_preprocessor(
        str(data_path)
    )

    X_dummy = pd.DataFrame(
        np.random.RandomState(42).rand(20, len(expected_cols)), columns=expected_cols
    )
    y_dummy = np.array([0, 1] * 10)
    lr = LogisticRegression(random_state=42, class_weight="balanced", max_iter=500).fit(
        X_dummy, y_dummy
    )
    model_json = tmp_path / "lr_model.json"
    save_model_json(lr, str(model_json))

    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "DATA_PATH", str(data_path))
    monkeypatch.setattr(app_module, "MODEL_JSON_PATH", str(model_json))

    app_module.main()
    assert called["set_page_config"] is True
    assert called["stopped"] is False


def test_app_main_handles_missing_data_and_model(tmp_path, monkeypatch):
    """Verify app.main() shows generic error messages without crashing when files are missing."""
    called = {"set_page_config": False, "errors": [], "stopped": False}
    stub_st = _make_stub_streamlit(called)
    monkeypatch.setitem(sys.modules, "streamlit", stub_st)

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    monkeypatch.setattr(app_module, "DATA_PATH", str(tmp_path / "missing.csv"))
    app_module.main()
    assert called["stopped"] is True
    assert len(called["errors"]) == 1
    assert str(tmp_path) not in called["errors"][0]

    # Existing dataset but missing model artifact
    data_path = tmp_path / "telco_customer_churn.csv"
    _make_sample_df().to_csv(data_path, index=False)
    called["errors"].clear()
    called["stopped"] = False
    monkeypatch.setattr(app_module, "DATA_PATH", str(data_path))
    monkeypatch.setattr(app_module, "MODEL_JSON_PATH", str(tmp_path / "missing.json"))
    app_module.main()
    assert called["stopped"] is True
    assert len(called["errors"]) == 1
    assert str(tmp_path) not in called["errors"][0]
