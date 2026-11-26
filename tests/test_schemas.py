import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas import CustomerProfile


def _valid_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 906.0,
    }


def test_customer_profile_accepts_valid_payload():
    """Verify CustomerProfile accepts valid Telco customer attributes."""
    profile = CustomerProfile(**_valid_payload())
    assert profile.tenure == 12
    assert profile.Contract == "Month-to-month"


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenure": 0, "TotalCharges": 0.0, "MonthlyCharges": 0.0},
        {"tenure": 120, "MonthlyCharges": 1000.0, "TotalCharges": 120000.0, "SeniorCitizen": 1},
    ],
)
def test_customer_profile_boundary_values(overrides):
    """Verify boundary values for numeric and binary fields pass validation."""
    payload = {**_valid_payload(), **overrides}
    profile = CustomerProfile(**payload)
    for key, val in overrides.items():
        assert getattr(profile, key) == val


@pytest.mark.parametrize(
    "invalid_override",
    [
        {"gender": "Unknown"},
        {"SeniorCitizen": 2},
        {"SeniorCitizen": -1},
        {"tenure": -1},
        {"tenure": 121},
        {"tenure": 12.5},
        {"Contract": "Weekly"},
        {"PaymentMethod": "Cash"},
        {"MonthlyCharges": -0.01},
        {"MonthlyCharges": 1000.01},
        {"MonthlyCharges": float("nan")},
        {"MonthlyCharges": float("inf")},
        {"TotalCharges": -0.01},
        {"TotalCharges": 120000.01},
        {"TotalCharges": float("nan")},
        {"TotalCharges": float("inf")},
        {"InternetService": "Satellite"},
        {"unexpected_key": "injected"},
    ],
)
def test_customer_profile_rejects_invalid_inputs(invalid_override):
    """Verify CustomerProfile rejects out-of-range, non-finite, unknown, or extra fields."""
    payload = {**_valid_payload(), **invalid_override}
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)


def test_customer_profile_rejects_missing_required_field():
    """Verify CustomerProfile rejects payloads missing required fields."""
    payload = _valid_payload()
    del payload["Contract"]
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)
