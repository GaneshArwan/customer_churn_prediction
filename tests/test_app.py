import pytest
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_and_prepare_preprocessor, preprocess_input

def test_load_and_prepare_preprocessor(tmp_path):
    """Test that the preprocessor loads data and extracts the expected columns and scaler."""
    # Create fake dataset
    df = pd.DataFrame({
        'customerID': ['001', '002', '003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No'],
        'tenure': [1, 20, 5],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [50.0, 80.0, 20.0],
        'TotalCharges': [50.0, 1600.0, 100.0],
        'Churn': ['No', 'Yes', 'No'],
    })
    data_path = tmp_path / "telco_customer_churn.csv"
    df.to_csv(data_path, index=False)
    
    scaler, cat_cols, num_cols, expected_cols, template_X = load_and_prepare_preprocessor(str(data_path))
    
    # Check outputs
    assert scaler is not None
    assert 'gender' in cat_cols
    assert 'tenure' in num_cols
    assert len(expected_cols) > len(cat_cols) # Dummy variables expand columns
    assert not template_X.empty

def test_preprocess_input(tmp_path):
    """Test that a single row of input is processed identically to the training set."""
    # Create fake dataset
    df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [10, 20],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['Yes', 'Yes'],
        'StreamingMovies': ['No', 'Yes'],
        'Contract': ['Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [50.0, 80.0],
        'TotalCharges': [500.0, 1600.0],
        'Churn': ['No', 'Yes'],
    })
    data_path = tmp_path / "telco_customer_churn.csv"
    df.to_csv(data_path, index=False)
    
    scaler, cat_cols, num_cols, expected_cols, template_X = load_and_prepare_preprocessor(str(data_path))
    
    # Create a test input
    input_df = pd.DataFrame([{
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 15, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.0, 'TotalCharges': 1425.0
    }])
    
    X_processed = preprocess_input(input_df, scaler, cat_cols, num_cols, expected_cols, template_X)
    
    assert X_processed.shape[0] == 1
    assert X_processed.shape[1] == len(expected_cols)
    assert list(X_processed.columns) == expected_cols
