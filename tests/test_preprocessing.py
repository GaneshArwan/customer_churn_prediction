import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import load_and_preprocess_data

@pytest.fixture
def dummy_data_path(tmp_path):
    df = pd.DataFrame({
        'customerID': ['001', '002', '003', '004'],
        'tenure': [1, 10, 5, 20],
        'MonthlyCharges': [50.0, 60.5, 70.0, ' '], # includes a space (missing val)
        'TotalCharges': ['50.0', '605.0', '350.0', ' '], # space to test errors='coerce'
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
        'Churn': ['No', 'Yes', 'No', 'No']
    })
    
    csv_path = tmp_path / "dummy_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

def test_load_and_preprocess_data(dummy_data_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dummy_data_path)
    
    # 1 row should be dropped because of missing TotalCharges
    total_samples = len(X_train) + len(X_test)
    assert total_samples == 3
    
    # customerID should be dropped
    assert 'customerID' not in X_train.columns
    
    # Check if target is encoded correctly
    assert set(y_train).union(set(y_test)).issubset({0, 1})
    
    # Check scaling (mean should be approx 0)
    assert np.isclose(X_train['tenure'].mean(), 0, atol=1.0)
    
    # Check one hot encoding (Contract columns)
    assert 'Contract_One year' in X_train.columns
    assert 'Contract_Two year' in X_train.columns
