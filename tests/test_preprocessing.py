import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import load_and_preprocess_data


@pytest.fixture
def dummy_data_path(tmp_path):
    df = pd.DataFrame(
        {
            'customerID': ['001', '002', '003', '004', '005', '006'],
            'tenure': [1, 10, 5, 20, 30, 40],
            'MonthlyCharges': [50.0, 60.5, 70.0, 80.0, 90.0, 45.0],
            'TotalCharges': ['50.0', '605.0', '350.0', ' ', '2700.0', '1800.0'],
            'Contract': [
                'Month-to-month',
                'One year',
                'Two year',
                'Month-to-month',
                'One year',
                'Two year',
            ],
            'Churn': ['No', 'Yes', 'No', 'No', 'Yes', 'No'],
        }
    )

    csv_path = tmp_path / "dummy_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_preprocess_data(dummy_data_path):
    """Verify dropping invalid TotalCharges, identifier removal, target encoding, and scaling."""
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dummy_data_path)

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 5
    assert 'customerID' not in X_train.columns
    assert set(y_train).union(set(y_test)).issubset({0, 1})
    assert np.isclose(X_train['tenure'].mean(), 0.0, atol=1e-9)
    assert np.isclose(X_train['MonthlyCharges'].mean(), 0.0, atol=1e-9)
    assert 'Contract_One year' in X_train.columns
    assert 'Contract_Two year' in X_train.columns


def test_load_and_preprocess_data_no_leakage_across_test_outliers(tmp_path):
    """Verify scaling parameters depend only on the training split, not test outliers."""
    base_df = pd.DataFrame(
        {
            'tenure': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            'MonthlyCharges': [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
            'TotalCharges': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
            'Contract': ['Month-to-month', 'One year'] * 5,
            'Churn': ['No', 'Yes'] * 5,
        }
    )
    p1 = tmp_path / "d1.csv"
    base_df.to_csv(p1, index=False)
    X_train_1, X_test_1, _, _ = load_and_preprocess_data(str(p1))

    # Modify a test-split row (index 8 in random_state=42 split on 10 rows -> let's identify test idx)
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(base_df.index, test_size=0.2, random_state=42)
    modified_df = base_df.copy()
    modified_df.loc[test_idx[0], 'MonthlyCharges'] = 99999.0
    p2 = tmp_path / "d2.csv"
    modified_df.to_csv(p2, index=False)
    X_train_2, _, _, _ = load_and_preprocess_data(str(p2))

    assert len(train_idx) == len(X_train_1)
    pd.testing.assert_frame_equal(X_train_1, X_train_2)
