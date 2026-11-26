import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train_model import main, train_model


class TestTrainModel:
    """Tests for train_model.py training logic."""

    def test_train_model_returns_fitted_estimator(self):
        """Verify train_model returns a fitted sklearn estimator with predict and predict_proba."""
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)

        model = train_model(X_train, y_train)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'coef_')

    def test_train_model_uses_balanced_weights(self):
        """Verify the trained model uses balanced class weights."""
        np.random.seed(42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)

        model = train_model(X_train, y_train)
        assert model.class_weight == 'balanced'

    def test_train_model_predictions_are_binary(self):
        """Verify predictions are strictly 0 or 1."""
        np.random.seed(42)
        X_train = np.random.rand(80, 4)
        y_train = np.random.randint(0, 2, 80)

        model = train_model(X_train, y_train)
        preds = model.predict(X_train)
        assert set(preds).issubset({0, 1})

    def test_train_model_predict_proba_sums_to_one(self):
        """Verify predict_proba outputs sum to 1.0 for each sample."""
        np.random.seed(42)
        X_train = np.random.rand(80, 4)
        y_train = np.random.randint(0, 2, 80)

        model = train_model(X_train, y_train)
        probas = model.predict_proba(X_train[:10])
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_main_saves_pickle_and_json_artifacts(self, tmp_path, monkeypatch):
        """Verify train_model.main() saves both rf_model.pkl and lr_model.json."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        np.random.seed(42)
        df = pd.DataFrame(
            {
                'customerID': [f'{i:03d}' for i in range(40)],
                'tenure': np.random.randint(1, 72, 40),
                'MonthlyCharges': np.round(np.random.uniform(20, 100, 40), 2),
                'TotalCharges': np.round(np.random.uniform(100, 5000, 40), 2).astype(str),
                'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 40),
                'Churn': np.random.choice(['Yes', 'No'], 40, p=[0.3, 0.7]),
            }
        )
        df.to_csv(data_dir / "telco_customer_churn.csv", index=False)

        monkeypatch.setattr('train_model.os.path.abspath', lambda p: str(src_dir / "train_model.py"))
        main()

        assert (tmp_path / "models" / "rf_model.pkl").exists()
        assert (tmp_path / "models" / "lr_model.json").exists()
