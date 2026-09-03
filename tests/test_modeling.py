import pytest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train_model import train_model, main


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
        assert hasattr(model, 'coef_')  # LogisticRegression exposes coef_

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
