import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import load_and_preprocess_data
from evaluate import evaluate_model
from inference import save_model_json


@pytest.fixture
def mock_evaluation_env(tmp_path):
    """Set up an isolated project environment with a JSON model and dummy data."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    np.random.seed(42)
    df = pd.DataFrame(
        {
            'customerID': [f'{i:03d}' for i in range(50)],
            'tenure': np.random.randint(1, 72, 50),
            'MonthlyCharges': np.round(np.random.uniform(20, 100, 50), 2),
            'TotalCharges': np.round(np.random.uniform(100, 5000, 50), 2).astype(str),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 50),
            'Churn': np.random.choice(['Yes', 'No'], 50, p=[0.27, 0.73]),
        }
    )
    csv_path = data_dir / "telco_customer_churn.csv"
    df.to_csv(csv_path, index=False)

    X_train, _, y_train, _ = load_and_preprocess_data(str(csv_path))
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    save_model_json(model, str(models_dir / "lr_model.json"))

    return tmp_path


class TestEvaluateModel:
    """Tests for evaluate.py evaluation logic."""

    def test_evaluate_model_runs_without_error(
        self, mock_evaluation_env, monkeypatch, capsys
    ):
        """Verify evaluate_model() executes end-to-end, prints metrics, and saves plot."""
        fake_script = str(mock_evaluation_env / "src" / "evaluate.py")
        monkeypatch.setattr('evaluate.os.path.abspath', lambda p: fake_script)

        evaluate_model()

        captured = capsys.readouterr()
        assert "Accuracy:" in captured.out
        assert "Classification Report:" in captured.out

        plot_path = mock_evaluation_env / "plots" / "confusion_matrix.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
