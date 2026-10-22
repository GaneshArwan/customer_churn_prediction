import pytest
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda import main as eda_main

_real_dirname = os.path.dirname


@pytest.fixture
def eda_data_env(tmp_path):
    """Set up a fake project with src/ subdirectory and dummy Telco CSV."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    np.random.seed(42)
    df = pd.DataFrame({
        'customerID': [f'{i:03d}' for i in range(30)],
        'tenure': np.random.randint(1, 72, 30),
        'MonthlyCharges': np.round(np.random.uniform(20, 100, 30), 2),
        'TotalCharges': np.round(np.random.uniform(100, 5000, 30), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 30),
        'Churn': np.random.choice(['Yes', 'No'], 30, p=[0.27, 0.73]),
    })
    csv_path = data_dir / "telco_customer_churn.csv"
    df.to_csv(csv_path, index=False)

    return tmp_path


class TestEDA:
    """Tests for eda.py exploratory data analysis."""

    def test_eda_generates_all_plots(self, eda_data_env, monkeypatch):
        """Verify eda.main() generates the 4 expected plot files."""
        # eda.main() resolves its base directory via:
        #   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Patch the module's __file__ attribute so abspath(__file__) points
        # into the tmp tree, without replacing os.path.abspath globally
        # (which would break matplotlib's internal font resolution).
        import eda
        fake_script = str(eda_data_env / "src" / "eda.py")
        monkeypatch.setattr(eda, '__file__', fake_script)

        eda_main()

        plots_dir = eda_data_env / "plots"
        expected_plots = [
            'churn_distribution.png',
            'tenure_vs_churn.png',
            'monthly_charges_vs_churn.png',
            'contract_vs_churn.png',
        ]
        for plot_name in expected_plots:
            plot_path = plots_dir / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"
            assert plot_path.stat().st_size > 0, f"Empty plot: {plot_name}"

    def test_eda_creates_plots_dir(self, tmp_path, monkeypatch):
        """Verify eda.main() creates the plots/ directory if it does not exist."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        df = pd.DataFrame({
            'customerID': ['001', '002', '003'],
            'tenure': [1, 10, 5],
            'MonthlyCharges': [50.0, 60.5, 70.0],
            'TotalCharges': [50.0, 605.0, 350.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No'],
        })
        csv_path = data_dir / "telco_customer_churn.csv"
        df.to_csv(csv_path, index=False)

        # Patch __file__ instead of os.path.abspath to avoid breaking matplotlib.
        import eda
        fake_script = str(tmp_path / "src" / "eda.py")
        monkeypatch.setattr(eda, '__file__', fake_script)

        eda_main()

        plots_dir = tmp_path / "plots"
        assert plots_dir.exists()
