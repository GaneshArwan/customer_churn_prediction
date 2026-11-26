import os
import sys
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generate_report import (
    compute_business_impact,
    create_metric_table,
    ieee_header_footer,
    save_confusion_matrix,
    save_feature_importance,
    save_lr_coefficients,
    save_pr_curve,
    save_roc_curve,
)


@pytest.fixture
def trained_models():
    """Train small RF and LR models on synthetic data for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=500)
    lr.fit(X, y)

    return rf, lr, X, y


class TestSaveConfusionMatrix:
    """Tests for the save_confusion_matrix helper."""

    def test_creates_image_file(self, tmp_path, trained_models):
        rf, _, X, y = trained_models
        y_pred = rf.predict(X)
        out_path = str(tmp_path / "cm.png")
        save_confusion_matrix(y, y_pred, "Test CM", out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

    def test_handles_perfect_predictions(self, tmp_path):
        y = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        out_path = str(tmp_path / "perfect_cm.png")
        save_confusion_matrix(y, y_pred, "Perfect", out_path)
        assert os.path.exists(out_path)


class TestSaveROCCurve:
    """Tests for the save_roc_curve helper."""

    def test_creates_roc_image(self, tmp_path, trained_models):
        rf, lr, X, y = trained_models
        models_dict = {"RF": rf, "LR": lr}
        out_path = str(tmp_path / "roc.png")
        save_roc_curve(models_dict, X, y, "Test ROC", out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

    def test_single_model_roc(self, tmp_path, trained_models):
        rf, _, X, y = trained_models
        models_dict = {"RF": rf}
        out_path = str(tmp_path / "roc_single.png")
        save_roc_curve(models_dict, X, y, "Single ROC", out_path)
        assert os.path.exists(out_path)

    def test_decision_function_branch(self, tmp_path, trained_models):
        _, _, X, y = trained_models
        svc = LinearSVC(random_state=42, dual="auto").fit(X, y)
        out_path = str(tmp_path / "roc_svc.png")
        save_roc_curve({"SVC": svc}, X, y, "SVC ROC", out_path)
        assert os.path.exists(out_path)


class TestSavePRCurve:
    """Tests for the save_pr_curve helper."""

    def test_creates_pr_image(self, tmp_path, trained_models):
        rf, lr, X, y = trained_models
        out_path = str(tmp_path / "pr.png")
        save_pr_curve({"RF": rf, "LR": lr}, X, y, "Test PR", out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


class TestSaveFeatureImportanceAndCoefficients:
    """Tests for feature importance and Logistic Regression coefficient charts."""

    def test_creates_fi_image(self, tmp_path, trained_models):
        rf, _, X, y = trained_models
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        out_path = str(tmp_path / "fi.png")
        save_feature_importance(rf, feature_names, "Test FI", out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

    def test_top_10_with_fewer_features(self, tmp_path):
        """Verify it works when there are fewer than 10 features."""
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        feature_names = ["a", "b", "c"]
        out_path = str(tmp_path / "fi_small.png")
        save_feature_importance(rf, feature_names, "Small FI", out_path)
        assert os.path.exists(out_path)

    def test_save_lr_coefficients_image(self, tmp_path, trained_models):
        """Verify save_lr_coefficients renders signed bars for LR model."""
        _, lr, X, _ = trained_models
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        out_path = str(tmp_path / "lr_coef.png")
        save_lr_coefficients(lr, feature_names, "Test LR Coefs", out_path, top_n=4)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


class TestComputeBusinessImpact:
    """Tests for the compute_business_impact economics helper."""

    def test_exact_arithmetic_and_breakeven(self):
        res = compute_business_impact(tp=200, fp=100, ltv=780.0, campaign_cost=50.0, save_rate=0.50)
        assert res["targeted"] == 300
        assert res["campaign_spend"] == pytest.approx(15000.0)
        assert res["retained_value"] == pytest.approx(78000.0)
        assert res["net_profit"] == pytest.approx(63000.0)
        assert res["breakeven_save_rate"] == pytest.approx(15000.0 / (200 * 780.0))

    def test_zero_tp_returns_inf_breakeven(self):
        res = compute_business_impact(tp=0, fp=10, ltv=780.0, campaign_cost=50.0, save_rate=0.35)
        assert res["retained_value"] == 0.0
        assert res["net_profit"] == pytest.approx(-500.0)
        assert res["breakeven_save_rate"] == float("inf")


class TestCreateMetricTable:
    """Tests for the create_metric_table helper."""

    def test_returns_table_object(self, trained_models):
        rf, _, X, y = trained_models
        y_pred = rf.predict(X)
        report_dict = classification_report(y, y_pred, output_dict=True)
        table = create_metric_table(report_dict)
        assert table is not None
        assert hasattr(table, '_cellvalues')

    def test_table_has_correct_row_count(self, trained_models):
        rf, _, X, y = trained_models
        y_pred = rf.predict(X)
        report_dict = classification_report(y, y_pred, output_dict=True)
        table = create_metric_table(report_dict)
        assert len(table._cellvalues) == 5


class TestIEEEHeaderFooter:
    """Tests for the ieee_header_footer canvas callback."""

    def test_ieee_header_footer_page_one_omits_running_header(self):
        """Verify page 1 draws only the page number footer."""
        canvas = MagicMock()
        doc = MagicMock()
        doc.page = 1
        ieee_header_footer(canvas, doc)
        canvas.saveState.assert_called_once()
        canvas.drawCentredString.assert_called_once()
        assert canvas.drawCentredString.call_args.args[2] == "1"
        canvas.restoreState.assert_called_once()

    def test_ieee_header_footer_renders_page_number(self):
        """Verify page 2+ draws both running header and page number."""
        canvas = MagicMock()
        doc = MagicMock()
        doc.page = 2
        ieee_header_footer(canvas, doc)
        canvas.saveState.assert_called_once()
        assert canvas.drawCentredString.call_count == 2
        assert any(call.args[2] == "2" for call in canvas.drawCentredString.call_args_list)
        canvas.restoreState.assert_called_once()
