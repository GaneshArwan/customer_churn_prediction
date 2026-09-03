import pytest
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generate_report import (
    save_confusion_matrix,
    save_roc_curve,
    save_feature_importance,
    create_metric_table,
    header_footer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


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
        assert os.path.getsize(out_path) > 1000  # a real PNG is > 1KB

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


class TestSaveFeatureImportance:
    """Tests for the save_feature_importance helper."""

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


class TestCreateMetricTable:
    """Tests for the create_metric_table helper."""

    def test_returns_table_object(self, trained_models):
        rf, _, X, y = trained_models
        y_pred = rf.predict(X)
        report_dict = classification_report(y, y_pred, output_dict=True)
        table = create_metric_table(report_dict)
        # ReportLab Table object
        assert table is not None
        assert hasattr(table, '_cellvalues')

    def test_table_has_correct_row_count(self, trained_models):
        rf, _, X, y = trained_models
        y_pred = rf.predict(X)
        report_dict = classification_report(y, y_pred, output_dict=True)
        table = create_metric_table(report_dict)
        # 1 header + 2 classes + 2 averages = 5 rows
        assert len(table._cellvalues) == 5


class TestHeaderFooter:
    """Tests for the header_footer canvas callback."""

    def test_header_footer_callable(self):
        """Verify header_footer is callable (it's used as a callback by reportlab)."""
        assert callable(header_footer)
