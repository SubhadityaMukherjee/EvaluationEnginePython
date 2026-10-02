import math

import numpy as np
import pytest

from src.runs.metrics import (
    classification_metrics,
    kb_relative_information,
    regression_metrics,
)


class TestRegressionMetrics:
    def test_golden_values(self):
        y_true = np.array([3.0, 4.0])
        y_pred = np.array([2.5, 4.5])
        y_train = np.array([3.0, 3.0, 5.0])  # prior = 11/3

        m = regression_metrics(y_true, y_pred, y_train)

        assert m["mean_absolute_error"] == pytest.approx(0.5)
        assert m["root_mean_squared_error"] == pytest.approx(0.5)
        assert m["mean_prior_absolute_error"] == pytest.approx(0.5)
        assert m["root_mean_prior_squared_error"] == pytest.approx(np.sqrt(5.0 / 18.0))
        assert m["relative_absolute_error"] == pytest.approx(1.0)
        assert m["root_relative_squared_error"] == pytest.approx(0.9486833)
        assert m["number_of_instances"] == 2.0

    def test_zero_prior_error_gives_zero_relative_errors(self):
        y_true = np.array([5.0, 5.0])
        y_pred = np.array([5.0, 5.0])
        y_train = np.array([5.0, 5.0])  # constant target

        m = regression_metrics(y_true, y_pred, y_train)

        assert m["mean_prior_absolute_error"] == 0.0
        assert m["relative_absolute_error"] == 0.0
        assert m["root_relative_squared_error"] == 0.0


class TestKbRelativeInformation:
    def test_perfect_predictions_score_one(self):
        y_true = np.array([0, 1])
        conf = np.array([[1.0, 0.0], [0.0, 1.0]])
        prior = np.array([0.5, 0.5])

        assert kb_relative_information(y_true, conf, prior) == pytest.approx(1.0)

    def test_uninformative_predictions_score_zero(self):
        y_true = np.array([0, 1])
        conf = np.array([[0.5, 0.5], [0.5, 0.5]])
        prior = np.array([0.5, 0.5])

        assert kb_relative_information(y_true, conf, prior) == pytest.approx(0.0)

    def test_zero_prior_returns_zero(self):
        assert (
            kb_relative_information(np.array([0]), np.array([[1.0]]), np.array([0.0]))
            == 0.0
        )

    def test_deterministic_prior_returns_zero(self):
        assert (
            kb_relative_information(
                np.array([0, 0]),
                np.array([[0.9, 0.1], [0.8, 0.2]]),
                np.array([1.0, 0.0]),
            )
            == 0.0
        )

    def test_worse_than_prior_is_negative(self):
        y_true = np.array([0, 1])
        # confidence actively contradicts the truth
        conf = np.array([[0.0, 1.0], [1.0, 0.0]])
        prior = np.array([0.5, 0.5])

        assert kb_relative_information(y_true, conf, prior) < 0.0


class TestClassificationMetrics:
    @pytest.fixture
    def args(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        conf = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.05, 0.95]])
        class_names = ["neg", "pos"]
        y_train = ["neg", "pos", "pos", "neg"]
        return y_true, y_pred, conf, class_names, y_train

    def test_golden_values(self, args):
        y_true, y_pred, conf, class_names, y_train = args

        m = classification_metrics(y_true, y_pred, conf, class_names, y_train)

        assert m["predictive_accuracy"] == pytest.approx(0.75)
        assert m["kappa"] == pytest.approx(0.5)
        assert m["confusion_matrix"] == [[1, 1], [0, 2]]
        assert m["number_of_instances"] == 4.0
        assert m["area_under_roc_curve"] == pytest.approx(1.0)

    def test_string_labels_are_encoded(self, args):
        _, _, conf, class_names, y_train = args
        y_true = np.array(["neg", "neg", "pos", "pos"], dtype=object)
        y_pred = np.array(["neg", "pos", "pos", "pos"], dtype=object)

        m = classification_metrics(y_true, y_pred, conf, class_names, y_train)

        assert m["predictive_accuracy"] == pytest.approx(0.75)

    def test_per_class_block(self, args):
        y_true, y_pred, conf, class_names, y_train = args

        m = classification_metrics(y_true, y_pred, conf, class_names, y_train)

        per_class = m["_per_class"]
        assert per_class["instances_per_class"] == [2, 2]
        # class 'neg' (index 0): tp=1, fp=0, fn=1 → precision 1.0, recall 0.5
        assert per_class["precision"][0] == pytest.approx(1.0)
        assert per_class["recall"][0] == pytest.approx(0.5)
        assert per_class["f_measure"][0] == pytest.approx(2 / 3)

    def test_cost_matrix_adds_cost_metrics(self, args):
        y_true, y_pred, conf, class_names, y_train = args

        m = classification_metrics(
            y_true,
            y_pred,
            conf,
            class_names,
            y_train,
            cost_matrix=np.array([[0.0, 1.0], [5.0, 0.0]]),
        )

        assert m["total_cost"] == pytest.approx(1.0)
        assert m["average_cost"] == pytest.approx(0.25)

    def test_single_class_target_yields_nan_auroc(self):
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        conf = np.array([[0.8, 0.2], [0.6, 0.4]])

        m = classification_metrics(y_true, y_pred, conf, ["a", "b"], ["a", "a"])

        # sklearn emits nan (not an exception) when only one class is present
        assert math.isnan(m["area_under_roc_curve"])
        assert all(math.isnan(a) for a in m["_per_class"]["auroc"])

    def test_prior_entropy(self, args):
        _, _, conf, class_names, y_train = args

        m = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), conf, class_names, y_train
        )

        assert m["prior_entropy"] == pytest.approx(1.0)
