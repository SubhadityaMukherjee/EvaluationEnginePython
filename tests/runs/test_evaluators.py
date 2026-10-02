import pandas as pd
import pytest

from src.exceptions import PredictionValidationError
from src.models import EstimationProcedureType
from src.runs.evaluators import (
    TASK_TYPE_ID_TO_TASK_TYPE,
    TaskType,
    evaluate_batch,
    evaluate_run,
    evaluate_stream,
    evaluate_survival,
)

# ============================================================================
# Shared fixtures: a 6-row dataset, 1 repeat x 2 folds CV splits.
# ============================================================================


@pytest.fixture
def dataset_df():
    return pd.DataFrame(
        {
            "outlook": pd.Categorical(
                ["sunny", "sunny", "overcast", "rainy", "rainy", "overcast"],
                categories=["sunny", "overcast", "rainy"],
            ),
            "temp": [75.0, 80.0, 83.0, 70.0, 68.0, 72.0],
            "play": pd.Categorical(
                ["yes", "no", "yes", "yes", "no", "no"],
                categories=["no", "yes"],
            ),
        }
    )


@pytest.fixture
def splits_df():
    return pd.DataFrame(
        [
            ("TRAIN", 0, 0, 0),
            ("TRAIN", 1, 0, 0),
            ("TRAIN", 2, 0, 0),
            ("TEST", 3, 0, 0),
            ("TEST", 4, 0, 0),
            ("TEST", 5, 0, 0),
            ("TEST", 0, 0, 1),
            ("TEST", 1, 0, 1),
            ("TEST", 2, 0, 1),
            ("TRAIN", 3, 0, 1),
            ("TRAIN", 4, 0, 1),
            ("TRAIN", 5, 0, 1),
        ],
        columns=["type", "rowid", "repeat", "fold"],
    )


@pytest.fixture
def predictions_df():
    # True labels: yes no yes | yes no no
    return pd.DataFrame(
        [
            # fold 0 tests rows 3,4,5
            (3, 0, 0, "yes", 0.1, 0.9),
            (4, 0, 0, "no", 0.9, 0.1),
            (5, 0, 0, "yes", 0.2, 0.8),  # wrong (true label: no)
            # fold 1 tests rows 0,1,2 — all correct
            (0, 0, 1, "yes", 0.05, 0.95),
            (1, 0, 1, "no", 0.85, 0.15),
            (2, 0, 1, "yes", 0.1, 0.9),
        ],
        columns=[
            "row_id",
            "repeat",
            "fold",
            "prediction",
            "confidence.no",
            "confidence.yes",
        ],
    )


class TestEvaluateBatchClassification:
    def test_global_accuracy_golden_value(self, dataset_df, splits_df, predictions_df):
        _, global_scores, _ = evaluate_batch(
            dataset_df, splits_df, predictions_df, "play", TaskType.CLASSIFICATION
        )

        by_name = {s.function: s for s in global_scores}
        assert by_name["predictive_accuracy"].value == pytest.approx(5 / 6)

    def test_per_cell_scores_carry_repeat_and_fold(
        self, dataset_df, splits_df, predictions_df
    ):
        per_cell, global_scores, _ = evaluate_batch(
            dataset_df, splits_df, predictions_df, "play", TaskType.CLASSIFICATION
        )

        assert per_cell, "expected per-fold scores"
        assert all(s.fold is not None for s in per_cell)
        assert {s.fold for s in per_cell} == {0, 1}
        assert all(s.fold is None for s in global_scores)

    def test_fold0_accuracy(self, dataset_df, splits_df, predictions_df):
        per_cell, _, _ = evaluate_batch(
            dataset_df, splits_df, predictions_df, "play", TaskType.CLASSIFICATION
        )

        fold0_acc = [
            s for s in per_cell if s.function == "predictive_accuracy" and s.fold == 0
        ]
        assert len(fold0_acc) == 1
        assert fold0_acc[0].value == pytest.approx(2 / 3)

    def test_global_stdev_computed_over_folds(
        self, dataset_df, splits_df, predictions_df
    ):
        _, global_scores, _ = evaluate_batch(
            dataset_df, splits_df, predictions_df, "play", TaskType.CLASSIFICATION
        )

        acc = next(s for s in global_scores if s.function == "predictive_accuracy")
        assert acc.stdev is not None
        # population std (ddof=0) of the per-fold accuracies [2/3, 1.0]
        # deviations from the 5/6 mean: -1/6 and +1/6
        assert acc.stdev == pytest.approx(1 / 6)

    def test_missing_target_column_raises(self, dataset_df, splits_df, predictions_df):
        with pytest.raises(ValueError, match="not present in the dataset"):
            evaluate_batch(
                dataset_df,
                splits_df,
                predictions_df,
                "missing_target",
                TaskType.CLASSIFICATION,
            )

    def test_missing_row_id_column_raises(self, dataset_df, splits_df):
        bad = pd.DataFrame({"repeat": [0], "fold": [0], "prediction": ["yes"]})
        with pytest.raises(PredictionValidationError, match="row_id"):
            evaluate_batch(dataset_df, splits_df, bad, "play", TaskType.CLASSIFICATION)

    def test_missing_prediction_column_raises(
        self, dataset_df, splits_df, predictions_df
    ):
        bad = predictions_df.drop(columns=["prediction"])
        with pytest.raises(PredictionValidationError, match="'prediction' column"):
            evaluate_batch(dataset_df, splits_df, bad, "play", TaskType.CLASSIFICATION)

    def test_missing_confidence_column_raises(
        self, dataset_df, splits_df, predictions_df
    ):
        bad = predictions_df.drop(columns=["confidence.yes"])
        with pytest.raises(PredictionValidationError, match="confidence.yes"):
            evaluate_batch(dataset_df, splits_df, bad, "play", TaskType.CLASSIFICATION)

    def test_prediction_count_mismatch_raises(
        self, dataset_df, splits_df, predictions_df
    ):
        with pytest.raises(
            PredictionValidationError, match="Prediction counts do not match"
        ):
            evaluate_batch(
                dataset_df,
                splits_df,
                predictions_df.iloc[:-1],  # one prediction missing
                "play",
                TaskType.CLASSIFICATION,
            )

    def test_rowid_out_of_range_raises(self, dataset_df, splits_df, predictions_df):
        bad = predictions_df.copy()
        bad.iloc[0, bad.columns.get_loc("row_id")] = 100
        with pytest.raises(PredictionValidationError, match="row_id 100"):
            evaluate_batch(dataset_df, splits_df, bad, "play", TaskType.CLASSIFICATION)

    def test_object_dtype_target_resolves_sorted_class_names(
        self, splits_df, predictions_df
    ):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "play": ["b", "a", "b", "b", "a", "a"],
            }
        )
        # class names sorted → ["a", "b"]; confidence columns must match
        preds = predictions_df.rename(
            columns={"confidence.no": "confidence.a", "confidence.yes": "confidence.b"}
        )
        preds = preds.replace({"prediction": {"no": "a", "yes": "b"}})

        _, global_scores, _ = evaluate_batch(
            df, splits_df, preds, "play", TaskType.CLASSIFICATION
        )

        by_name = {s.function: s for s in global_scores}
        assert by_name["predictive_accuracy"].value == pytest.approx(5 / 6)

    def test_leaveoneout_suppresses_per_cell_scores(
        self, dataset_df, splits_df, predictions_df
    ):
        per_cell, global_scores, _ = evaluate_batch(
            dataset_df,
            splits_df,
            predictions_df,
            "play",
            TaskType.CLASSIFICATION,
            estimation_procedure_type=EstimationProcedureType.LEAVEONEOUT,
        )

        assert per_cell == []
        assert global_scores


class TestEvaluateBatchRegression:
    @pytest.fixture
    def reg_predictions(self):
        return pd.DataFrame(
            [
                (3, 0, 0, 70.0),
                (4, 0, 0, 68.0),
                (5, 0, 0, 73.0),  # off by 1
                (0, 0, 1, 75.0),
                (1, 0, 1, 80.0),
                (2, 0, 1, 83.0),
            ],
            columns=["row_id", "repeat", "fold", "prediction"],
        )

    def test_global_mae_golden_value(self, dataset_df, splits_df, reg_predictions):
        _, global_scores, _ = evaluate_batch(
            dataset_df, splits_df, reg_predictions, "temp", TaskType.REGRESSION
        )

        by_name = {s.function: s for s in global_scores}
        assert by_name["mean_absolute_error"].value == pytest.approx(1 / 6)
        assert by_name["number_of_instances"].value == 6.0

    def test_regression_needs_no_confidence_columns(
        self, dataset_df, splits_df, reg_predictions
    ):
        per_cell, _, _ = evaluate_batch(
            dataset_df, splits_df, reg_predictions, "temp", TaskType.REGRESSION
        )

        assert any(s.function == "root_mean_squared_error" for s in per_cell)


class TestEvaluateBatchLearningCurve:
    @pytest.fixture
    def lc_splits(self):
        return pd.DataFrame(
            [
                ("TRAIN", 0, 0, 0, 0),
                ("TEST", 2, 0, 0, 0),
                ("TRAIN", 0, 0, 0, 1),
                ("TRAIN", 1, 0, 0, 1),
                ("TEST", 2, 0, 0, 1),
            ],
            columns=["type", "rowid", "repeat", "fold", "sample"],
        )

    @pytest.fixture
    def lc_predictions(self):
        # row 2 has true label "yes"
        return pd.DataFrame(
            [
                (2, 0, 0, 0, "yes", 0.1, 0.9),
                (2, 0, 0, 1, "no", 0.6, 0.4),
            ],
            columns=[
                "row_id",
                "repeat",
                "fold",
                "sample",
                "prediction",
                "confidence.no",
                "confidence.yes",
            ],
        )

    def test_sample_column_is_required(self, dataset_df, lc_splits, lc_predictions):
        bad = lc_predictions.drop(columns=["sample"])
        with pytest.raises(ValueError, match="None of the expected columns"):
            evaluate_batch(dataset_df, lc_splits, bad, "play", TaskType.LEARNINGCURVE)

    def test_per_cell_scores_carry_sample_and_size(
        self, dataset_df, lc_splits, lc_predictions
    ):
        per_cell, global_scores, _ = evaluate_batch(
            dataset_df, lc_splits, lc_predictions, "play", TaskType.LEARNINGCURVE
        )

        samples = {s.sample for s in per_cell}
        assert samples == {0, 1}
        sample_sizes = {
            (s.sample, s.sample_size)
            for s in per_cell
            if s.function == "predictive_accuracy"
        }
        # sample 0 trains on 1 row, sample 1 on 2 rows
        assert sample_sizes == {(0, 1), (1, 2)}
        # only the last sample's predictions contribute globally
        acc = next(s for s in global_scores if s.function == "predictive_accuracy")
        assert acc.value == pytest.approx(0.0)  # last sample predicted "no" (wrong)


class TestEvaluateStream:
    @pytest.fixture
    def stream_dataset(self):
        return pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "class": pd.Categorical(["a", "b", "a", "b"], categories=["a", "b"]),
            }
        )

    @pytest.fixture
    def stream_predictions(self):
        return pd.DataFrame(
            [
                (0, "a", 0.8, 0.2),
                (1, "b", 0.1, 0.9),
                (2, "a", 0.9, 0.1),
                (3, "a", 0.4, 0.6),  # wrong
            ],
            columns=["row_id", "prediction", "confidence.a", "confidence.b"],
        )

    def test_scores_computed(self, stream_dataset, stream_predictions):
        scores = evaluate_stream(stream_dataset, stream_predictions, "class")

        by_name = {s.function: s for s in scores}
        assert by_name["predictive_accuracy"].value == pytest.approx(0.75)
        assert by_name["number_of_instances"].value == 4.0

    def test_length_mismatch_raises(self, stream_dataset, stream_predictions):
        with pytest.raises(PredictionValidationError, match="same order"):
            evaluate_stream(stream_dataset, stream_predictions.iloc[:-1], "class")

    def test_order_mismatch_raises(self, stream_dataset, stream_predictions):
        shuffled = stream_predictions.iloc[[1, 0, 2, 3]].reset_index(drop=True)
        with pytest.raises(PredictionValidationError, match="same order"):
            evaluate_stream(stream_dataset, shuffled, "class")

    def test_numeric_target_raises(self, stream_dataset, stream_predictions):
        df = stream_dataset.assign(x2=[1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="no class labels"):
            evaluate_stream(df, stream_predictions, "x2")

    def test_missing_confidence_column_raises(self, stream_dataset, stream_predictions):
        bad = stream_predictions.drop(columns=["confidence.b"])
        with pytest.raises(PredictionValidationError, match="confidence.b"):
            evaluate_stream(stream_dataset, bad, "class")


class TestEvaluateSurvival:
    def test_matching_predictions_pass(self, dataset_df, splits_df, predictions_df):
        scores, pc = evaluate_survival(dataset_df, splits_df, predictions_df, "play")

        assert scores == []
        assert pc.check() is True

    def test_count_mismatch_raises(self, dataset_df, splits_df, predictions_df):
        with pytest.raises(PredictionValidationError, match="counts do not match"):
            evaluate_survival(dataset_df, splits_df, predictions_df.iloc[:-2], "play")

    def test_rowid_out_of_range_raises(self, dataset_df, splits_df, predictions_df):
        bad = predictions_df.copy()
        bad.iloc[0, bad.columns.get_loc("row_id")] = 99
        with pytest.raises(PredictionValidationError, match="row_id 99"):
            evaluate_survival(dataset_df, splits_df, bad, "play")


class TestTaskTypeMapping:
    def test_survival_maps_to_none(self):
        assert TASK_TYPE_ID_TO_TASK_TYPE[7] is None

    @pytest.mark.parametrize(
        "task_type_id, expected",
        [
            (1, TaskType.CLASSIFICATION),
            (2, TaskType.REGRESSION),
            (3, TaskType.LEARNINGCURVE),
            (4, TaskType.TESTTHENTRAIN),
            (5, TaskType.CLASSIFICATION),
            (6, TaskType.CLASSIFICATION),
            (8, TaskType.CLASSIFICATION),
        ],
    )
    def test_known_mappings(self, task_type_id, expected):
        assert TASK_TYPE_ID_TO_TASK_TYPE[task_type_id] == expected


class TestEvaluateRun:
    def test_unsupported_task_type_sets_error(self, dataset_df, splits_df):
        result = evaluate_run(99, dataset_df, splits_df, predictions_df_head(), "play")

        assert result.error == "Task type not supported: 99"
        assert result.scores == []

    def test_empty_predictions_sets_error(self, dataset_df, splits_df):
        empty = pd.DataFrame()
        result = evaluate_run(1, dataset_df, splits_df, empty, "play")

        assert result.error == (
            "Required output files not present (e.g., arff predictions)."
        )

    def test_classification_dispatch(self, dataset_df, splits_df, predictions_df):
        result = evaluate_run(1, dataset_df, splits_df, predictions_df, "play")

        assert result.error is None
        by_name = {s.function: s for s in result.scores}
        assert by_name["predictive_accuracy"].value == pytest.approx(5 / 6)

    def test_survival_without_splits_sets_error(self, dataset_df, predictions_df):
        result = evaluate_run(7, dataset_df, None, predictions_df, "play")

        assert result.error == "Splits required for survival analysis tasks."

    def test_batch_without_splits_sets_error(self, dataset_df, predictions_df):
        result = evaluate_run(1, dataset_df, None, predictions_df, "play")

        assert result.error == "Splits required for batch evaluation tasks."

    def test_internal_exception_captured_as_error(
        self, dataset_df, splits_df, predictions_df
    ):
        # Missing confidence column → PredictionValidationError captured
        bad = predictions_df.drop(columns=["confidence.yes"])
        result = evaluate_run(1, dataset_df, splits_df, bad, "play")

        assert result.error is not None
        assert "confidence.yes" in result.error

    def test_stream_dispatch(self, dataset_df, predictions_df):
        stream_preds = pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3, 4, 5],
                "prediction": ["yes", "no", "yes", "yes", "no", "no"],
                "confidence.no": [0.1] * 6,
                "confidence.yes": [0.9] * 6,
            }
        )
        result = evaluate_run(4, dataset_df, None, stream_preds, "play")

        assert result.error is None
        assert any(s.function == "predictive_accuracy" for s in result.scores)


def predictions_df_head():
    return pd.DataFrame(
        {
            "row_id": [0],
            "repeat": [0],
            "fold": [0],
            "prediction": ["yes"],
            "confidence.no": [0.1],
            "confidence.yes": [0.9],
        }
    )
