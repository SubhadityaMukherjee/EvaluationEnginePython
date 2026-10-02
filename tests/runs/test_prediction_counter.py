import pandas as pd
import pytest

from src.exceptions import PredictionValidationError
from src.runs.prediction_counter import FoldsPredictionCounter


@pytest.fixture
def splits():
    return pd.DataFrame(
        [
            ("TRAIN", 0, 0, 0),
            ("TRAIN", 1, 0, 0),
            ("TEST", 2, 0, 0),
            ("TRAIN", 2, 0, 1),
            ("TEST", 1, 0, 1),
            ("TEST", 0, 0, 1),
            ("TRAIN", 0, 1, 0),
            ("TEST", 1, 1, 0),
        ],
        columns=["type", "rowid", "repeat", "fold"],
    )


class TestConstruction:
    def test_dimensions(self, splits):
        pc = FoldsPredictionCounter(splits)

        assert pc.repeats == 2
        assert pc.folds == 2
        assert pc.samples == 1

    def test_expected_rowids_sorted_per_cell(self, splits):
        pc = FoldsPredictionCounter(splits)

        assert pc.get_expected_rowids(0, 0, 0) == [2]
        assert pc.get_expected_rowids(0, 1, 0) == [0, 1]
        assert pc.get_expected_rowids(1, 0, 0) == [1]
        assert pc.get_expected_rowids(1, 1, 0) == []

    def test_shadow_type_size_counts_train_rows(self, splits):
        pc = FoldsPredictionCounter(splits)

        assert pc.get_shadow_type_size(0, 0, 0) == 2
        assert pc.get_shadow_type_size(0, 1, 0) == 1
        assert pc.get_shadow_type_size(1, 0, 0) == 1
        assert pc.get_shadow_type_size(1, 1, 0) == 0

    def test_expected_total_counts_test_rows(self, splits):
        pc = FoldsPredictionCounter(splits)

        assert pc.get_expected_total() == 4

    def test_column_aliases(self):
        splits = pd.DataFrame(
            [
                ("TEST", 0, 0, 0, 0),
                ("TEST", 1, 0, 0, 1),
                ("TRAIN", 2, 0, 0, 0),
            ],
            columns=["type", "row_id", "repeat_nr", "fold_nr", "sample_nr"],
        )

        pc = FoldsPredictionCounter(splits)

        assert pc.samples == 2
        assert pc.get_expected_rowids(0, 0, 0) == [0]
        assert pc.get_expected_rowids(0, 0, 1) == [1]

    def test_missing_sample_column_defaults_to_one(self, splits):
        pc = FoldsPredictionCounter(splits)

        assert pc.samples == 1


class TestAddPrediction:
    def test_out_of_range_repeat_raises(self, splits):
        pc = FoldsPredictionCounter(splits)

        with pytest.raises(PredictionValidationError, match="repeat 5"):
            pc.add_prediction(5, 0, 0, 0)

    def test_out_of_range_fold_raises(self, splits):
        pc = FoldsPredictionCounter(splits)

        with pytest.raises(PredictionValidationError, match="fold 3"):
            pc.add_prediction(0, 3, 0, 0)


class TestCheck:
    def test_matching_predictions_pass(self, splits):
        pc = FoldsPredictionCounter(splits)
        pc.add_prediction(0, 0, 0, 2)
        pc.add_prediction(0, 1, 0, 1)
        pc.add_prediction(0, 1, 0, 0)
        pc.add_prediction(1, 0, 0, 1)

        assert pc.check() is True
        assert pc.get_error_message() == ""

    def test_missing_prediction_fails_with_message(self, splits):
        pc = FoldsPredictionCounter(splits)
        pc.add_prediction(0, 0, 0, 2)
        # fold (0, 1) predictions missing entirely

        assert pc.check() is False
        assert "Repeat 0 fold 1 sample 0" in pc.get_error_message()
        assert "[0, 1]" in pc.get_error_message()

    def test_extra_prediction_fails(self, splits):
        pc = FoldsPredictionCounter(splits)
        pc.add_prediction(0, 0, 0, 2)
        pc.add_prediction(0, 0, 0, 7)  # wrong row id in same cell

        assert pc.check() is False

    def test_unsorted_additions_still_pass_after_sort(self, splits):
        pc = FoldsPredictionCounter(splits)
        pc.add_prediction(0, 1, 0, 0)
        pc.add_prediction(0, 1, 0, 1)  # reverse order on purpose

        pc.add_prediction(0, 0, 0, 2)
        pc.add_prediction(1, 0, 0, 1)

        assert pc.check() is True
