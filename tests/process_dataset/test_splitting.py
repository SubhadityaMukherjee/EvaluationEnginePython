import numpy as np
import pandas as pd
import pytest

from src.models import EstimationProcedure, EstimationProcedureType
from src.process_dataset.splitting import (
    _is_nominal,
    crossvalidation_splits,
    holdout_ordered_splits,
    holdout_splits,
    learning_curve_splits,
    leave_one_out_splits,
    num_samples,
    sample_size,
    train_on_test_splits,
)


class TestSampleSize:
    @pytest.mark.parametrize(
        "number, train_size, expected",
        [
            (0, 1000, 64),  # 2 ** 6
            (1, 1000, 91),  # round(2 ** 6.5)
            (2, 1000, 128),  # 2 ** 7
            (4, 1000, 256),  # 2 ** 8
            (10, 100, 100),  # capped at train_size
            (10, 4096, 2048),  # 2 ** (6 + 5)
        ],
    )
    def test_golden_values(self, number, train_size, expected):
        assert sample_size(number, train_size) == expected


class TestNumSamples:
    def test_small_training_set_is_one_sample(self):
        # 64 == train_size already at i=0
        assert num_samples(64) == 1

    def test_geometric_growth_until_capped(self):
        # 64, 91, 128, ... 100 caps at i=2
        assert num_samples(100) == 3

    def test_exact_power_of_two(self):
        # 64 < 128 at i=0, round(2**6.5)=91 < 128 at i=1, 128 at i=2
        assert num_samples(128) == 3


class TestIsNominal:
    def test_categorical_is_nominal(self):
        df = pd.DataFrame({"y": pd.Categorical(["a", "b"])})
        assert _is_nominal(df, "y") is True

    def test_object_dtype_is_nominal(self):
        # explicit object dtype (pandas 3 defaults strings to `str` dtype)
        df = pd.DataFrame({"y": pd.Series(["a", "b"], dtype=object)})
        assert _is_nominal(df, "y") is True

    def test_numeric_is_not_nominal(self):
        df = pd.DataFrame({"y": [1, 2]})
        assert _is_nominal(df, "y") is False

    def test_missing_column_is_not_nominal(self):
        df = pd.DataFrame({"y": [1, 2]})
        assert _is_nominal(df, "z") is False

    def test_none_target_is_not_nominal(self):
        df = pd.DataFrame({"y": ["a"]})
        assert _is_nominal(df, None) is False


def cv_df(n=12):
    return pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "y": pd.Categorical((["a", "b"] * (n // 2 + 1))[:n], categories=["a", "b"]),
        }
    )


class TestCrossvalidationSplits:
    @pytest.fixture
    def procedure(self):
        return EstimationProcedure(
            type=EstimationProcedureType.CROSSVALIDATION, folds=3, repeats=2
        )

    def test_row_counts(self, procedure):
        splits = crossvalidation_splits(cv_df(), procedure, target="y", seed=1)

        # per repeat: 3 folds x 12 assignment rows = 36; 2 repeats
        assert len(splits) == 72
        assert splits["repeat"].nunique() == 2
        assert set(splits["fold"].unique()) == {0, 1, 2}

    def test_train_test_partition_per_fold(self, procedure):
        df = cv_df()
        splits = crossvalidation_splits(df, procedure, target="y", seed=1)

        for (repeat, fold), group in splits.groupby(["repeat", "fold"]):
            train = set(group[group["type"] == "TRAIN"]["rowid"])
            test = set(group[group["type"] == "TEST"]["rowid"])
            assert train | test == set(range(12))
            assert not train & test

    def test_each_instance_tested_once_per_repeat(self, procedure):
        splits = crossvalidation_splits(cv_df(), procedure, target="y", seed=1)

        for repeat, group in splits[splits["type"] == "TEST"].groupby("repeat"):
            assert sorted(group["rowid"]) == list(range(12))

    def test_stratified_folds_preserve_class_ratio(self, procedure):
        # 6 a / 6 b → each fold's TEST side must have 2 a and 2 b
        df = cv_df()
        splits = crossvalidation_splits(df, procedure, target="y", seed=1)

        test_rows = splits[splits["type"] == "TEST"]
        for _, group in test_rows.groupby(["repeat", "fold"]):
            labels = df["y"].iloc[group["rowid"]].tolist()
            assert labels.count("a") == 2
            assert labels.count("b") == 2

    def test_deterministic_given_seed(self, procedure):
        df = cv_df()
        s1 = crossvalidation_splits(df, procedure, target="y", seed=1)
        s2 = crossvalidation_splits(df, procedure, target="y", seed=1)

        pd.testing.assert_frame_equal(s1, s2)

    def test_repeats_use_distinct_permutations(self, procedure):
        splits = crossvalidation_splits(cv_df(), procedure, target="y", seed=1)

        r0 = set(
            splits[(splits["repeat"] == 0) & (splits["type"] == "TEST")][
                ["rowid", "fold"]
            ].apply(tuple, axis=1)
        )
        r1 = set(
            splits[(splits["repeat"] == 1) & (splits["type"] == "TEST")][
                ["rowid", "fold"]
            ].apply(tuple, axis=1)
        )
        assert r0 != r1

    def test_unstratified_when_target_missing(self, procedure):
        splits = crossvalidation_splits(cv_df(), procedure, target=None, seed=1)

        test_counts = (
            splits[splits["type"] == "TEST"].groupby(["repeat", "fold"]).size()
        )
        assert (test_counts == 4).all()


class TestLeaveOneOutSplits:
    def test_n_squared_rows(self):
        splits = leave_one_out_splits(cv_df(4))

        assert len(splits) == 16
        assert splits["repeat"].eq(0).all()
        assert set(splits["fold"].unique()) == {0, 1, 2, 3}

    def test_fold_f_tests_exactly_instance_f(self):
        splits = leave_one_out_splits(cv_df(4))

        for fold in range(4):
            group = splits[splits["fold"] == fold]
            test = group[group["type"] == "TEST"]["rowid"].tolist()
            train = group[group["type"] == "TRAIN"]["rowid"].tolist()
            assert test == [fold]
            assert sorted(train) == [i for i in range(4) if i != fold]


class TestTrainOnTestSplits:
    def test_every_instance_twice(self):
        splits = train_on_test_splits(cv_df(3))

        assert len(splits) == 6
        assert sorted(splits[splits["type"] == "TEST"]["rowid"]) == [0, 1, 2]
        assert sorted(splits[splits["type"] == "TRAIN"]["rowid"]) == [0, 1, 2]
        assert splits["fold"].eq(0).all()
        assert splits["repeat"].eq(0).all()


class TestHoldoutSplits:
    @pytest.fixture
    def procedure(self):
        return EstimationProcedure(
            type=EstimationProcedureType.HOLDOUT, repeats=2, percentage=30.0
        )

    def test_test_size_rounded(self, procedure):
        df = cv_df(10)
        splits = holdout_splits(df, procedure, seed=1)

        for repeat, group in splits.groupby("repeat"):
            test = group[group["type"] == "TEST"]
            train = group[group["type"] == "TRAIN"]
            assert len(test) == 3
            assert len(train) == 7
            assert set(test["rowid"]) | set(train["rowid"]) == set(range(10))

    def test_repeat_count(self, procedure):
        splits = holdout_splits(cv_df(10), procedure, seed=1)

        assert splits["repeat"].nunique() == 2
        assert splits["fold"].eq(0).all()


class TestHoldoutOrderedSplits:
    def test_first_rows_train_tail_test(self):
        procedure = EstimationProcedure(
            type=EstimationProcedureType.HOLDOUT_ORDERED, percentage=33.0
        )
        df = cv_df(10)

        splits = holdout_ordered_splits(df, procedure)

        test = splits[splits["type"] == "TEST"]["rowid"].tolist()
        train = splits[splits["type"] == "TRAIN"]["rowid"].tolist()
        # test_size = 10 * 33 / 100 = 3.3 → threshold = 6.7 → 0..6 TRAIN
        assert train == [0, 1, 2, 3, 4, 5, 6]
        assert test == [7, 8, 9]
        assert splits["repeat"].eq(0).all()
        assert splits["fold"].eq(0).all()

    def test_file_order_is_preserved(self):
        procedure = EstimationProcedure(
            type=EstimationProcedureType.HOLDOUT_ORDERED, percentage=50.0
        )
        splits = holdout_ordered_splits(cv_df(4), procedure)

        assert splits["rowid"].tolist() == [0, 1, 2, 3]


class TestLearningCurveSplits:
    @pytest.fixture
    def procedure(self):
        return EstimationProcedure(
            type=EstimationProcedureType.LEARNINGCURVE_CV, folds=2, repeats=1
        )

    @pytest.fixture
    def df(self):
        # n=150 → 2 folds → train size 75 → samples at 64 then 75 (cap)
        n = 150
        return pd.DataFrame(
            {
                "x": np.arange(n, dtype=float),
                "y": pd.Categorical(["a", "b"] * (n // 2), categories=["a", "b"]),
            }
        )

    def test_emits_sample_column(self, procedure, df):
        splits = learning_curve_splits(df, procedure, target="y", seed=1)

        assert "sample" in splits.columns
        assert set(splits["sample"].unique()) == {0, 1}

    def test_geometric_sample_sizes(self, procedure, df):
        splits = learning_curve_splits(df, procedure, target="y", seed=1)

        for (repeat, fold), group in splits.groupby(["repeat", "fold"]):
            train_sizes = {
                sample: len(g[(g["type"] == "TRAIN")])
                for sample, g in group.groupby("sample")
            }
            assert train_sizes == {0: 64, 1: 75}

    def test_test_side_repeated_for_every_sample(self, procedure, df):
        splits = learning_curve_splits(df, procedure, target="y", seed=1)

        for (repeat, fold), group in splits.groupby(["repeat", "fold"]):
            test_sets = [
                tuple(sorted(g[g["type"] == "TEST"]["rowid"]))
                for sample, g in group.groupby("sample")
            ]
            assert test_sets[0] == test_sets[1]

    def test_smaller_sample_is_subset_of_larger(self, procedure, df):
        splits = learning_curve_splits(df, procedure, target="y", seed=1)

        for (repeat, fold), group in splits.groupby(["repeat", "fold"]):
            s0 = set(
                group[(group["type"] == "TRAIN") & (group["sample"] == 0)]["rowid"]
            )
            s1 = set(
                group[(group["type"] == "TRAIN") & (group["sample"] == 1)]["rowid"]
            )
            assert s0 <= s1
