import numpy as np
import pytest

from src.qualities.module import build_xy, compute_dataset_qualities

ATTRIBUTES = [
    ("outlook", ["sunny", "overcast", "rainy"]),
    ("temp", "NUMERIC"),
    ("play", ["yes", "no"]),
]

ROWS = [
    ("sunny", 75, "yes"),
    ("sunny", 80, "no"),
    ("overcast", None, "yes"),
    ("rainy", 68, None),
    ("rainy", 72, "no"),
]


def qualities_by_name(qualities):
    return {q.name: q.value for q in qualities}


class TestComputeDatasetQualities:
    def test_counts_and_dimensions(self):
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, {"play"}))

        assert q["NumberOfInstances"] == 5.0
        assert q["NumberOfFeatures"] == 3.0
        assert q["NumberOfClasses"] == 2.0
        assert q["Dimensionality"] == pytest.approx(0.6)

    def test_missing_value_qualities(self):
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, {"play"}))

        assert q["NumberOfInstancesWithMissingValues"] == 2.0
        assert q["NumberOfMissingValues"] == 2.0
        assert q["PercentageOfInstancesWithMissingValues"] == pytest.approx(40.0)
        assert q["PercentageOfMissingValues"] == pytest.approx(2 / 15 * 100)

    def test_feature_type_qualities(self):
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, {"play"}))

        assert q["NumberOfNumericFeatures"] == 1.0
        assert q["NumberOfSymbolicFeatures"] == 2.0
        # 'play' is binary; 'outlook' has 3 categories; temp has 4 distinct
        assert q["NumberOfBinaryFeatures"] == 1.0
        assert q["PercentageOfNumericFeatures"] == pytest.approx(100 / 3)
        assert q["PercentageOfSymbolicFeatures"] == pytest.approx(200 / 3)

    def test_binary_numeric_feature_counted(self):
        attributes = [("flag", "NUMERIC"), ("y", ["a", "b"])]
        rows = [(0, "a"), (1, "b"), (0, "a"), (1, "b")]

        q = qualities_by_name(compute_dataset_qualities(attributes, rows, {"y"}))

        assert q["NumberOfBinaryFeatures"] == 2.0

    def test_class_distribution(self):
        # play values: yes, no, yes, None, no → 2 vs 2
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, {"play"}))

        assert q["MajorityClassSize"] == 2.0
        assert q["MinorityClassSize"] == 2.0
        assert q["MajorityClassPercentage"] == pytest.approx(50.0)
        assert q["MinorityClassPercentage"] == pytest.approx(50.0)

    def test_missing_target_value_undefined_autocorrelation(self):
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, {"play"}))

        assert q["AutoCorrelation"] is None

    def test_no_target_gives_class_none_qualities(self):
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, set()))

        assert q["NumberOfClasses"] is None
        assert q["MajorityClassSize"] is None
        assert q["MinorityClassSize"] is None
        assert q["AutoCorrelation"] is None

    def test_numeric_target_gives_class_none_qualities(self):
        q = qualities_by_name(compute_dataset_qualities(ATTRIBUTES, ROWS, {"temp"}))

        assert q["NumberOfClasses"] is None
        assert q["MajorityClassPercentage"] is None


class TestAutoCorrelation:
    def _autocorr(self, labels, numeric=False):
        if numeric:
            attributes = [("v", "NUMERIC")]
            rows = [(v,) for v in labels]
            target = {"v"}
        else:
            attributes = [("v", ["a", "b", "c"])]
            rows = [(v,) for v in labels]
            target = {"v"}
        q = qualities_by_name(compute_dataset_qualities(attributes, rows, target))
        return q["AutoCorrelation"]

    def test_nominal_golden_value(self):
        # changes: a→a (0), a→b (1), b→b (0) → (4-1-1)/(4-1)
        assert self._autocorr(["a", "a", "b", "b"]) == pytest.approx(2 / 3)

    def test_constant_labels_score_one(self):
        assert self._autocorr(["a", "a", "a", "a"]) == pytest.approx(1.0)

    def test_alternating_labels_score_zero(self):
        assert self._autocorr(["a", "b", "a", "b"]) == pytest.approx(0.0)

    def test_single_instance_returns_one(self):
        assert self._autocorr(["a"]) == 1.0

    def test_numeric_golden_value(self):
        # changes = |1-3| + |3-2| = 3 → (3-1-3)/(3-1)
        assert self._autocorr([1, 3, 2], numeric=True) == pytest.approx(-0.5)

    def test_numeric_constant_returns_one(self):
        assert self._autocorr([2.0, 2.0, 2.0], numeric=True) == pytest.approx(1.0)

    def test_missing_value_mid_series_is_none(self):
        assert self._autocorr(["a", None, "b"]) is None


class TestBuildXy:
    def test_string_target_is_categorically_encoded(self):
        attributes = [
            ("f", "NUMERIC"),
            ("target", ["x", "y"]),
        ]
        rows = [(1.0, "y"), (2.0, "x"), (3.0, "y")]

        X, y = build_xy(attributes, rows, {"target"})

        np.testing.assert_allclose(X[:, 0], [1.0, 2.0, 3.0])
        # categorical codes use sorted categories: x -> 0, y -> 1
        np.testing.assert_allclose(y, [1.0, 0.0, 1.0])

    def test_numeric_target_stays_float(self):
        attributes = [("f", "NUMERIC"), ("target", "NUMERIC")]
        rows = [(1.0, 10.0), (2.0, 20.0)]

        _, y = build_xy(attributes, rows, {"target"})

        np.testing.assert_allclose(y, [10.0, 20.0])

    def test_object_feature_columns_are_encoded(self):
        attributes = [("cat", ["a", "b"]), ("target", ["x", "y"])]
        rows = [("a", "x"), ("b", "y")]

        X, y = build_xy(attributes, rows, {"target"})

        np.testing.assert_allclose(X[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(y, [0.0, 1.0])

    def test_multi_word_string_columns_are_dropped(self):
        attributes = [
            ("f", "NUMERIC"),
            ("words", "STRING"),
            ("target", ["x", "y"]),
        ]
        rows = [
            (1.0, "hello world", "x"),
            (2.0, "single", "y"),
        ]

        X, _ = build_xy(attributes, rows, {"target"})

        assert X.shape[1] == 1

    def test_target_column_excluded_from_features(self):
        attributes = [("f", "NUMERIC"), ("target", ["x", "y"])]
        rows = [(1.0, "x"), (2.0, "y")]

        X, _ = build_xy(attributes, rows, {"target"})

        assert X.shape == (2, 1)

    def test_missing_target_raises(self):
        attributes = [("f", "NUMERIC")]
        rows = [(1.0,)]

        with pytest.raises(ValueError, match="Target feature not found"):
            build_xy(attributes, rows, {"target"})
