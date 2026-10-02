import math

import pytest

from src.features.module import _fill_nominal_feature, _fill_numeric_feature
from src.models import Feature


@pytest.fixture
def feat() -> Feature:
    """A fresh, unpopulated numeric Feature for each test."""
    return Feature(index=0, name="col", data_type="numeric")


class TestEmptyAndMissing:
    def test_empty_column(self, feat):
        _fill_numeric_feature([], feat)

        assert feat.number_of_values == 0
        assert feat.number_of_missing_values == 0
        # Early return: nothing else should have been touched.
        assert feat.number_of_integer_values is None
        assert feat.number_of_real_values is None
        assert feat.maximum_value is None
        assert feat.minimum_value is None
        assert feat.mean_value is None
        assert feat.standard_deviation is None
        assert feat.number_of_distinct_values is None
        assert feat.number_of_unique_values is None

    def test_all_missing_values(self, feat):
        _fill_numeric_feature([None, None, None], feat)

        assert feat.number_of_values == 3
        assert feat.number_of_missing_values == 3
        # present.size == 0 -> early return, stats left untouched
        assert feat.number_of_integer_values is None
        assert feat.mean_value is None

    def test_partial_missing_values(self, feat):
        _fill_numeric_feature([1, None, 2, None, 3], feat)

        assert feat.number_of_values == 5
        assert feat.number_of_missing_values == 2
        assert feat.mean_value == pytest.approx(2.0)


class TestIntegerAndRealCounts:
    def test_all_integers(self, feat):
        _fill_numeric_feature([1, 2, 3, 4], feat)

        assert feat.number_of_integer_values == 4
        assert feat.number_of_real_values == 0

    def test_all_reals(self, feat):
        _fill_numeric_feature([1.5, 2.25, 3.75], feat)

        assert feat.number_of_integer_values == 0
        assert feat.number_of_real_values == 3

    def test_mixed_integers_and_reals(self, feat):
        # 2.0 counts as "integer" because it equals floor(2.0)
        _fill_numeric_feature([1, 2.0, 3.5, 4], feat)

        assert feat.number_of_integer_values == 3
        assert feat.number_of_real_values == 1


class TestSummaryStatistics:
    def test_min_max_mean_std(self, feat):
        data = [1, 2, 3, 4, 5]
        _fill_numeric_feature(data, feat)

        assert feat.minimum_value == 1.0
        assert feat.maximum_value == 5.0
        assert feat.mean_value == pytest.approx(3.0)
        assert feat.standard_deviation == pytest.approx(math.sqrt(2.0))

    def test_single_value(self, feat):
        _fill_numeric_feature([42], feat)

        assert feat.minimum_value == 42.0
        assert feat.maximum_value == 42.0
        assert feat.mean_value == pytest.approx(42.0)
        assert feat.standard_deviation == pytest.approx(0.0)

    def test_negative_and_float_values(self, feat):
        _fill_numeric_feature([-1.5, 0, 1.5], feat)

        assert feat.minimum_value == -1.5
        assert feat.maximum_value == 1.5
        assert feat.mean_value == pytest.approx(0.0)


class TestDistinctAndUniqueValues:
    def test_all_distinct(self, feat):
        _fill_numeric_feature([1, 2, 3, 4], feat)

        assert feat.number_of_distinct_values == 4
        assert feat.number_of_unique_values == 4

    def test_with_duplicates(self, feat):
        # 1 appears twice, 2 appears once, 3 appears once
        _fill_numeric_feature([1, 1, 2, 3], feat)

        assert feat.number_of_distinct_values == 3
        assert feat.number_of_unique_values == 2  # only 2 and 3 appear exactly once

    def test_no_unique_values(self, feat):
        _fill_numeric_feature([1, 1, 2, 2], feat)

        assert feat.number_of_distinct_values == 2
        assert feat.number_of_unique_values == 0


class TestFullIntegration:
    def test_realistic_mixed_column(self, feat):
        data = [25, 30, None, 25, 40.5, None, 30, 22]
        _fill_numeric_feature(data, feat)

        assert feat.number_of_values == 8
        assert feat.number_of_missing_values == 2
        # present: [25, 30, 25, 40.5, 30, 22] -> 6 values
        assert feat.number_of_integer_values == 5  # 25, 30, 25, 30, 22
        assert feat.number_of_real_values == 1  # 40.5
        assert feat.minimum_value == 22.0
        assert feat.maximum_value == 40.5
        assert feat.mean_value == pytest.approx(sum([25, 30, 25, 40.5, 30, 22]) / 6)
        # distinct present values: 25, 30, 40.5, 22 -> 4 distinct
        assert feat.number_of_distinct_values == 4
        # unique (appear exactly once): 40.5, 22 -> 2
        assert feat.number_of_unique_values == 2

    def test_does_not_mutate_input_list(self, feat):
        data = [1, 2, None, 3]
        original = list(data)
        _fill_numeric_feature(data, feat)

        assert data == original


@pytest.fixture
def nominal_feat() -> Feature:
    """A fresh, unpopulated nominal Feature for each test."""
    return Feature(index=0, name="col", data_type="nominal")


class TestFillNominalFeature:
    def test_counts_present_values(self, nominal_feat):
        _fill_nominal_feature(["a", "b", "a", None, "b"], nominal_feat, ["b", "a"])

        assert nominal_feat.number_of_values == 5
        assert nominal_feat.number_of_missing_values == 1
        assert nominal_feat.number_of_nominal_values == 4
        assert nominal_feat.number_of_distinct_values == 2
        assert nominal_feat.number_of_unique_values == 0

    def test_unique_values_counted_once(self, nominal_feat):
        _fill_nominal_feature(["a", "b", "b", "c", None], nominal_feat, ["a", "b", "c"])

        assert nominal_feat.number_of_distinct_values == 3
        assert nominal_feat.number_of_unique_values == 2  # 'a' and 'c'

    def test_all_missing(self, nominal_feat):
        _fill_nominal_feature([None, None], nominal_feat, ["a", "b"])

        assert nominal_feat.number_of_missing_values == 2
        assert nominal_feat.number_of_nominal_values == 0
        assert nominal_feat.number_of_distinct_values == 0
        assert nominal_feat.class_distribution == ""

    def test_empty_column(self, nominal_feat):
        _fill_nominal_feature([], nominal_feat, ["a"])

        assert nominal_feat.number_of_values == 0
        assert nominal_feat.nominal_values == ["a"]

    def test_nominal_values_sorted_from_schema(self, nominal_feat):
        _fill_nominal_feature(["b", "a"], nominal_feat, ["b", "a"])

        assert nominal_feat.nominal_values == ["a", "b"]

    def test_class_distribution_sorted_string(self, nominal_feat):
        _fill_nominal_feature(
            ["yes", "no", "yes", "no", "yes"], nominal_feat, ["no", "yes"]
        )

        assert nominal_feat.class_distribution == "no:2,yes:3"

    def test_input_not_mutated(self, nominal_feat):
        data = ["a", None]
        original = list(data)
        _fill_nominal_feature(data, nominal_feat, ["a"])

        assert data == original
