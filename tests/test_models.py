import pytest

from src.models import (
    DataFeature,
    DataQuality,
    EstimationProcedure,
    EstimationProcedureType,
    EvaluationScore,
    Feature,
    Quality,
    RunEvaluation,
)


class TestEstimationProcedureTypeFromOmlType:
    @pytest.mark.parametrize(
        "type_str, expected",
        [
            ("crossvalidation", EstimationProcedureType.CROSSVALIDATION),
            ("CROSSVALIDATION", EstimationProcedureType.CROSSVALIDATION),
            ("holdout", EstimationProcedureType.HOLDOUT),
            ("holdout_ordered", EstimationProcedureType.HOLDOUT_ORDERED),
            ("leaveoneout", EstimationProcedureType.LEAVEONEOUT),
            ("testontrainingdata", EstimationProcedureType.TESTONTRAININGDATA),
            # verbose alias maps to the same member
            ("testonthetrainingdata", EstimationProcedureType.TESTONTRAININGDATA),
            ("learningcurve", EstimationProcedureType.LEARNINGCURVE_CV),
            ("", None),
            (None, None),
            ("bogus_procedure", None),
        ],
    )
    def test_mapping(self, type_str, expected):
        assert EstimationProcedureType.from_oml_type(type_str) == expected

    def test_all_members_reachable(self):
        members = set(EstimationProcedureType)
        mapped = {
            EstimationProcedureType.from_oml_type(t)
            for t in (
                "crossvalidation",
                "holdout",
                "holdout_ordered",
                "leaveoneout",
                "testontrainingdata",
                "learningcurve",
            )
        }
        assert mapped == members


class TestQuality:
    def test_str_includes_name_and_value(self):
        assert str(Quality(name="NumberOfInstances", value=14.0)) == (
            "NumberOfInstances - 14.0"
        )

    def test_default_value_is_none(self):
        assert Quality(name="x").value is None


class TestFeatureStr:
    def test_str(self):
        assert str(Feature(index=3, name="age", data_type="numeric")) == "3 - age"


class TestDataFeatureFeatureMap:
    def test_maps_by_name(self):
        f0 = Feature(index=0, name="b", data_type="numeric")
        f1 = Feature(index=1, name="a", data_type="nominal")
        df = DataFeature(features=[f0, f1])

        result = df.feature_map()

        assert result == {"b": f0, "a": f1}

    def test_sorted_names(self):
        f0 = Feature(index=0, name="b", data_type="numeric")
        f1 = Feature(index=1, name="a", data_type="nominal")
        df = DataFeature(features=[f0, f1])

        assert list(df.feature_map(sorted_names=True)) == ["a", "b"]

    def test_empty(self):
        assert DataFeature().feature_map() == {}


class TestDataQualityQualityMap:
    def test_maps_by_name(self):
        q0 = Quality(name="z_last", value=1.0)
        q1 = Quality(name="a_first", value=2.0)
        dq = DataQuality(qualities=[q0, q1])

        assert dq.quality_map() == {"z_last": q0, "a_first": q1}
        assert list(dq.quality_map(sorted_names=True)) == ["a_first", "z_last"]


class TestRunEvaluation:
    def test_add_scores_extends(self):
        re_ = RunEvaluation(run_id=7)
        re_.add_scores([EvaluationScore(function="acc")])
        re_.add_scores([EvaluationScore(function="kappa")])

        assert [s.function for s in re_.scores] == ["acc", "kappa"]

    def test_defaults(self):
        re_ = RunEvaluation()
        assert re_.run_id is None
        assert re_.evaluation_engine_id == 1
        assert re_.scores == []
        assert re_.error is None
        assert re_.warning is None


class TestEvaluationScoreDefaults:
    def test_only_function_required(self):
        s = EvaluationScore(function="predictive_accuracy")
        assert s.value is None
        assert s.stdev is None
        assert s.array is None
        assert s.repeat is None
        assert s.fold is None
        assert s.sample is None
        assert s.sample_size is None


class TestEstimationProcedure:
    def test_frozen(self):
        ep = EstimationProcedure(type=EstimationProcedureType.HOLDOUT)
        with pytest.raises(AttributeError):  # FrozenInstanceError
            ep.percentage = 33.0

    def test_defaults(self):
        ep = EstimationProcedure(type=EstimationProcedureType.CROSSVALIDATION)
        assert ep.folds is None
        assert ep.repeats is None
        assert ep.percentage is None
