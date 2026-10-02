import xmltodict

from src.models import EvaluationScore, RunEvaluation
from src.runs.serialization import (
    evaluation_score_to_oml_dict,
    run_evaluation_to_oml_dict,
    run_evaluation_to_xml,
)


class TestEvaluationScoreToOmlDict:
    def test_minimal_score_has_only_name(self):
        d = evaluation_score_to_oml_dict(EvaluationScore(function="accuracy"))

        assert d == {"oml:name": "accuracy"}

    def test_attributes_are_stringified(self):
        score = EvaluationScore(function="accuracy", repeat=0, fold=2, sample=1)

        d = evaluation_score_to_oml_dict(score)

        assert d["@repeat"] == "0"
        assert d["@fold"] == "2"
        assert d["@sample"] == "1"

    def test_value_uses_repr(self):
        score = EvaluationScore(function="accuracy", value=0.1)

        d = evaluation_score_to_oml_dict(score)

        assert d["oml:value"] == repr(0.1)

    def test_stdev_uses_repr(self):
        score = EvaluationScore(function="accuracy", stdev=0.05)

        assert evaluation_score_to_oml_dict(score)["oml:stdev"] == repr(0.05)

    def test_array_joined_with_commas(self):
        score = EvaluationScore(function="confusion_matrix", array=[1, 2, 3])

        assert evaluation_score_to_oml_dict(score)["oml:array_data"] == "1,2,3"

    def test_sample_size(self):
        score = EvaluationScore(function="accuracy", sample_size=64)

        assert evaluation_score_to_oml_dict(score)["oml:sample_size"] == "64"

    def test_none_fields_omitted(self):
        score = EvaluationScore(function="accuracy", value=None, stdev=None)

        d = evaluation_score_to_oml_dict(score)

        assert "oml:value" not in d
        assert "oml:stdev" not in d


class TestRunEvaluationToOmlDict:
    def test_envelope(self):
        run_eval = RunEvaluation(run_id=5, evaluation_engine_id=1)

        d = run_evaluation_to_oml_dict(run_eval)

        assert d == {
            "oml:run_evaluation": {
                "@xmlns:oml": "http://openml.org/openml",
                "oml:run_id": "5",
                "oml:evaluation_engine_id": "1",
            }
        }

    def test_error_and_warning_included_when_set(self):
        run_eval = RunEvaluation(run_id=5, error="boom", warning="hmm")

        d = run_evaluation_to_oml_dict(run_eval)

        assert d["oml:run_evaluation"]["oml:error"] == "boom"
        assert d["oml:run_evaluation"]["oml:warning"] == "hmm"

    def test_scores_become_evaluation_list(self):
        run_eval = RunEvaluation(
            run_id=5,
            scores=[
                EvaluationScore(function="acc", value=0.9),
                EvaluationScore(function="kappa", value=0.8, fold=1),
            ],
        )

        d = run_evaluation_to_oml_dict(run_eval)

        evaluations = d["oml:run_evaluation"]["oml:evaluation"]
        assert isinstance(evaluations, list)
        assert len(evaluations) == 2
        assert evaluations[0]["oml:name"] == "acc"


class TestRunEvaluationToXml:
    def test_round_trip(self):
        run_eval = RunEvaluation(
            run_id=42,
            scores=[
                EvaluationScore(
                    function="predictive_accuracy",
                    value=0.75,
                    repeat=0,
                    fold=1,
                ),
            ],
        )

        xml = run_evaluation_to_xml(run_eval, pretty=False)
        parsed = xmltodict.parse(xml)

        inner = parsed["oml:run_evaluation"]
        assert inner["oml:run_id"] == "42"
        eval_el = inner["oml:evaluation"]
        assert eval_el["@fold"] == "1"
        assert eval_el["oml:name"] == "predictive_accuracy"
        assert float(eval_el["oml:value"]) == 0.75

    def test_empty_scores_round_trip(self):
        run_eval = RunEvaluation(run_id=1, error="bad run")

        parsed = xmltodict.parse(run_evaluation_to_xml(run_eval, pretty=False))

        assert parsed["oml:run_evaluation"]["oml:error"] == "bad run"
        assert "oml:evaluation" not in parsed["oml:run_evaluation"]
