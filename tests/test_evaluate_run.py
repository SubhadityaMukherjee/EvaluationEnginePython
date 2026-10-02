from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.evaluate_run import (
    EvaluateRun,
    _cost_matrix_from_task,
    _estimation_procedure_type,
    _load_run_inputs,
)
from src.exceptions import OpenmlApiError
from src.models import EstimationProcedureType, EvaluationScore, RunEvaluation


def task_xml(task_type_id=1, with_splits_url=True, with_cost_matrix=False):
    inputs = [
        {
            "@name": "source_data",
            "oml:data_set": {
                "oml:data_set_id": "61",
                "oml:target_feature": "play",
            },
        },
        {
            "@name": "estimation_procedure",
            "oml:estimation_procedure": {
                "oml:type": "crossvalidation",
                **(
                    {"oml:data_splits_url": "https://x/splits.arff"}
                    if with_splits_url
                    else {}
                ),
            },
        },
    ]
    if with_cost_matrix:
        inputs.append(
            {
                "@name": "cost_matrix",
                "oml:cost_matrix": {"#text": "[[0, 2], [3, 0]]"},
            }
        )
    return {
        "oml:task_type_id": str(task_type_id),
        "oml:task_type": "Supervised Classification",
        "oml:input": inputs,
    }


def run_xml(include=("description", "predictions")):
    files = [
        {"oml:name": name, "oml:file_id": f"{100 + i}"}
        for i, name in enumerate(include)
    ]
    return {
        "oml:task_id": "1720",
        "oml:output_data": {"oml:file": files},
    }


@pytest.fixture
def client():
    c = MagicMock()
    c.base_url = "https://test.openml.org/api/v1/"
    return c


class TestEstimationProcedureType:
    def test_resolves_from_task(self):
        assert (
            _estimation_procedure_type(task_xml())
            is EstimationProcedureType.CROSSVALIDATION
        )

    def test_absent_returns_none(self):
        assert _estimation_procedure_type({"oml:input": []}) is None


class TestCostMatrixFromTask:
    def test_json_string_form(self):
        cm = _cost_matrix_from_task(task_xml(with_cost_matrix=True))

        assert cm is not None
        np.testing.assert_allclose(cm, [[0.0, 2.0], [3.0, 0.0]])

    def test_list_of_lists_form(self):
        xml = {
            "oml:input": {
                "@name": "cost_matrix",
                "oml:cost_matrix": [[1, 2], [3, 4]],
            }
        }

        cm = _cost_matrix_from_task(xml)

        assert cm is not None
        np.testing.assert_allclose(cm, [[1.0, 2.0], [3.0, 4.0]])

    def test_absent_returns_none(self):
        assert _cost_matrix_from_task(task_xml()) is None

    def test_non_2d_returns_none(self):
        xml = {
            "oml:input": {
                "@name": "cost_matrix",
                "oml:cost_matrix": {"#text": "[1, 2, 3]"},
            }
        }

        assert _cost_matrix_from_task(xml) is None


class TestEvaluate:
    @pytest.fixture
    def wired(self, mocker, client):
        """Patch the REST + pipeline collaborators of EvaluateRun.evaluate."""
        mocker.patch(
            "src.evaluate_run.get_run_xml",
            return_value=run_xml(),
        )
        mocker.patch(
            "src.evaluate_run.get_task_xml",
            return_value=task_xml(),
        )
        load_inputs = mocker.patch(
            "src.evaluate_run._load_run_inputs",
            return_value=(MagicMock(), MagicMock(), MagicMock()),
        )
        compute = mocker.patch.object(
            EvaluateRun,
            "_compute_scores",
            return_value=[EvaluationScore(function="acc", value=0.5)],
        )
        return load_inputs, compute

    def test_happy_path(self, client, wired):
        er = EvaluateRun(client=client)

        result = er.evaluate(5)

        assert result.run_id == 5
        assert result.error is None
        assert [s.function for s in result.scores] == ["acc"]
        assert er.last_result is result
        client.run_evaluate_upload.assert_called_once_with(result)

    def test_missing_description_file_sets_error(self, client, mocker):
        mocker.patch(
            "src.evaluate_run.get_run_xml",
            return_value=run_xml(include=["predictions"]),
        )
        mocker.patch("src.evaluate_run.get_task_xml", return_value=task_xml())

        result = EvaluateRun(client=client).evaluate(5)

        assert result.error == "Run description file not present."
        client.run_evaluate_upload.assert_called_once()

    def test_missing_predictions_file_sets_error(self, client, mocker):
        mocker.patch(
            "src.evaluate_run.get_run_xml",
            return_value=run_xml(include=["description"]),
        )
        mocker.patch("src.evaluate_run.get_task_xml", return_value=task_xml())

        result = EvaluateRun(client=client).evaluate(5)

        assert "Required output files not present" in (result.error or "")

    def test_unsupported_task_type_sets_error(self, client, mocker):
        mocker.patch(
            "src.evaluate_run.get_run_xml",
            return_value=run_xml(),
        )
        mocker.patch(
            "src.evaluate_run.get_task_xml",
            return_value=task_xml(task_type_id=99),
        )

        result = EvaluateRun(client=client).evaluate(5)

        assert "Task type not supported" in (result.error or "")

    def test_exception_is_captured_and_uploaded(self, client, mocker):
        mocker.patch(
            "src.evaluate_run.get_run_xml",
            side_effect=OSError("network down"),
        )

        result = EvaluateRun(client=client).evaluate(5)

        assert "network down" in (result.error or "")
        client.run_evaluate_upload.assert_called_once()

    def test_upload_disabled_skips_upload(self, client, wired):
        er = EvaluateRun(client=client, upload=False)

        result = er.evaluate(5)

        client.run_evaluate_upload.assert_not_called()
        assert result.error is None

    def test_upload_failure_triggers_error_only_retry(self, client, wired):
        client.run_evaluate_upload.side_effect = [
            OpenmlApiError(500, "upload failed"),
            None,
        ]

        EvaluateRun(client=client).evaluate(5)

        assert client.run_evaluate_upload.call_count == 2
        retry = client.run_evaluate_upload.call_args_list[1].args[0]
        assert retry.error is not None
        assert "upload failed" in retry.error

    def test_constructor_with_run_id_evaluates_immediately(self, client, wired):
        EvaluateRun(run_id=7, client=client)

        assert client.run_evaluate_upload.called


class TestComputeScores:
    def test_batch_returns_per_cell_plus_global(self, client, mocker):
        from src.models import EvaluationScore as ES

        mocker.patch(
            "src.evaluate_run.evaluate_batch",
            return_value=(
                [ES(function="fold_acc", fold=0)],
                [ES(function="acc")],
                MagicMock(),
            ),
        )

        er = EvaluateRun(client=client, upload=False)
        scores = er._compute_scores(
            task_type_id=1,
            task_xml=task_xml(),
            dataset_df=MagicMock(),
            splits_df=MagicMock(),
            predictions_df=MagicMock(),
            target_feature="play",
        )

        assert [s.function for s in scores] == ["fold_acc", "acc"]

    def test_survival_requires_splits(self, client):
        er = EvaluateRun(client=client, upload=False)

        with pytest.raises(ValueError, match="Splits required"):
            er._compute_scores(
                task_type_id=7,
                task_xml=task_xml(),
                dataset_df=MagicMock(),
                splits_df=None,
                predictions_df=MagicMock(),
                target_feature="play",
            )


class TestLoadRunInputs:
    def test_downloads_splits_and_predictions(self, client, mocker, tmp_path):
        splits_path = tmp_path / "splits.arff"
        splits_path.write_text("splits")
        preds_path = tmp_path / "preds.arff"
        preds_path.write_text("preds")

        mocker.patch(
            "src.process_dataset.module.load_dataset",
            return_value=(pd.DataFrame({"play": ["a"]}), "play"),
        )
        download = mocker.patch(
            "src.evaluate_run.download_to_temp_file",
            side_effect=[str(splits_path), str(preds_path)],
        )
        mocker.patch(
            "src.evaluate_run.load_arff_to_df",
            side_effect=[
                pd.DataFrame({"type": ["TRAIN"]}),
                pd.DataFrame({"row_id": [0]}),
            ],
        )

        _dataset_df, splits_df, predictions_df = _load_run_inputs(
            task_xml=task_xml(),
            dataset_id=61,
            file_ids={"predictions": "123"},
            run_id=5,
            base_url=client.base_url,
        )

        assert splits_df is not None
        assert list(predictions_df.columns) == ["row_id"]
        # splits URL from the task, predictions URL derived from file id
        assert download.call_args_list[0].args[0] == "https://x/splits.arff"
        assert (
            download.call_args_list[1]
            .args[0]
            .endswith("data/download/123/Run_5_predictions.arff")
        )

    def test_no_splits_url_gives_none(self, client, mocker):
        mocker.patch(
            "src.process_dataset.module.load_dataset",
            return_value=(pd.DataFrame(), "play"),
        )
        mocker.patch(
            "src.evaluate_run.download_to_temp_file",
            return_value="/tmp/preds.arff",
        )
        mocker.patch(
            "src.evaluate_run.load_arff_to_df",
            return_value=pd.DataFrame({"row_id": [0]}),
        )

        _, splits_df, _ = _load_run_inputs(
            task_xml=task_xml(with_splits_url=False),
            dataset_id=61,
            file_ids={"predictions": "123"},
            run_id=5,
            base_url=client.base_url,
        )

        assert splits_df is None


class TestPoll:
    def test_evaluates_returned_runs_then_stops(self, client, mocker):
        client.evaluation_request.side_effect = [
            [1, 2],
            OpenmlApiError(1013, "no unevaluated runs"),
        ]
        evaluate = mocker.patch.object(
            EvaluateRun, "evaluate", return_value=RunEvaluation()
        )

        EvaluateRun(client=client).poll()

        assert [c.args[0] for c in evaluate.call_args_list] == [1, 2]

    def test_filters_built_from_fields(self, client, mocker):
        client.evaluation_request.side_effect = [
            OpenmlApiError(1013, "done"),
        ]

        EvaluateRun(
            client=client,
            task_type_ids={1, 2},
            task_ids="1720,1721",
            tag="study",
            uploader_id=42,
        ).poll()

        kwargs = client.evaluation_request.call_args.kwargs
        filters = kwargs["filters"]
        assert filters["ttid"] == "1,2"
        assert filters["task"] == "1720,1721"
        assert filters["tag"] == "study"
        assert filters["uploader"] == "42"

    def test_other_api_errors_propagate(self, client):
        client.evaluation_request.side_effect = OpenmlApiError(500, "boom")

        with pytest.raises(OpenmlApiError):
            EvaluateRun(client=client).poll()
