import pandas as pd
import pytest

from src.main import _build_parser, main
from src.models import EvaluationScore, RunEvaluation


def make_client():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.base_url = "https://test.openml.org/api/v1/"
    return client


class TestParser:
    def test_function_is_required(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_defaults(self):
        args = _build_parser().parse_args(["-f", "evaluate_run"])

        assert args.id is None
        assert args.dataset_format == "arff"
        assert args.test is False
        assert args.no_upload is False

    def test_short_and_long_forms(self):
        short = _build_parser().parse_args(
            ["-f", "generate_folds", "-id", "5", "-o", "out.arff"]
        )
        long_ = _build_parser().parse_args(
            [
                "--function",
                "generate_folds",
                "--id",
                "5",
                "--output",
                "out.arff",
            ]
        )

        assert short == long_


class TestDispatch:
    def test_unknown_function_returns_one(self, capsys):
        code = main(["-f", "warp_drive"])

        assert code == 1
        assert "unknown function" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "function",
        ["all_wrong", "different_predictions", "challenge"],
    )
    def test_not_ported_functions_return_zero(self, function, capsys):
        code = main(["-f", function])

        assert code == 0
        assert "Not implemented" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "function, arg",
        [
            ("evaluate_run", "--id"),
            ("process_dataset_print", "--id"),
            ("generate_folds", "--id"),
            ("merge_datasets", "--id"),
        ],
    )
    def test_missing_id_exits(self, function, arg):
        with pytest.raises(SystemExit):
            main(["-f", function])


class TestEvaluateRunCommand:
    @pytest.fixture
    def mock_engine(self, mocker):
        instance = mocker.MagicMock()
        engine_cls = mocker.patch("src.evaluate_run.EvaluateRun", return_value=instance)
        return engine_cls, instance

    @pytest.fixture
    def mock_client(self, mocker):
        return mocker.patch("src.client.OpenmlClient", return_value=make_client())

    def test_prints_xml_in_dry_run_mode(self, mock_engine, mock_client, capsys):
        mock_engine[1].last_result = RunEvaluation(
            run_id=5,
            scores=[EvaluationScore(function="acc", value=0.5)],
        )

        code = main(["-f", "evaluate_run", "--id", "5", "--no-upload"])

        assert code == 0
        out = capsys.readouterr().out
        assert "oml:run_evaluation" in out
        assert "acc" in out

    def test_prints_score_summary_when_uploading(
        self, mock_engine, mock_client, capsys
    ):
        mock_engine[1].last_result = RunEvaluation(
            run_id=5,
            scores=[
                EvaluationScore(function="acc", value=0.9),
                EvaluationScore(function="fold_acc", value=0.8, fold=0),
            ],
        )

        code = main(["-f", "evaluate_run", "--id", "5"])

        assert code == 0
        out = capsys.readouterr().out
        assert "2 scores (1 per-cell, 1 global)" in out
        assert "acc" in out

    def test_prints_error_to_stderr(self, mock_engine, mock_client, capsys):
        mock_engine[1].last_result = RunEvaluation(run_id=5, error="task vanished")

        code = main(["-f", "evaluate_run", "--id", "5"])

        assert code == 0
        assert "task vanished" in capsys.readouterr().err

    def test_non_integer_mode_exits(self, mock_engine, mock_client):
        with pytest.raises(SystemExit):
            main(["-f", "evaluate_run", "--id", "5", "--mode", "fast"])

    def test_integer_mode_overrides_ttid_filter(self, mock_engine, mock_client):
        mock_engine[1].last_result = RunEvaluation(run_id=5)

        main(["-f", "evaluate_run", "--id", "5", "--mode", "2"])

        kwargs = mock_engine[0].call_args.kwargs
        assert kwargs["task_type_ids"] == {2}


class TestProcessDatasetCommand:
    @pytest.fixture
    def mock_engine_cls(self, mocker):
        return mocker.patch("src.process_dataset.ProcessDataset")

    @pytest.fixture
    def mock_client(self, mocker):
        return mocker.patch("src.client.OpenmlClient", return_value=make_client())

    def test_without_id_polls(self, mock_engine_cls, mock_client):
        main(["-f", "process_dataset"])

        mock_engine_cls.return_value.poll.assert_called_once()

    def test_with_id_reports_counts(self, mock_engine_cls, mock_client, capsys):
        from src.models import DataFeature, DataQuality, Feature

        instance = mock_engine_cls.return_value
        instance.last_features = DataFeature(
            features=[Feature(index=0, name="a", data_type="numeric")]
        )
        instance.last_qualities = DataQuality(qualities=[])

        main(["-f", "process_dataset", "--id", "61"])

        out = capsys.readouterr().out
        assert "1 features extracted" in out
        assert "0 qualities extracted" in out

    def test_feature_error_reported_on_stderr(
        self, mock_engine_cls, mock_client, capsys
    ):
        from src.models import DataFeature, DataQuality

        instance = mock_engine_cls.return_value
        instance.last_features = DataFeature(error="boom")
        instance.last_qualities = DataQuality()

        main(["-f", "process_dataset", "--id", "61"])

        assert "boom" in capsys.readouterr().err


class TestGenerateFoldsCommand:
    @pytest.fixture
    def mock_folds(self, mocker):
        splits = pd.DataFrame(
            {
                "type": ["TRAIN", "TEST"],
                "rowid": [0, 1],
                "repeat": [0, 0],
                "fold": [0, 0],
            }
        )
        return mocker.patch(
            "src.process_dataset.module.generate_folds_for_task",
            return_value=(splits, pd.DataFrame(), "play"),
        )

    @pytest.fixture
    def mock_client(self, mocker):
        client = make_client()
        mocker.patch("src.client.OpenmlClient", return_value=client)
        return client

    def test_writes_output_file(self, mock_folds, mock_client, tmp_path):
        out = tmp_path / "splits.arff"

        code = main(["-f", "generate_folds", "--id", "1720", "-o", str(out)])

        assert code == 0
        assert out.exists()
        assert "TRAIN" in out.read_text()

    def test_prints_to_stdout_without_output(self, mock_folds, mock_client, capsys):
        code = main(["-f", "generate_folds", "--id", "1720"])

        assert code == 0
        assert "TRAIN" in capsys.readouterr().out


class TestExtractFeaturesCommand:
    @pytest.fixture
    def mock_extract(self, mocker):
        instance = mocker.MagicMock()
        cls = mocker.patch(
            "src.qualities.extract.ExtractFeatures", return_value=instance
        )
        return cls, instance

    @pytest.fixture
    def mock_client(self, mocker):
        return mocker.patch("src.client.OpenmlClient", return_value=make_client())

    def test_without_id_polls(self, mock_extract, mock_client):
        main(["-f", "extract_features_simple"])

        mock_extract[1].poll.assert_called_once()

    def test_simple_uses_simple_set(self, mock_extract, mock_client):
        from src.models import DataQuality

        mock_extract[1].process.return_value = DataQuality(qualities=[])

        main(["-f", "extract_features_simple", "--id", "61"])

        assert mock_extract[0].call_args.kwargs["characterizer_set"] == "simple"

    def test_all_uses_all_set(self, mock_extract, mock_client):
        from src.models import DataQuality

        mock_extract[1].process.return_value = DataQuality(qualities=[])

        main(["-f", "extract_features_all", "--id", "61"])

        assert mock_extract[0].call_args.kwargs["characterizer_set"] == "all"
