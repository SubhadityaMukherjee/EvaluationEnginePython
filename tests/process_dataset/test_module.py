from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.models import (
    DatasetDownloadInfo,
    EstimationProcedure,
    EstimationProcedureType,
)
from src.process_dataset.module import (
    _estimation_procedure_from_task,
    _procedure_parameters,
    _splits_for_procedure,
    generate_folds,
    generate_folds_for_task,
    load_dataset,
)


@pytest.fixture
def mock_download(mocker):
    info = DatasetDownloadInfo(
        file_path="/tmp/fake.arff", default_target_attribute="play"
    )
    return mocker.patch(
        "src.process_dataset.module.get_data_and_meta_information_from_did",
        return_value=info,
    )


@pytest.fixture
def mock_loader(mocker):
    loader_instance = MagicMock()
    loader_instance.load.return_value = (
        [
            ("outlook", ["sunny", "rainy"]),
            ("temp", "NUMERIC"),
            ("play", ["no", "yes"]),
        ],
        [("sunny", 75, "yes"), ("rainy", 68, "no")],
    )
    loader_cls = mocker.patch(
        "src.process_dataset.module.DataLoader", return_value=loader_instance
    )
    return loader_cls, loader_instance


class TestLoadDataset:
    def test_returns_df_and_target(self, mock_download, mock_loader):
        df, target = load_dataset(61)

        assert target == "play"
        assert list(df.columns) == ["outlook", "temp", "play"]
        assert df["temp"].tolist() == [75, 68]

    def test_nominal_columns_become_categorical(self, mock_download, mock_loader):
        df, _ = load_dataset(61)

        assert isinstance(df["outlook"].dtype, pd.CategoricalDtype)
        assert list(df["outlook"].cat.categories) == ["sunny", "rainy"]
        assert not isinstance(df["temp"].dtype, pd.CategoricalDtype)

    def test_row_indices_follow_file_order(self, mock_download, mock_loader):
        df, _ = load_dataset(61)

        assert list(df.index) == [0, 1]


class TestSplitsForProcedure:
    @pytest.mark.parametrize(
        "proc_type, splitter_name",
        [
            (EstimationProcedureType.CROSSVALIDATION, "crossvalidation_splits"),
            (
                EstimationProcedureType.LEARNINGCURVE_CV,
                "learning_curve_splits",
            ),
            (EstimationProcedureType.HOLDOUT, "holdout_splits"),
            (EstimationProcedureType.HOLDOUT_ORDERED, "holdout_ordered_splits"),
            (EstimationProcedureType.LEAVEONEOUT, "leave_one_out_splits"),
            (EstimationProcedureType.TESTONTRAININGDATA, "train_on_test_splits"),
        ],
    )
    def test_routes_to_matching_splitter(self, mocker, proc_type, splitter_name):
        splitter = mocker.patch(f"src.process_dataset.module.{splitter_name}")
        df = pd.DataFrame({"x": [1.0]})
        procedure = EstimationProcedure(type=proc_type, folds=2, repeats=1)

        _splits_for_procedure(df, procedure, target="x", seed=7)

        splitter.assert_called_once()

    def test_unknown_type_raises(self):
        bogus_type = "bogus"
        procedure = EstimationProcedure(type=bogus_type)

        with pytest.raises(ValueError, match="No fold generator"):
            _splits_for_procedure(pd.DataFrame(), procedure)


class TestProcedureParameters:
    def test_list_form(self):
        ep = {
            "oml:parameter": [
                {"@name": "number_folds", "#text": "10"},
                {"@name": "number_repeats", "#text": "5"},
            ]
        }

        assert _procedure_parameters(ep) == {
            "number_folds": "10",
            "number_repeats": "5",
        }

    def test_single_dict_form(self):
        ep = {"oml:parameter": {"@name": "percentage", "#text": "33"}}

        assert _procedure_parameters(ep) == {"percentage": "33"}

    def test_drops_empty_parameters(self):
        ep = {
            "oml:parameter": [
                {"@name": "number_folds", "#text": "10"},
                {"@name": "empty_one"},
            ]
        }

        assert _procedure_parameters(ep) == {"number_folds": "10"}


class TestEstimationProcedureFromTask:
    def test_crossvalidation_parameters(self):
        ep = {
            "oml:type": "crossvalidation",
            "oml:parameter": [
                {"@name": "number_folds", "#text": "10"},
                {"@name": "number_repeats", "#text": "5"},
            ],
        }

        procedure = _estimation_procedure_from_task(ep)

        assert procedure.type is EstimationProcedureType.CROSSVALIDATION
        assert procedure.folds == 10
        assert procedure.repeats == 5
        assert procedure.percentage is None

    def test_holdout_percentage(self):
        ep = {
            "oml:type": "holdout",
            "oml:parameter": [{"@name": "percentage", "#text": "33.0"}],
        }

        procedure = _estimation_procedure_from_task(ep)

        assert procedure.type is EstimationProcedureType.HOLDOUT
        assert procedure.percentage == 33.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported estimation procedure"):
            _estimation_procedure_from_task({"oml:type": "warp_drive"})


class TestGenerateFolds:
    def test_returns_splits_df_and_target(self, mocker):
        df = pd.DataFrame({"x": range(4)})
        mocker.patch(
            "src.process_dataset.module.load_dataset",
            return_value=(df, "x"),
        )
        splitter = mocker.patch(
            "src.process_dataset.module.crossvalidation_splits",
            return_value="SPLITS",
        )
        procedure = EstimationProcedure(
            type=EstimationProcedureType.CROSSVALIDATION, folds=2, repeats=1
        )

        splits, out_df, target = generate_folds(61, procedure, seed=3)

        assert splits == "SPLITS"
        assert out_df is df
        assert target == "x"
        splitter.assert_called_once_with(df, procedure, target="x", seed=3)


class TestGenerateFoldsForTask:
    def _task_xml(self):
        return {
            "oml:task_type_id": "1",
            "oml:input": [
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
                        "oml:parameter": [
                            {"@name": "number_folds", "#text": "2"},
                            {"@name": "number_repeats", "#text": "1"},
                        ],
                    },
                },
            ],
        }

    def test_reads_procedure_from_task(self, mocker):
        df = pd.DataFrame({"play": ["a", "b", "a", "b"]})
        mocker.patch(
            "src.process_dataset.module.get_task_xml",
            return_value=self._task_xml(),
        )
        mocker.patch(
            "src.process_dataset.module.load_dataset",
            return_value=(df, "play"),
        )
        splitter = mocker.patch(
            "src.process_dataset.module.crossvalidation_splits",
            return_value="SPLITS",
        )

        splits, out_df, target = generate_folds_for_task(1720)

        assert splits == "SPLITS"
        assert out_df is df
        assert target == "play"
        _, kwargs = splitter.call_args
        procedure = splitter.call_args.args[1]
        assert procedure.folds == 2
        assert procedure.repeats == 1
        assert kwargs["target"] == "play"

    def test_task_without_estimation_procedure_raises(self, mocker):
        task_xml = {
            "oml:input": [
                {
                    "@name": "source_data",
                    "oml:data_set": {"oml:data_set_id": "61"},
                }
            ]
        }
        mocker.patch("src.process_dataset.module.get_task_xml", return_value=task_xml)

        with pytest.raises(ValueError, match="estimation_procedure"):
            generate_folds_for_task(1720)
