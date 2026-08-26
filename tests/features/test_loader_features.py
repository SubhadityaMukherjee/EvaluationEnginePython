from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.features.loader import load_features
from src.models import DataFeature


@pytest.fixture
def dataset():
    return SimpleNamespace(
        file_path="/tmp/fake.arff",
        default_target_attribute="class",
    )


@pytest.fixture
def mock_loader(mocker):
    loader_instance = MagicMock()
    loader_cls = mocker.patch(
        "src.features.loader.DataLoader", return_value=loader_instance
    )
    return loader_cls, loader_instance


@pytest.fixture
def mock_normalize_target_names(mocker):
    return mocker.patch(
        "src.features.loader.normalize_target_names",
        return_value={"class"},
    )


@pytest.fixture
def mock_liac_type(mocker):
    return mocker.patch(
        "src.features.loader._liac_type",
        return_value=("numeric", None),
    )


@pytest.fixture
def mock_fill_numeric(mocker):
    return mocker.patch("src.features.loader._fill_numeric_feature")


@pytest.fixture
def mock_fill_nominal(mocker):
    return mocker.patch("src.features.loader._fill_nominal_feature")


class TestLoadFeatures:
    def test_builds_features_with_correct_index_and_name(
        self,
        dataset,
        mock_loader,
        mock_normalize_target_names,
        mock_liac_type,
        mock_fill_numeric,
    ):
        _, loader_instance = mock_loader
        loader_instance.load.return_value = (
            [("age", "numeric"), ("class", "nominal")],
            [(25, "yes"), (30, "no")],
        )

        result = load_features(dataset, did=1, evaluation_engine_id=2)

        assert isinstance(result, DataFeature)
        assert result.did == 1
        assert result.evaluation_engine_id == 2
        assert result.error is None
        assert [f.name for f in result.features] == ["age", "class"]
        assert [f.index for f in result.features] == [0, 1]

    def test_uses_requested_data_format(
        self,
        dataset,
        mock_loader,
        mock_normalize_target_names,
        mock_liac_type,
        mock_fill_numeric,
    ):
        loader_cls, loader_instance = mock_loader
        loader_instance.load.return_value = ([], [])

        load_features(dataset, data_format="parquet")

        loader_cls.assert_called_once_with("parquet")

    def test_marks_target_column_as_is_target(
        self,
        dataset,
        mock_loader,
        mock_normalize_target_names,
        mock_liac_type,
        mock_fill_numeric,
    ):
        _, loader_instance = mock_loader
        loader_instance.load.return_value = (
            [("age", "numeric"), ("class", "nominal")],
            [(25, "yes")],
        )

        result = load_features(dataset)

        feats = {f.name: f for f in result.features}
        assert feats["class"].is_target is True
        assert feats["age"].is_target is False

    def test_no_rows_produces_empty_columns_without_crashing(
        self,
        dataset,
        mock_loader,
        mock_normalize_target_names,
        mock_liac_type,
        mock_fill_numeric,
    ):
        _, loader_instance = mock_loader
        loader_instance.load.return_value = (
            [("age", "numeric"), ("class", "nominal")],
            [],  # no rows at all
        )

        result = load_features(dataset)

        assert len(result.features) == 2
        # each feature should have been filled with an empty column
        for call in mock_fill_numeric.call_args_list:
            assert call.args[0] == ()

    def test_no_attributes_returns_empty_feature_list(
        self,
        dataset,
        mock_loader,
        mock_normalize_target_names,
    ):
        _, loader_instance = mock_loader
        loader_instance.load.return_value = ([], [])

        result = load_features(dataset)

        assert result.features == []
        assert result.error is None
