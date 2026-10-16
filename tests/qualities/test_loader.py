from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.models import DataQuality
from src.qualities.loader import load_qualities

ATTRIBUTES = [
    ("outlook", ["sunny", "rainy"]),
    ("play", ["no", "yes"]),
]
ROWS = [("sunny", "yes"), ("rainy", "no"), ("sunny", "no")]


@pytest.fixture
def dataset():
    return SimpleNamespace(file_path="/tmp/fake.arff", default_target_attribute="play")


@pytest.fixture
def mock_loader(mocker):
    loader_instance = MagicMock()
    loader_instance.load.return_value = (ATTRIBUTES, ROWS)
    loader_cls = mocker.patch(
        "src.qualities.loader.DataLoader", return_value=loader_instance
    )
    return loader_cls, loader_instance


@pytest.fixture
def mock_mfe(mocker):
    mfe_instance = MagicMock()
    mfe_instance.extract.return_value = (
        ["mfe_one", "mfe_none", "mfe_nan"],
        [1.5, None, float("nan")],
    )
    mfe_cls = mocker.patch("src.qualities.loader.MFE", return_value=mfe_instance)
    return mfe_cls, mfe_instance


class TestLoadQualities:
    def test_combines_simple_and_mfe_qualities(self, dataset, mock_loader, mock_mfe):
        result = load_qualities(dataset, did=61, evaluation_engine_id=1)

        assert isinstance(result, DataQuality)
        assert result.error is None
        names = [q.name for q in result.qualities]
        assert "NumberOfInstances" in names
        assert "mfe_one" in names

    def test_simple_quality_golden_value(self, dataset, mock_loader, mock_mfe):
        result = load_qualities(dataset)

        by_name = {q.name: q.value for q in result.qualities}
        assert by_name["NumberOfInstances"] == 3.0
        assert by_name["NumberOfClasses"] == 2.0

    def test_mfe_none_and_nan_become_none(self, dataset, mock_loader, mock_mfe):
        result = load_qualities(dataset)

        by_name = {q.name: q.value for q in result.qualities}
        assert by_name["mfe_one"] == 1.5
        assert by_name["mfe_none"] is None
        assert by_name["mfe_nan"] is None

    def test_mfe_configured_with_groups_and_seed(self, dataset, mock_loader, mock_mfe):
        mfe_cls, _ = mock_mfe

        load_qualities(dataset, groups=("general",), random_state=7)

        mfe_cls.assert_called_once_with(groups=("general",), random_state=7)

    def test_uses_requested_data_format(self, dataset, mock_loader, mock_mfe):
        loader_cls, _ = mock_loader

        load_qualities(dataset, data_format="parquet")

        loader_cls.assert_called_once_with("parquet")

    def test_loader_failure_is_captured_as_error(self, dataset, mocker):
        mocker.patch(
            "src.qualities.loader.DataLoader",
            side_effect=OSError("file gone"),
        )

        result = load_qualities(dataset, did=61)

        assert result.error == "file gone"
        assert result.qualities == []

    def test_no_target_keeps_base_qualities_and_skips_mfe(
        self, dataset, mock_loader, mock_mfe
    ):
        # no default_target_attribute → supervised extraction is skipped
        # explicitly; target-independent base qualities survive
        dataset.default_target_attribute = None

        result = load_qualities(dataset)

        assert result.error is None
        by_name = {q.name: q.value for q in result.qualities}
        assert by_name["NumberOfInstances"] == 3.0
        assert by_name["NumberOfFeatures"] == 2.0
        assert by_name["NumberOfClasses"] is None
        assert by_name["AutoCorrelation"] is None
        assert not any(name.startswith("mfe_") for name in by_name)
        mfe_cls, _ = mock_mfe
        mfe_cls.assert_not_called()

    def test_unknown_target_name_skips_mfe(
        self, dataset, mock_loader, mock_mfe
    ):
        # a target that matches no attribute is as good as no target
        dataset.default_target_attribute = "not_a_column"

        result = load_qualities(dataset)

        assert result.error is None
        assert not any(
            q.name.startswith("mfe_") for q in result.qualities
        )
        mfe_cls, _ = mock_mfe
        mfe_cls.assert_not_called()
