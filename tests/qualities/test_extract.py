from unittest.mock import MagicMock

import pytest

from src.exceptions import OpenmlApiError
from src.models import DataQuality, Quality
from src.qualities.extract import ExtractFeatures
from src.qualities.landmarkers import expected_landmarker_ids


@pytest.fixture
def client():
    c = MagicMock()
    c.base_url = "https://test.openml.org/api/v1/"
    return c


@pytest.fixture
def mock_download(mocker):
    return mocker.patch(
        "src.qualities.extract.get_data_and_meta_information_from_did",
        return_value=MagicMock(),
    )


@pytest.fixture
def mock_load_qualities(mocker):
    return mocker.patch(
        "src.qualities.extract.load_qualities",
        return_value=DataQuality(
            did=61, qualities=[Quality(name="NumberOfInstances", value=3.0)]
        ),
    )


class TestConstructor:
    def test_invalid_characterizer_set_raises(self, client):
        with pytest.raises(ValueError, match="characterizer_set"):
            ExtractFeatures(client, characterizer_set="deluxe")

    @pytest.mark.parametrize("char_set", ["simple", "all"])
    def test_valid_sets(self, client, char_set):
        assert (
            ExtractFeatures(client, characterizer_set=char_set).characterizer_set
            == char_set
        )


class TestProcess:
    def test_simple_set_uploads_qualities(
        self, client, mock_download, mock_load_qualities
    ):
        ef = ExtractFeatures(client, characterizer_set="simple")

        result = ef.process(61)

        client.data_qualities_upload.assert_called_once_with(result)
        assert ef.last_result is result
        assert [q.name for q in result.qualities] == ["NumberOfInstances"]

    def test_all_set_appends_landmarkers(
        self, client, mock_download, mock_load_qualities, mocker
    ):
        loader = MagicMock()
        loader.load.return_value = ([("x", "NUMERIC")], [(1.0,), (2.0,)])
        mocker.patch("src.qualities.extract.DataLoader", return_value=loader)
        mocker.patch(
            "src.qualities.extract.build_xy",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch(
            "src.qualities.extract.compute_all_landmarkers",
            return_value={"kNN1NAUC": 0.9, "kNN1NErrRate": 0.1},
        )

        result = ExtractFeatures(client, characterizer_set="all").process(61)

        by_name = {q.name: q.value for q in result.qualities}
        assert by_name["kNN1NAUC"] == 0.9
        assert by_name["kNN1NErrRate"] == 0.1

    def test_all_set_skips_landmarkers_on_data_quality_error(
        self, client, mock_download, mock_load_qualities, mocker
    ):
        mock_load_qualities.return_value = DataQuality(did=61, error="bad")
        landmarks = mocker.patch("src.qualities.extract.compute_all_landmarkers")

        ExtractFeatures(client, characterizer_set="all").process(61)

        landmarks.assert_not_called()

    def test_all_set_landmarker_failure_is_silent(
        self, client, mock_download, mock_load_qualities, mocker
    ):
        # the loader must succeed so the failure is genuinely build_xy's
        loader = MagicMock()
        loader.load.return_value = ([("x", "NUMERIC")], [(1.0,), (2.0,)])
        mocker.patch("src.qualities.extract.DataLoader", return_value=loader)
        build_xy = mocker.patch(
            "src.qualities.extract.build_xy",
            side_effect=ValueError("no target"),
        )

        result = ExtractFeatures(client, characterizer_set="all").process(61)

        # the patched build_xy really was the failure point
        build_xy.assert_called_once()
        # base qualities survive, no landmarker entries, no error
        assert result.error is None
        assert [q.name for q in result.qualities] == ["NumberOfInstances"]

    def test_upload_error_propagates(self, client, mock_download, mock_load_qualities):
        client.data_qualities_upload.side_effect = OpenmlApiError(
            431, "already processed"
        )

        with pytest.raises(OpenmlApiError):
            ExtractFeatures(client).process(61)

    def test_empty_qualities_skip_upload(
        self, client, mock_download, mock_load_qualities
    ):
        mock_load_qualities.return_value = DataQuality(did=61)

        ExtractFeatures(client).process(61)

        client.data_qualities_upload.assert_not_called()


class TestExpectedQualityIds:
    def test_simple_set_is_the_19_simple_meta_features(self, client):
        ids = ExtractFeatures(client)._expected_quality_ids()

        assert len(ids) == 19
        assert "NumberOfInstances" in ids
        assert "AutoCorrelation" in ids

    def test_all_set_extends_with_landmarker_ids(self, client):
        ids = ExtractFeatures(client, characterizer_set="all")._expected_quality_ids()

        assert len(ids) == 19 + len(expected_landmarker_ids())
        assert "kNN1NAUC" in ids


class TestPoll:
    def test_processes_returned_datasets_then_stops(
        self, client, mock_download, mock_load_qualities, mocker
    ):
        client.data_qualities_unprocessed.side_effect = [
            [61, 62],
            OpenmlApiError(542, "No unprocessed datasets"),
        ]
        process = mocker.patch.object(
            ExtractFeatures, "process", return_value=DataQuality()
        )

        ExtractFeatures(client).poll()

        assert process.call_args_list == [mocker.call(61), mocker.call(62)]

    def test_requests_use_expected_qualities_filter(
        self, client, mock_download, mock_load_qualities, mocker
    ):
        client.data_qualities_unprocessed.side_effect = [
            [],
        ]

        ExtractFeatures(client, priority_tag="staging").poll()

        call = client.data_qualities_unprocessed.call_args
        assert call.args[2] == ExtractFeatures(client)._expected_quality_ids()
        assert call.kwargs["priority_tag"] == "staging"
