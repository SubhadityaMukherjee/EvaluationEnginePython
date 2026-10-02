from unittest.mock import MagicMock

import pytest

from src.exceptions import OpenmlApiError
from src.models import DataFeature, DataQuality
from src.process_dataset.engine import ProcessDataset


@pytest.fixture
def client():
    c = MagicMock()
    c.base_url = "https://test.openml.org/api/v1/"
    return c


@pytest.fixture
def mocks(mocker):
    """Patch every collaborator of ProcessDataset.process."""
    dsd = mocker.patch(
        "src.process_dataset.engine.get_dataset_description_xml",
        return_value={
            "oml:default_target_attribute": "play",
            "oml:status": "active",
        },
    )
    info = mocker.patch(
        "src.process_dataset.engine.get_data_and_meta_information_from_did",
        return_value=MagicMock(),
    )
    features = mocker.patch(
        "src.process_dataset.engine.load_features",
        return_value=DataFeature(did=61, features=[]),
    )
    qualities = mocker.patch(
        "src.process_dataset.engine.load_qualities",
        return_value=DataQuality(did=61, qualities=[]),
    )
    return MagicMock(dsd=dsd, info=info, features=features, qualities=qualities)


class TestProcess:
    def test_happy_path_uploads_features_and_qualities(self, client, mocks):
        engine = ProcessDataset(client=client)

        features, qualities = engine.process(61)

        client.data_features_upload.assert_called_once_with(features)
        client.data_qualities_upload.assert_called_once_with(qualities)
        assert engine.last_features is features
        assert engine.last_qualities is qualities

    def test_active_dataset_does_not_flip_status(self, client, mocks):
        ProcessDataset(client=client).process(61)

        client.data_status_update.assert_not_called()

    def test_in_preparation_flips_to_active(self, client, mocks):
        mocks.dsd.return_value = {
            "oml:default_target_attribute": "play",
            "oml:status": "in_preparation",
        }

        ProcessDataset(client=client).process(61)

        client.data_status_update.assert_called_once_with(61, "active")

    def test_code_441_during_features_upload_is_swallowed(self, client, mocks):
        client.data_features_upload.side_effect = OpenmlApiError(
            441, "features already uploaded"
        )

        ProcessDataset(client=client).process(61)

        client.data_qualities_upload.assert_called_once()

    def test_other_api_error_during_features_upload_is_recorded(self, client, mocks):
        client.data_features_upload.side_effect = OpenmlApiError(
            432, "some other problem"
        )

        features, qualities = ProcessDataset(client=client).process(61)

        assert features.error and "some other problem" in features.error
        assert qualities.error and "some other problem" in qualities.error
        client.data_qualities_upload.assert_not_called()

    def test_code_431_short_circuits_as_already_processed(self, client, mocks):
        mocks.qualities.side_effect = OpenmlApiError(431, "already processed")

        features, qualities = ProcessDataset(client=client).process(61)

        assert features.error and "Dataset already processed" in features.error
        assert qualities.error and "Dataset already processed" in qualities.error

    def test_unexpected_exception_is_recorded_and_uploaded(self, client, mocks):
        mocks.info.side_effect = ValueError("download exploded")

        features, _ = ProcessDataset(client=client).process(61)

        assert features.error and "download exploded" in features.error
        # Java uploads the error-carrying feature object
        assert client.data_features_upload.call_count >= 1

    def test_missing_default_target_sets_feature_error(self, client, mocks):
        mocks.dsd.return_value = {
            "oml:default_target_attribute": None,
            "oml:status": "active",
        }

        features, _ = ProcessDataset(client=client).process(61)

        assert features.error and "no default_target_attribute" in features.error
        mocks.features.assert_not_called()

    def test_constructor_with_dataset_id_processes_immediately(self, client, mocks):
        ProcessDataset(dataset_id=61, client=client)

        assert client.data_features_upload.called


class TestProcessAndPrint:
    def test_prints_feature_xml_without_uploading(self, client, mocks, capsys):
        ProcessDataset(client=client).process_and_print(61)

        out = capsys.readouterr().out
        assert "oml:data_features" in out
        client.data_features_upload.assert_not_called()
        client.data_qualities_upload.assert_not_called()


class TestPoll:
    def test_processes_each_returned_dataset_then_stops(self, client, mocks, mocker):
        client.data_unprocessed.side_effect = [
            [61, 62],
            OpenmlApiError(542, "No unprocessed datasets remaining"),
        ]
        process = mocker.patch.object(ProcessDataset, "process")

        ProcessDataset(client=client).poll()

        assert process.call_args_list == [mocker.call(61), mocker.call(62)]

    def test_empty_list_stops_without_processing(self, client, mocks):
        client.data_unprocessed.return_value = []

        ProcessDataset(client=client).poll()

        client.data_unprocessed.assert_called_once()

    def test_other_api_errors_propagate(self, client, mocks):
        client.data_unprocessed.side_effect = OpenmlApiError(500, "server sad")

        with pytest.raises(OpenmlApiError):
            ProcessDataset(client=client).poll()
