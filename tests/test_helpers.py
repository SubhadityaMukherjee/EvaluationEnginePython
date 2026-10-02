import numpy as np
import pandas as pd
import pytest

from src.exceptions import PredictionValidationError
from src.helpers import (
    _server_root,
    class_counts,
    class_ratios,
    download_and_parse,
    download_to_temp_file,
    get_data_and_meta_information_from_did,
    get_row_index,
    get_row_index_multi,
    load_arff_to_df,
    normalize_target_names,
    openml_file_url,
    prediction_to_confidences,
    run_output_file_ids,
    task_cost_matrix,
    task_estimation_procedure,
    task_source_data,
    to_prob_dist,
)
from src.models import DatasetDownloadInfo


class TestServerRoot:
    @pytest.mark.parametrize(
        "base_url, expected",
        [
            ("https://www.openml.org/api/v1/", "https://www.openml.org/"),
            ("https://test.openml.org/api/v1/", "https://test.openml.org/"),
        ],
    )
    def test_strips_api_suffix(self, base_url, expected):
        assert _server_root(base_url) == expected

    def test_openml_file_url_uses_server_root(self):
        url = openml_file_url(
            "1234",
            "predictions.arff",
            "https://test.openml.org/api/v1/",
        )
        assert url == "https://test.openml.org/data/download/1234/predictions.arff"


class TestDownloadAndParse:
    def test_parses_xml_body(self, mocker):
        response = mocker.Mock(content=b"<oml:run><oml:run_id>1</oml:run_id></oml:run>")
        get = mocker.patch("src.helpers.requests.get", return_value=response)

        parsed = download_and_parse("https://example.org/run/1")

        get.assert_called_once_with("https://example.org/run/1")
        assert parsed["oml:run"]["oml:run_id"] == "1"

    def test_raises_on_http_error(self, mocker):
        response = mocker.Mock()
        response.raise_for_status.side_effect = RuntimeError("boom")
        mocker.patch("src.helpers.requests.get", return_value=response)

        with pytest.raises(RuntimeError, match="boom"):
            download_and_parse("https://example.org/404")


class TestDownloadToTempFile:
    def test_writes_chunks_to_temp_file(self, mocker, tmp_path):
        response = mocker.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"abc", b"def"]
        mocker.patch("src.helpers.requests.get", return_value=response)
        target = tmp_path / "download.bin"
        # intentionally not a context manager: the code under test closes it
        real_file = open(target, "wb")  # noqa: SIM115
        mocker.patch("src.helpers.NamedTemporaryFile", return_value=real_file)

        path = download_to_temp_file("https://example.org/file", suffix=".bin")

        assert path.endswith(".bin")
        with open(path, "rb") as f:
            assert f.read() == b"abcdef"

    def test_propagates_http_error(self, mocker):
        response = mocker.Mock()
        response.__enter__ = mocker.Mock(return_value=response)
        response.__exit__ = mocker.Mock(return_value=False)
        response.raise_for_status.side_effect = RuntimeError("404")
        mocker.patch("src.helpers.requests.get", return_value=response)

        with pytest.raises(RuntimeError, match="404"):
            download_to_temp_file("https://example.org/missing")


class TestNormalizeTargetNames:
    def test_none_gives_empty_set(self):
        assert normalize_target_names(None) == set()

    def test_single_string(self):
        assert normalize_target_names("class") == {"class"}

    def test_comma_separated_string_is_split(self):
        assert normalize_target_names("a, b ,c,") == {"a", "b", "c"}

    def test_list_strips_and_drops_empty_entries(self):
        assert normalize_target_names([" x ", "", "y"]) == {"x", "y"}


class TestGetRowIndex:
    def test_found(self):
        assert get_row_index("b", ["a", "b", "c"]) == 1

    def test_missing_returns_minus_one(self):
        assert get_row_index("z", ["a", "b"]) == -1


class TestGetRowIndexMulti:
    def test_returns_first_present_name(self):
        assert get_row_index_multi(["z", "b"], ["a", "b", "c"]) == 1

    def test_raises_when_none_present(self):
        with pytest.raises(ValueError, match="None of the expected columns"):
            get_row_index_multi(["x", "y"], ["a", "b"])


class TestToProbDist:
    def test_normalizes_to_unit_sum(self):
        result = to_prob_dist([1.0, 3.0])
        assert result.tolist() == [0.25, 0.75]

    def test_first_inf_element_takes_all_mass(self):
        result = to_prob_dist([1.0, np.inf, -np.inf])
        assert result.tolist() == [0.0, 1.0, 0.0]

    def test_all_zero_total_puts_mass_on_first(self):
        result = to_prob_dist([0.0, 0.0, 0.0])
        assert result.tolist() == [1.0, 0.0, 0.0]

    def test_nans_become_zero(self):
        result = to_prob_dist([np.nan, 2.0])
        assert result.tolist() == [0.0, 1.0]

    def test_negative_total_keeps_raw_values(self):
        # total < 0: elements are returned unchanged (Java parity quirk).
        result = to_prob_dist([-1.0, -3.0])
        assert result.tolist() == [-1.0, -3.0]

    def test_does_not_mutate_input(self):
        data = [1.0, 1.0]
        to_prob_dist(data)
        assert data == [1.0, 1.0]


class TestPredictionToConfidences:
    def test_positive_confidences_pass_through(self):
        conf = prediction_to_confidences([0.7, 0.3], "yes", ["yes", "no"])
        np.testing.assert_allclose(conf, [0.7, 0.3])

    def test_missing_confidence_raises(self):
        with pytest.raises(PredictionValidationError, match="missing value"):
            prediction_to_confidences([np.nan, 0.5], "yes", ["yes", "no"])

    def test_all_zero_falls_back_to_string_label(self):
        conf = prediction_to_confidences([0.0, 0.0], "no", ["yes", "no"])
        np.testing.assert_allclose(conf, [0.0, 1.0])

    def test_all_zero_falls_back_to_integer_label(self):
        conf = prediction_to_confidences([0.0, 0.0], 0, ["yes", "no"])
        np.testing.assert_allclose(conf, [1.0, 0.0])

    def test_does_not_mutate_input(self):
        data = [0.0, 0.0]
        prediction_to_confidences(data, "yes", ["yes", "no"])
        assert data == [0.0, 0.0]


class TestClassCountsAndRatios:
    def test_counts_bin_labels(self):
        assert class_counts([0, 1, 1, 2], 4).tolist() == [1, 2, 1, 0]

    def test_ratios_normalize(self):
        np.testing.assert_allclose(class_ratios([0, 1, 1], 2), [1 / 3, 2 / 3])

    def test_ratios_empty_gives_zeros(self):
        assert class_ratios([], 3).tolist() == [0.0, 0.0, 0.0]


class TestTaskXmlHelpers:
    def test_run_output_file_ids_list_form(self):
        run_xml = {
            "oml:output_data": {
                "oml:file": [
                    {"oml:name": "predictions", "oml:file_id": "10"},
                    {"oml:name": "description", "oml:file_id": "11"},
                ]
            }
        }
        assert run_output_file_ids(run_xml) == {
            "predictions": "10",
            "description": "11",
        }

    def test_run_output_file_ids_single_dict_form(self):
        run_xml = {
            "oml:output_data": {
                "oml:file": {"oml:name": "predictions", "oml:file_id": "10"}
            }
        }
        assert run_output_file_ids(run_xml) == {"predictions": "10"}

    def test_run_output_file_ids_empty(self):
        assert run_output_file_ids({}) == {}

    def test_task_source_data(self):
        task_xml = {
            "oml:input": [
                {
                    "@name": "source_data",
                    "oml:data_set": {"oml:data_set_id": "61"},
                },
                {"@name": "estimation_procedure"},
            ]
        }
        assert task_source_data(task_xml) == {"oml:data_set_id": "61"}

    def test_task_source_data_missing_raises(self):
        with pytest.raises(ValueError, match="no 'source_data' input"):
            task_source_data({"oml:input": []})

    def test_task_estimation_procedure_present(self):
        task_xml = {
            "oml:input": {
                "@name": "estimation_procedure",
                "oml:estimation_procedure": {"oml:type": "crossvalidation"},
            }
        }
        assert task_estimation_procedure(task_xml) == {"oml:type": "crossvalidation"}

    def test_task_estimation_procedure_absent(self):
        assert task_estimation_procedure({"oml:input": []}) is None

    def test_task_cost_matrix(self):
        task_xml = {
            "oml:input": {
                "@name": "cost_matrix",
                "oml:cost_matrix": {"#text": "[[0,1],[1,0]]"},
            }
        }
        assert task_cost_matrix(task_xml) == {"#text": "[[0,1],[1,0]]"}

    def test_task_cost_matrix_absent(self):
        assert task_cost_matrix({"oml:input": []}) is None


class TestGetDataAndMetaInformationFromDid:
    def _mock_downloads(self, mocker, metadata):
        mocker.patch(
            "src.helpers.download_and_parse",
            return_value={"oml:data_set_description": metadata},
        )
        return mocker.patch(
            "src.helpers.download_to_temp_file",
            return_value="/tmp/dataset.arff",
        )

    def test_arff_uses_oml_url(self, mocker):
        dl = self._mock_downloads(
            mocker,
            {
                "oml:url": "https://example.org/ds.arff",
                "oml:default_target_attribute": "class",
            },
        )

        info = get_data_and_meta_information_from_did(61)

        assert isinstance(info, DatasetDownloadInfo)
        assert info.file_path == "/tmp/dataset.arff"
        assert info.default_target_attribute == "class"
        dl.assert_called_once_with("https://example.org/ds.arff", suffix=".arff")

    def test_parquet_uses_parquet_url(self, mocker):
        dl = self._mock_downloads(
            mocker,
            {
                "oml:parquet_url": "https://example.org/ds.parquet",
            },
        )

        info = get_data_and_meta_information_from_did(61, dataset_type="parquet")

        assert info.default_target_attribute is None
        dl.assert_called_once_with("https://example.org/ds.parquet", suffix=".parquet")

    def test_unsupported_type_raises(self, mocker):
        bogus_format = "csv"
        with pytest.raises(ValueError, match="Unsupported dataset_type"):
            get_data_and_meta_information_from_did(61, dataset_type=bogus_format)


ARFF_FIXTURE = """\
@relation test

@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute play {yes, no}

@data
sunny,75,yes
overcast,80,no
?,65,yes
"""


class TestLoadArffToDf:
    def test_loads_columns_and_categoricals(self, tmp_path):
        path = tmp_path / "test.arff"
        path.write_text(ARFF_FIXTURE)

        df = load_arff_to_df(str(path))

        assert list(df.columns) == ["outlook", "temperature", "play"]
        assert df["outlook"][:2].tolist() == ["sunny", "overcast"]
        assert bool(df["outlook"][2] is None or pd.isna(df["outlook"][2]))
        assert isinstance(df["outlook"].dtype, pd.CategoricalDtype)
        assert list(df["play"].cat.categories) == ["yes", "no"]
        assert df["temperature"].tolist() == [75.0, 80.0, 65.0]
