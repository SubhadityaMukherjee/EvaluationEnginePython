import pytest

from src.client import (
    PROD_BASE_URL,
    TEST_BASE_URL,
    TEST_DEFAULT_API_KEY,
    OpenmlClient,
)
from src.exceptions import OpenmlApiError, OpenmlConfigError
from src.models import (
    DataFeature,
    DataQuality,
    EvaluationScore,
    Feature,
    Quality,
    RunEvaluation,
)


def ok_response(body: bytes):
    resp = pytest.importorskip("requests").Response()
    resp.status_code = 200
    resp._content = body
    return resp


class TestConstructor:
    def test_test_server_defaults(self):
        client = OpenmlClient(test=True)

        assert client.base_url == TEST_BASE_URL
        assert client.api_key == TEST_DEFAULT_API_KEY

    def test_prod_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENML_API_KEY", raising=False)

        with pytest.raises(OpenmlConfigError, match="API key"):
            OpenmlClient()

    def test_prod_reads_env_key(self, monkeypatch):
        monkeypatch.setenv("OPENML_API_KEY", "env-key")

        client = OpenmlClient()

        assert client.api_key == "env-key"
        assert client.base_url == PROD_BASE_URL

    def test_explicit_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENML_API_KEY", "env-key")

        client = OpenmlClient(api_key="explicit-key")

        assert client.api_key == "explicit-key"

    def test_base_url_gets_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("OPENML_API_KEY", "k")

        client = OpenmlClient(base_url="https://example.org/api/v1")

        assert client.base_url == "https://example.org/api/v1/"

    def test_test_server_default_wins_over_env(self, monkeypatch):
        # documented resolution order: explicit key > test default > env var
        monkeypatch.setenv("OPENML_API_KEY", "my-key")

        client = OpenmlClient(test=True)

        assert client.api_key == TEST_DEFAULT_API_KEY


class TestParse:
    def test_plain_envelope(self):
        parsed = OpenmlClient._parse(
            b"<oml:data_features_upload><oml:did>61</oml:did>"
            b"</oml:data_features_upload>"
        )

        assert parsed["oml:data_features_upload"]["oml:did"] == "61"

    def test_api_error_raises(self):
        body = (
            b"<oml:api_error><oml:code>431</oml:code>"
            b"<oml:message>Dataset already processed</oml:message>"
            b"</oml:api_error>"
        )

        with pytest.raises(OpenmlApiError) as err:
            OpenmlClient._parse(body)

        assert err.value.code == 431
        assert err.value.message == "Dataset already processed"

    def test_api_error_appends_additional_information(self):
        body = (
            b"<oml:api_error><oml:code>431</oml:code>"
            b"<oml:message>already done</oml:message>"
            b"<oml:additional_information>did=61</oml:additional_information>"
            b"</oml:api_error>"
        )

        with pytest.raises(OpenmlApiError) as err:
            OpenmlClient._parse(body)

        assert err.value.message == "already done: did=61"


@pytest.fixture
def client():
    return OpenmlClient(api_key="test-key")


class TestDataGet:
    def test_returns_description(self, client, mocker):
        get = mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(
                b"<oml:data_set_description><oml:id>61</oml:id>"
                b"<oml:url>https://x/ds.arff</oml:url>"
                b"</oml:data_set_description>"
            ),
        )

        desc = client.data_get(61)

        assert desc["oml:url"] == "https://x/ds.arff"
        url = get.call_args.args[0]
        assert url.endswith("data/61")
        assert get.call_args.kwargs["params"] == {"api_key": "test-key"}


class TestUploads:
    def test_data_features_upload_posts_xml_and_returns_did(self, client, mocker):
        post = mocker.patch(
            "src.client.requests.post",
            return_value=ok_response(
                b"<oml:data_features_upload><oml:did>61</oml:did>"
                b"</oml:data_features_upload>"
            ),
        )
        features = DataFeature(
            did=61, features=[Feature(index=0, name="a", data_type="numeric")]
        )

        did = client.data_features_upload(features)

        assert did == 61
        url = post.call_args.args[0]
        assert url.endswith("data/features")
        files = post.call_args.kwargs["files"]
        assert files["api_key"] == (None, "test-key")
        assert files["description"][0] == "description.xml"
        assert b"oml:data_features" in files["description"][1]

    def test_data_qualities_upload_returns_did(self, client, mocker):
        mocker.patch(
            "src.client.requests.post",
            return_value=ok_response(
                b"<oml:data_qualities_upload><oml:did>62</oml:did>"
                b"</oml:data_qualities_upload>"
            ),
        )
        qualities = DataQuality(
            did=62, qualities=[Quality(name="NumberOfInstances", value=1.0)]
        )

        assert client.data_qualities_upload(qualities) == 62

    def test_run_evaluate_upload_returns_run_id(self, client, mocker):
        post = mocker.patch(
            "src.client.requests.post",
            return_value=ok_response(
                b"<oml:run_evaluate><oml:run_id>9</oml:run_id></oml:run_evaluate>"
            ),
        )
        run_eval = RunEvaluation(
            run_id=9,
            scores=[EvaluationScore(function="acc", value=1.0)],
        )

        assert client.run_evaluate_upload(run_eval) == 9
        url = post.call_args.args[0]
        assert url.endswith("run/evaluate")


class TestDataStatusUpdate:
    def test_posts_data_id_and_status(self, client, mocker):
        post = mocker.patch(
            "src.client.requests.post",
            return_value=ok_response(
                b"<oml:data_status_update><oml:data_id>61</oml:data_id>"
                b"</oml:data_status_update>"
            ),
        )

        assert client.data_status_update(61, "active") == 61

        url = post.call_args.args[0]
        assert url.endswith("data/status/update")
        files = post.call_args.kwargs["files"]
        assert files["data_id"] == (None, "61")
        assert files["status"] == (None, "active")
        assert files["api_key"] == (None, "test-key")


class TestDataUnprocessed:
    def test_list_form(self, client, mocker):
        mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(
                b"<oml:data_unprocessed>"
                b"<oml:dataset><oml:did>1</oml:did></oml:dataset>"
                b"<oml:dataset><oml:did>2</oml:did></oml:dataset>"
                b"</oml:data_unprocessed>"
            ),
        )

        assert client.data_unprocessed(1, "normal") == [1, 2]

    def test_single_dataset_form(self, client, mocker):
        mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(
                b"<oml:data_unprocessed>"
                b"<oml:dataset><oml:did>7</oml:did></oml:dataset>"
                b"</oml:data_unprocessed>"
            ),
        )

        assert client.data_unprocessed(1, "normal") == [7]

    def test_empty(self, client, mocker):
        mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(b"<oml:data_unprocessed/>"),
        )

        assert client.data_unprocessed(1, "normal") == []


class TestDataQualitiesUnprocessed:
    def test_posts_joined_qualities(self, client, mocker):
        post = mocker.patch(
            "src.client.requests.post",
            return_value=ok_response(
                b"<oml:data_unprocessed>"
                b"<oml:dataset><oml:did>5</oml:did></oml:dataset>"
                b"</oml:data_unprocessed>"
            ),
        )

        dids = client.data_qualities_unprocessed(
            1, "normal", ["NumberOfInstances", "AutoCorrelation"]
        )

        assert dids == [5]
        url = post.call_args.args[0]
        assert url.endswith("data/qualities/unprocessed/1/normal")
        files = post.call_args.kwargs["files"]
        assert files["qualities"] == (
            None,
            "NumberOfInstances,AutoCorrelation",
        )

    def test_feature_and_priority_segments(self, client, mocker):
        post = mocker.patch(
            "src.client.requests.post",
            return_value=ok_response(b"<oml:data_unprocessed/>"),
        )

        client.data_qualities_unprocessed(
            1, "normal", ["x"], feature_qualities=True, priority_tag="vip"
        )

        url = post.call_args.args[0]
        assert url.endswith("data/qualities/unprocessed/1/normal/feature/vip")


class TestEvaluationRequest:
    def test_returns_run_ids(self, client, mocker):
        mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(
                b"<oml:evaluation_request>"
                b"<oml:run><oml:run_id>10</oml:run_id></oml:run>"
                b"<oml:run><oml:run_id>11</oml:run_id></oml:run>"
                b"</oml:evaluation_request>"
            ),
        )

        assert client.evaluation_request(1, "normal", 1000) == [10, 11]

    def test_filters_appended_to_path(self, client, mocker):
        get = mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(
                b"<oml:evaluation_request>"
                b"<oml:run><oml:run_id>10</oml:run_id></oml:run>"
                b"</oml:evaluation_request>"
            ),
        )

        client.evaluation_request(
            1, "normal", 100, filters={"ttid": "1,2", "tag": "study_1"}
        )

        url = get.call_args.args[0]
        assert url.endswith("evaluation/request/1/normal/100/ttid/1,2/tag/study_1")

    def test_single_run_form(self, client, mocker):
        mocker.patch(
            "src.client.requests.get",
            return_value=ok_response(
                b"<oml:evaluation_request>"
                b"<oml:run><oml:run_id>3</oml:run_id></oml:run>"
                b"</oml:evaluation_request>"
            ),
        )

        assert client.evaluation_request(1, "normal", 1) == [3]
