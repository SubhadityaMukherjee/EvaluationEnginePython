"""Thin REST client for the OpenML API — Python port of the subset of
``org.openml.apiconnector.io.OpenmlConnector`` that ``ProcessDataset`` uses.

Auth: the API key is read from the ``OPENML_API_KEY`` environment variable
(Java's connector holds it as a field on the instance; here we read it in the
constructor so the client is explicit about what it's using).

Endpoints (all under ``base_url``, default ``https://www.openml.org/api/v1/``):
  * ``POST /data/features``        — ``data_features_upload``
  * ``POST /data/qualities``       — ``data_qualities_upload``
  * ``POST /data/status/update``   — ``data_status_update``
  * ``GET  /data/unprocessed/{engine_id}/{mode}`` — ``data_unprocessed``
  * ``POST /run/evaluate``         — ``run_evaluate_upload``
  * ``GET  /evaluation/request/{engine_id}/{mode}/{n}[/{k}/{v}]`` — ``evaluation_request``

Server errors come back as ``<oml:api_error><oml:code>…</oml:code>
<oml:message>…</oml:message>[<oml:additional_information>…]
</oml:api_error>`` and are raised as ``OpenmlApiError(code, message)`` —
mirrors Java's ``ApiException`` so callers can check ``err.code`` against the
OpenML status codes (431 = dataset already processed, 441 = features already
uploaded, 542 = no unprocessed datasets remaining, …).
"""

from __future__ import annotations

import os
from typing import Optional

import requests
import xmltodict

from src.exceptions import OpenmlApiError, OpenmlConfigError
from src.features import features_to_xml
from src.models import DataFeature, DataQuality
from src.qualities import qualities_to_xml
from src.runs import RunEvaluation, run_evaluation_to_xml

PROD_BASE_URL = "https://www.openml.org/api/v1/"
TEST_BASE_URL = "https://test.openml.org/api/v1/"
# Public demo key shipped with the openml test server (see openml-python's
# ``_TEST_SERVERS`` table). Lets ``OpenmlClient(test=True)`` work without
# ``OPENML_API_KEY`` set. Production has no default — callers must supply one.
TEST_DEFAULT_API_KEY = "normaluser"


class OpenmlClient:
    """Stateful REST client — port of the ``OpenmlConnector`` methods used by
    ``ProcessDataset``. Construct with an explicit ``api_key`` or let it fall
    back to ``OPENML_API_KEY``; the key is sent as a multipart form field on
    POSTs (Java: ``entity.addPart("api_key", ...)``) and as ``?api_key=`` on
    GETs (Java: ``doApiGetRequest``)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        test: bool = False,
    ) -> None:
        # Server resolution: explicit base_url wins; otherwise pick by ``test``.
        if base_url is None:
            base_url = TEST_BASE_URL if test else PROD_BASE_URL

        # Key resolution: explicit api_key > test-server default > env var.
        # The test server's "normaluser" demo key lets ``--test`` work with no
        # env setup; production never falls back, so a missing key fails loud.
        resolved = api_key
        if resolved is None and test:
            resolved = TEST_DEFAULT_API_KEY
        if resolved is None:
            resolved = os.environ.get("OPENML_API_KEY")
        if not resolved:
            raise OpenmlConfigError(
                "OpenML API key not set. Pass api_key=... when constructing the "
                "client or set the OPENML_API_KEY environment variable."
            )
        self.api_key = resolved
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"

    # ----------------------------------------------------------------------
    # Low-level request helpers
    # ----------------------------------------------------------------------

    def _post(
        self,
        path: str,
        *,
        files: Optional[dict] = None,
        fields: Optional[dict[str, str]] = None,
    ) -> dict:
        """POST multipart form data and return the parsed XML envelope.

        ``files`` carries file parts (``"description"`` for uploads); ``fields``
        carries string parts (``"data_id"``, ``"status"`` for status update).
        ``api_key`` is always appended as a string part, matching Java's
        ``HttpConnector.doApiPostRequest``."""
        form_files: dict = dict(files or {})
        form_fields: dict[str, str] = dict(fields or {})
        form_fields["api_key"] = self.api_key

        # requests sends multipart/form-data when ``files=`` is set. String
        # parts are represented as ``(None, value)`` tuples so they attach as
        # form fields rather than file uploads.
        for name, value in form_fields.items():
            form_files[name] = (None, value)

        url = self.base_url + path.lstrip("/")
        response = requests.post(url, files=form_files)
        response.raise_for_status()
        return self._parse(response.content)

    def _get_xml(self, path: str) -> dict:
        """GET ``path`` with ``?api_key=`` appended and return parsed XML."""
        url = self.base_url + path.lstrip("/")
        response = requests.get(url, params={"api_key": self.api_key})
        response.raise_for_status()
        return self._parse(response.content)

    @staticmethod
    def _parse(body: bytes) -> dict:
        """Parse an OpenML XML response, raising ``OpenmlApiError`` for
        ``oml:api_error`` envelopes (Java's ``wrapHttpResponse`` fallthrough)."""
        parsed = xmltodict.parse(body)
        if "oml:api_error" in parsed:
            err = parsed["oml:api_error"]
            code = int(err["oml:code"])
            message = err.get("oml:message") or ""
            additional = err.get("oml:additional_information")
            if additional:
                message = f"{message}: {additional}"
            raise OpenmlApiError(code, message)
        return parsed

    # ----------------------------------------------------------------------
    # ProcessDataset-facing methods
    # ----------------------------------------------------------------------

    def data_get(self, did: int) -> dict:
        """Port of ``OpenmlBasicConnector.dataGet`` — GET ``/data/{did}``,
        return the ``oml:data_set_description`` node. Used by MergeDataset to
        fetch the ARFF URL server-side (the description's ``oml:url`` is
        relative to whatever server this client targets)."""
        parsed = self._get_xml(f"data/{did}")
        return parsed["oml:data_set_description"]

    def data_features_upload(self, features: DataFeature) -> int:
        """Port of ``OpenmlConnector.dataFeaturesUpload`` — POST ``/data/features``
        with the serialized XML as the ``description`` file part. Returns the
        ``did`` the server echoes back (``<oml:data_features_upload><oml:did>``)."""
        xml_bytes = features_to_xml(features, pretty=False).encode("utf-8")
        files = {"description": ("description.xml", xml_bytes, "application/xml")}
        parsed = self._post("data/features", files=files)
        return int(parsed["oml:data_features_upload"]["oml:did"])

    def data_qualities_upload(self, qualities: DataQuality) -> int:
        """Port of ``OpenmlConnector.dataQualitiesUpload`` — POST ``/data/qualities``.
        Returns the ``did`` the server echoes back."""
        xml_bytes = qualities_to_xml(qualities, pretty=False).encode("utf-8")
        files = {"description": ("description.xml", xml_bytes, "application/xml")}
        parsed = self._post("data/qualities", files=files)
        return int(parsed["oml:data_qualities_upload"]["oml:did"])

    def data_status_update(self, did: int, status: str) -> int:
        """Port of ``OpenmlBasicConnector.dataStatusUpdate`` — POST
        ``/data/status/update`` with ``data_id`` + ``status`` string parts.
        Returns the ``data_id`` the server echoes back."""
        fields = {"data_id": str(did), "status": status}
        parsed = self._post("data/status/update", fields=fields)
        return int(parsed["oml:data_status_update"]["oml:data_id"])

    def data_unprocessed(self, engine_id: int, mode: str) -> list[int]:
        """Port of ``OpenmlBasicConnector.dataUnprocessed`` — GET
        ``/data/unprocessed/{engine_id}/{mode}``. Returns the list of dataset
        ids needing processing. An empty page comes back as an ``OpenmlApiError``
        whose message contains "No unprocessed" (Java catches the same string
        in ``ProcessDataset.fetchUnprocessed``)."""
        parsed = self._get_xml(f"data/unprocessed/{engine_id}/{mode}")
        root = parsed.get("oml:data_unprocessed") or {}
        datasets = root.get("oml:dataset", [])
        if isinstance(datasets, dict):  # single-dataset edge case
            datasets = [datasets]
        return [int(d["oml:did"]) for d in datasets]

    def data_qualities_unprocessed(
        self,
        engine_id: int,
        mode: str,
        qualities: list[str],
        *,
        feature_qualities: bool = False,
        priority_tag: Optional[str] = None,
    ) -> list[int]:
        """Port of ``OpenmlBasicConnector.dataqualitiesUnprocessed`` — POST
        ``/data/qualities/unprocessed/{engine_id}/{mode}`` with a comma-joined
        ``qualities`` field, optional ``/feature`` segment and optional
        ``/{priority_tag}`` suffix. Returns dataset ids missing any of the
        listed qualities. As with ``data_unprocessed``, an empty page comes
        back as ``OpenmlApiError`` whose message contains "No unprocessed"."""
        path = f"data/qualities/unprocessed/{engine_id}/{mode}"
        if feature_qualities:
            path += "/feature"
        if priority_tag:
            path += f"/{priority_tag}"
        fields = {"qualities": ",".join(qualities)}
        parsed = self._post(path, fields=fields)
        root = parsed.get("oml:data_unprocessed") or {}
        datasets = root.get("oml:dataset", [])
        if isinstance(datasets, dict):  # single-dataset edge case
            datasets = [datasets]
        return [int(d["oml:did"]) for d in datasets]

    # ----------------------------------------------------------------------
    # EvaluateRun-facing methods
    # ----------------------------------------------------------------------

    def run_evaluate_upload(self, run_eval: RunEvaluation) -> int:
        """Port of ``OpenmlConnector.runEvaluate`` — POST ``/run/evaluate``
        with the serialized XML as the ``description`` file part. Returns the
        ``run_id`` the server echoes back
        (``<oml:run_evaluate><oml:run_id>``)."""
        xml_bytes = run_evaluation_to_xml(run_eval, pretty=False).encode("utf-8")
        files = {"description": ("description.xml", xml_bytes, "application/xml")}
        parsed = self._post("run/evaluate", files=files)
        return int(parsed["oml:run_evaluate"]["oml:run_id"])

    def evaluation_request(
        self,
        engine_id: int,
        mode: str,
        num_requests: int,
        filters: Optional[dict[str, str]] = None,
    ) -> list[int]:
        """Port of ``OpenmlBasicConnector.evaluationRequest`` — GET
        ``/evaluation/request/{engine_id}/{mode}/{num_requests}`` followed by
        ``/{key}/{value}`` for each filter. Returns the list of run ids to
        evaluate.

        When no runs remain the server returns API error 1013
        (``NO_UNEVALUATED_RUNS`` — see ``ApiErrorMapping``), which surfaces here
        as ``OpenmlApiError``; callers should catch it to stop the loop."""
        path = f"evaluation/request/{engine_id}/{mode}/{num_requests}"
        if filters:
            for key, value in filters.items():
                path += f"/{key}/{value}"
        parsed = self._get_xml(path)
        root = parsed.get("oml:evaluation_request", {})
        runs = root.get("oml:run", [])
        if isinstance(runs, dict):  # single-run edge case
            runs = [runs]
        return [int(r["oml:run_id"]) for r in runs]
