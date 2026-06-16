import xmltodict
import requests
from tempfile import NamedTemporaryFile
from typing import Literal
from src.features.models import DatasetDownloadInfo


def download_and_parse(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return xmltodict.parse(response.content)


def download_to_temp_file(
    url: str,
    suffix: str = "",
    chunk_size: int = 8192,
) -> str:
    """
    Download a URL to a temporary file.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:
            for chunk in response.iter_content(chunk_size=chunk_size):
                tmp.write(chunk)

    return tmp.name


# ============================================================================
# Dataset retrieval
# ============================================================================


def get_data_and_meta_information_from_did(
    did: int,
    dataset_type: Literal["arff", "parquet"] = "arff",
) -> DatasetDownloadInfo:
    dataset_type = dataset_type.lower()

    if dataset_type not in {"arff", "parquet"}:
        raise ValueError("dataset_type must be 'arff' or 'parquet'")

    metadata = download_and_parse(f"https://www.openml.org/api/v1/xml/data/{did}")[
        "oml:data_set_description"
    ]

    url_key = "oml:url" if dataset_type == "arff" else "oml:parquet_url"

    return DatasetDownloadInfo(
        file_path=download_to_temp_file(
            metadata[url_key],
            suffix=f".{dataset_type}",
        ),
        default_target_attribute=metadata.get("oml:default_target_attribute"),
    )
