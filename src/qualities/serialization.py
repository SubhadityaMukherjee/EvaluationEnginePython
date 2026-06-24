import xmltodict

from src.models import (
    DataQuality,
    Quality,
)


def quality_to_oml_dict(qua: Quality) -> dict:
    return {
        "oml:name": qua.name,
        "oml:value": str(qua.value) if qua.value is not None else None,
    }


def qualities_to_oml_dict(data_quality: DataQuality) -> dict:
    return {
        "oml:data_qualities": {
            "@xmlns:oml": "http://openml.org/openml",
            "oml:quality": [
                quality_to_oml_dict(q) for q in data_quality.qualities
            ],
        }
    }


def qualities_to_xml(
    data_quality: DataQuality,
    *,
    pretty: bool = True,
) -> str:
    return xmltodict.unparse(
        qualities_to_oml_dict(data_quality),
        pretty=pretty,
    )


def parse_qualities_xml(xml_str: str) -> dict:
    return xmltodict.parse(
        xml_str,
        force_list=("oml:quality",),
    )
