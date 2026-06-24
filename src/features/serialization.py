import xmltodict

from src.models import (
    DataFeature,
    Feature,
    _OML_BOOL_FIELDS,
    _OML_FLOAT_FIELDS,
    _OML_INT_FIELDS,
)


def feature_to_oml_dict(feat: Feature) -> dict:
    result = {
        "oml:index": str(feat.index),
        "oml:name": feat.name,
        "oml:data_type": feat.data_type,
    }

    if feat.nominal_values:
        result["oml:nominal_value"] = feat.nominal_values

    for attr in _OML_BOOL_FIELDS:
        v = getattr(feat, attr)
        if v is not None:
            result[f"oml:{attr}"] = str(v).lower()

    for attr in (*_OML_INT_FIELDS, *_OML_FLOAT_FIELDS):
        v = getattr(feat, attr)
        if v is not None:
            result[f"oml:{attr}"] = str(v)

    if feat.class_distribution:
        result["oml:class_distribution"] = feat.class_distribution

    return result


def features_to_oml_dict(data_feature: DataFeature) -> dict:
    return {
        "oml:data_features": {
            "@xmlns:oml": "http://openml.org/openml",
            "oml:feature": [feature_to_oml_dict(f) for f in data_feature.features],
        }
    }


def features_to_xml(
    data_feature: DataFeature,
    *,
    pretty: bool = True,
) -> str:
    return xmltodict.unparse(
        features_to_oml_dict(data_feature),
        pretty=pretty,
    )


def parse_features_xml(xml_str: str) -> dict:
    return xmltodict.parse(
        xml_str,
        force_list=("oml:feature", "oml:nominal_value"),
    )
