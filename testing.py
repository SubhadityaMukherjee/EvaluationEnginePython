"""Sanity check: compare locally computed ARFF features against the OpenML server."""

from src.features import (
    DataFeature,
    feature_to_oml_dict,
    features_to_xml,
    load_arff_features,
    load_arff_qualities,
    parse_features_xml,
    qualities_to_xml,
)
from src.helpers import (
    download_and_parse,
    get_data_and_meta_information_from_did,
)

# ============================================================================
# Computing features
# ============================================================================


# Server features - only for sanity check
features_url = lambda data_id: f"https://www.openml.org/api/v1/data/features/{data_id}"
server_features = lambda data_id: download_and_parse(features_url(data_id))

# Server qualities - names differ from local (Weka vs pymfe), so only eyeball-comparable
qualities_url = (
    lambda data_id: f"https://www.openml.org/api/v1/data/qualities/{data_id}"
)
server_qualities = lambda data_id: download_and_parse(qualities_url(data_id))


# Get diffs between versions
def get_diff_between_server_and_local_for_did(
    server: dict,
    local: DataFeature,
) -> None:
    server_features = server["oml:data_features"]["oml:feature"]

    local_features = {f.index: feature_to_oml_dict(f) for f in local.features}

    for s in server_features:
        idx = int(s["oml:index"])
        l = local_features.get(idx)

        if l is None:
            print(f"feature {idx}: missing locally")
            continue

        if s == l:
            continue

        print(f"feature {idx}:")

        for k in sorted(set(s) | set(l)):
            # if s.get(k) != l.get(k):
            print(f"  {k}: server={s.get(k)!r}  local={l.get(k)!r}")


if __name__ == "__main__":
    data_id = 47246

    arff_local_features = load_arff_features(
        get_data_and_meta_information_from_did(data_id),
        did=data_id,
    )

    xml_local_features = features_to_xml(arff_local_features)
    local_features = parse_features_xml(xml_local_features)
    print(xml_local_features)

    download_info = get_data_and_meta_information_from_did(did=47246)

    get_diff_between_server_and_local_for_did(
        server=server_features(data_id=data_id),
        local=arff_local_features,
    )

    # Qualities: server uses Weka names, local uses pymfe names - not directly comparable.
    qual_data_id = 200
    qual_download_info = get_data_and_meta_information_from_did(qual_data_id)
    local_qualities = load_arff_qualities(qual_download_info, did=qual_data_id)
    print("SERVER QUALITIES:", server_qualities(qual_data_id))
    print("LOCAL QUALITIES:", qualities_to_xml(local_qualities))
