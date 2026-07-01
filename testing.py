"""Sanity check: compare locally computed ARFF features against the OpenML server."""

from src.features import (
    DataFeature,
    feature_to_oml_dict,
    features_to_xml,
    load_arff_features,
    parse_features_xml,
)
from process_dataset.module import generate_folds
from process_dataset.arff import splits_to_arff, save_splits_arff, arff_head
from models import EstimationProcedureType, EstimationProcedure
from src.helpers import (
    download_and_parse,
    get_data_and_meta_information_from_did,
)
from src.qualities import (
    load_arff_qualities,
    qualities_to_xml,
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

    # ========================================================================
    # Folds: 10-fold CV on did=61 (iris)
    # ========================================================================
    # Stratified 10-fold CV, 1 repeat, seed 1. Expect 10 * N rows where every
    # rowid appears exactly 9 times as TRAIN and once as TEST.
    cv_splits, cv_df, cv_target = generate_folds(
        did=61,
        procedure=EstimationProcedure(
            type=EstimationProcedureType.CROSSVALIDATION,
            folds=10,
            repeats=1,
        ),
        seed=1,
    )
    print(f"\n[CV] dataset: {len(cv_df)} rows, target = {cv_target!r}")
    print(f"[CV] splits : {len(cv_splits)} rows")

    cv_pivot = cv_splits.pivot_table(
        index="rowid",
        columns="type",
        values="fold",
        aggfunc="count",
        fill_value=0,
    )
    print("[CV] TRAIN counts per rowid:", cv_pivot["TRAIN"].value_counts().to_dict())
    print("[CV] TEST  counts per rowid:", cv_pivot["TEST"].value_counts().to_dict())

    cv_arff_text = splits_to_arff(cv_splits, relation="iris_splits")
    save_splits_arff(cv_splits, "iris_cv_splits.arff", relation="iris_splits")
    print("\n[CV] --- ARFF head ---")
    print(arff_head(cv_arff_text, n=12))

    # ========================================================================
    # Folds: 66/33 holdout, 3 repeats
    # ========================================================================
    # Holdout emits the same 4-column schema as CV.
    holdout_splits_df, _, _ = generate_folds(
        did=61,
        procedure=EstimationProcedure(
            type=EstimationProcedureType.HOLDOUT,
            percentage=33,
            repeats=3,
        ),
        seed=1,
    )
    print("\n[HOLDOUT] rows per (repeat, type):")
    print(
        holdout_splits_df.groupby(["repeat", "type"])
        .size()
        .unstack(fill_value=0)
    )

    holdout_arff_text = splits_to_arff(holdout_splits_df, relation="iris_holdout_splits")
    print("\n[HOLDOUT] --- ARFF head ---")
    print(arff_head(holdout_arff_text, n=10))

    # ========================================================================
    # Folds: learning curve (5-fold CV, subsampled training sets)
    # ========================================================================
    # The only generator that emits a `sample` column; training-set size per
    # (fold, sample) should roughly double every two samples.
    lc_splits, _, _ = generate_folds(
        did=61,
        procedure=EstimationProcedure(
            type=EstimationProcedureType.LEARNINGCURVE_CV,
            folds=5,
            repeats=1,
        ),
        seed=1,
    )
    print("\n[LC] training-set size per (fold, sample):")
    print(
        lc_splits[lc_splits.type == "TRAIN"]
        .groupby(["fold", "sample"])
        .size()
        .unstack(fill_value=0)
    )

    lc_arff_text = splits_to_arff(lc_splits, relation="iris_learningcurve_splits")
    save_splits_arff(lc_splits, "iris_lc_splits.arff", relation="iris_learningcurve_splits")
    print("\n[LC] --- ARFF head (note the @ATTRIBUTE sample) ---")
    print(arff_head(lc_arff_text, n=12))
