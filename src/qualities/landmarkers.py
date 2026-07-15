"""sklearn port of Weka's ``GenericLandmarker`` characterizers.

Each landmarker runs N-fold cross-validation of a classifier on the dataset
and reports three meta-features — ``{name}AUC``, ``{name}ErrRate``,
``{name}Kappa`` — mirroring
``org.openml.webapplication.fantail.dc.landmarking.GenericLandmarker``.

Faithfulness gaps (Weka classifiers without sklearn equivalents):
  * ``J48.*`` — Weka's C4.5 with confidence-based pruning (``-C``). sklearn's
    ``DecisionTreeClassifier`` is CART with no equivalent pruning flag, so all
    three J48 variants below use the same plain tree and **emit identical
    values**. The Java metric IDs are preserved so the server schema matches.
  * ``REPTree*`` / ``RandomTree*`` — approximated via ``DecisionTreeClassifier``
    with the matching ``max_depth``. Different split logic, so values diverge
    from Java.
  * ``CfsSubsetEval_*`` — **SKIPPED**. CFS (Correlation-based Feature
    Selection) is a Weka-specific subset evaluator with no sklearn equivalent;
    faking one would misrepresent the meta-feature. ``ALL_LANDMARKERS`` omits
    the three CFS entries that Java's ``CharacterizerFactory.all()`` includes.

The CV harness itself is faithful: Weka's
``Evaluation.crossValidateModel(cls, data, 2, new Random(1))`` randomizes,
stratifies (nominal target), and accumulates predictions across folds before
computing ``weightedAreaUnderROC()`` / ``errorRate()`` / ``kappa()``. Here,
``StratifiedKFold(2, shuffle=True, random_state=1)`` + pooled
``cross_val_predict`` plays the same role. Numeric targets return all-None
(Java parity via ``UnassignedClassException``).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# CharacterizerFactory.all() hardcodes 2 folds; GenericLandmarker.characterize
# hardcodes new Random(1).
_NUM_FOLDS = 2
_SEED = 1

# GenericLandmarker.measures — emitted in this order, suffixed to each name.
_MEASURES = ("AUC", "ErrRate", "Kappa")


def _cross_validated_metrics(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    num_folds: int = _NUM_FOLDS,
    seed: int = _SEED,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Run stratified ``num_folds`` CV, return ``(auc, err_rate, kappa)``
    pooled across folds — mirrors Weka's ``Evaluation`` accumulation.

    Returns ``(None, None, None)`` if anything is undefined: single-class
    target, AUC failure, or any sklearn exception. Java's GenericLandmarker
    returns null under the same conditions (``UnassignedClassException`` or
    the catch-all ``Exception``)."""
    classes = np.unique(y)
    if len(classes) < 2:
        # Need at least 2 classes for kappa/AUC to be meaningful.
        return None, None, None

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    try:
        # One CV pass producing probabilities; hard predictions derived via
        # argmax (matches Weka training one model per fold and predicting).
        y_proba = cross_val_predict(estimator, X, y, cv=skf, method="predict_proba")
    except (Exception, NotFittedError):  # noqa: BLE001 — Java's catch-all parity
        return None, None, None

    y_pred = classes.take(np.argmax(y_proba, axis=1))
    err_rate = 1.0 - accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    auc: Optional[float]
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=classes,
            )
    except (ValueError, IndexError):  # undefined AUC — Java returns null
        auc = None

    return auc, err_rate, kappa


def generic_landmarker(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Optional[float]]:
    """Compute one landmarker's three meta-features. Returns ``{name}AUC``,
    ``{name}ErrRate``, ``{name}Kappa`` — matching ``GenericLandmarker.getIDs``."""
    auc, err, kappa = _cross_validated_metrics(estimator, X, y)
    return {
        f"{name}AUC": auc,
        f"{name}ErrRate": err,
        f"{name}Kappa": kappa,
    }


# (landmarker_name, estimator_factory). Mirrors CharacterizerFactory.all()
# minus the three CfsSubsetEval_* entries (CFS has no sklearn equivalent).
# J48 variants use identical estimators (see module docstring).
ALL_LANDMARKERS: list[tuple[str, Callable[[], object]]] = [
    ("kNN1N", lambda: KNeighborsClassifier(n_neighbors=1)),
    ("NaiveBayes", lambda: GaussianNB()),
    ("DecisionStump", lambda: DecisionTreeClassifier(max_depth=1)),
    ("J48.001.", lambda: DecisionTreeClassifier()),
    ("J48.0001.", lambda: DecisionTreeClassifier()),
    ("J48.00001.", lambda: DecisionTreeClassifier()),
    ("REPTreeDepth1", lambda: DecisionTreeClassifier(max_depth=1)),
    ("REPTreeDepth2", lambda: DecisionTreeClassifier(max_depth=2)),
    ("REPTreeDepth3", lambda: DecisionTreeClassifier(max_depth=3)),
    ("RandomTreeDepth1", lambda: DecisionTreeClassifier(max_depth=1)),
    ("RandomTreeDepth2", lambda: DecisionTreeClassifier(max_depth=2)),
    ("RandomTreeDepth3", lambda: DecisionTreeClassifier(max_depth=3)),
]


def expected_landmarker_ids() -> list[str]:
    """All metric IDs the ALL set emits — port of
    ``CharacterizerFactory.getExpectedQualities(all(null))`` (minus CFS)."""
    return [f"{name}{suffix}" for name, _ in ALL_LANDMARKERS for suffix in _MEASURES]


def compute_all_landmarkers(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Optional[float]]:
    """Run every landmarker in ``ALL_LANDMARKERS`` and merge results into one
    dict. X must be a clean numeric matrix; the caller (``ExtractFeatures``)
    is responsible for encoding via the same path pymfe uses."""
    result: dict[str, Optional[float]] = {}
    for name, factory in ALL_LANDMARKERS:
        result.update(generic_landmarker(name, factory(), X, y))
    return result
