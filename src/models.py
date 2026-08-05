from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# Dataset file formats supported throughout the engine. Picked once at the
# edge (CLI flag / notebook) and threaded through download + parse.
DataFormat = Literal["arff", "parquet"]

# ============================================================================
# Dataset metadata
# ============================================================================


@dataclass(slots=True)
class Quality:
    name: str
    value: float | None = None

    def __str__(self) -> str:
        return f"{self.name} - {self.value}"


@dataclass()
class DatasetDownloadInfo:
    file_path: str
    default_target_attribute: str | None


# ============================================================================
# Feature models
# ============================================================================


@dataclass()
class Feature:
    index: int
    name: str
    data_type: str

    nominal_values: list[str] = field(default_factory=list)

    is_target: bool = False
    is_ignore: bool = False
    is_row_identifier: bool = False

    number_of_distinct_values: int | None = None
    number_of_unique_values: int | None = None
    number_of_missing_values: int | None = None
    number_of_integer_values: int | None = None
    number_of_real_values: int | None = None
    number_of_nominal_values: int | None = None
    number_of_values: int | None = None

    maximum_value: float | None = None
    minimum_value: float | None = None
    mean_value: float | None = None
    standard_deviation: float | None = None

    class_distribution: str | None = None

    def __str__(self) -> str:
        return f"{self.index} - {self.name}"


@dataclass()
class DataFeature:
    did: int | None = None
    evaluation_engine_id: int | None = None
    features: list[Feature] = field(default_factory=list)
    error: str | None = None

    def feature_map(
        self,
        *,
        sorted_names: bool = False,
    ) -> dict[str, Feature]:
        result = {f.name: f for f in self.features}
        return dict(sorted(result.items())) if sorted_names else result


@dataclass()
class DataQuality:
    did: int | None = None
    evaluation_engine_id: int | None = None
    qualities: list[Quality] = field(default_factory=list)
    error: str | None = None

    def quality_map(
        self,
        *,
        sorted_names: bool = False,
    ) -> dict[str, Quality]:
        result: dict[str, Quality] = {f.name: f for f in self.qualities}
        return dict(sorted(result.items())) if sorted_names else result


# ============================================================================
# Run evaluation models
# ============================================================================


@dataclass
class EvaluationScore:
    """One computed metric. Mirrors ``org.openml.apiconnector.xml.EvaluationScore``."""

    function: str
    value: float | None = None
    stdev: float | None = None
    array: list | None = None
    repeat: int | None = None
    fold: int | None = None
    sample: int | None = None
    sample_size: int | None = None


@dataclass
class RunEvaluation:
    """Aggregated result of evaluating one run. Mirrors ``RunEvaluation``."""

    run_id: int | None = None
    # Canonical evaluation-engine id; mirrors ``EVALUATION_ENGINE_ID`` in
    # ``src.runs.evaluators`` (kept there to avoid a circular import).
    evaluation_engine_id: int = 1
    scores: list[EvaluationScore] = field(default_factory=list)
    error: str | None = None
    warning: str | None = None

    def add_scores(self, scores: Iterable[EvaluationScore]) -> None:
        self.scores.extend(scores)


# ============================================================================
# Constants
# ============================================================================
# ARFF/Parquet type tags treated as numeric across the engine (feature
# typing, quality computation). Public so the feature / quality extractors
# can check membership without importing a private name.
NUMERIC_TYPES = frozenset({"NUMERIC", "REAL", "INTEGER"})

# ``Feature`` dataclass field groups by their OpenML XML scalar type. Public
# so ``src.features.serialization`` can iterate them without reaching into a
# private name. ``OML_STR_FIELDS`` ("name") is emitted inline by
# ``feature_to_oml_dict`` because the name is mandatory, but kept here to
# document the full schema.
OML_STR_FIELDS = "name"
OML_BOOL_FIELDS = (
    "is_target",
    "is_ignore",
    "is_row_identifier",
)

OML_INT_FIELDS = (
    "number_of_missing_values",
    "number_of_distinct_values",
    "number_of_unique_values",
    "number_of_integer_values",
    "number_of_real_values",
    "number_of_nominal_values",
    "number_of_values",
)

OML_FLOAT_FIELDS = (
    "maximum_value",
    "minimum_value",
    "mean_value",
    "standard_deviation",
)


class EstimationProcedureType(str, Enum):
    CROSSVALIDATION = "CROSSVALIDATION"
    HOLDOUT = "HOLDOUT"
    HOLDOUT_ORDERED = "HOLDOUT_ORDERED"
    LEAVEONEOUT = "LEAVEONEOUT"
    TESTONTRAININGDATA = "TESTONTRAININGDATA"
    LEARNINGCURVE_CV = "LEARNINGCURVE_CV"

    @classmethod
    def from_oml_type(cls, type_str: str | None) -> EstimationProcedureType | None:
        """Map an OpenML task ``oml:type`` string to a procedure type.

        Case-insensitive; returns ``None`` for an unrecognized type. Source:
        ``org.openml.apiconnector.xml.EstimationProcedureType``.
        """
        return _OML_TYPE_TO_PROCEDURE.get((type_str or "").lower())


# OpenML task ``oml:type`` strings (lower-cased) → EstimationProcedureType.
# Single source of truth backing ``EstimationProcedureType.from_oml_type``;
# both the fold generator (``src.process_dataset``) and the run evaluator
# (``src.evaluate_run``) consume it through that classmethod. Note the
# two-string alias onto TESTONTRAININGDATA ("testontrainingdata" + the verbose
# "testonthetrainingdata"). Source:
# org.openml.apiconnector.xml.EstimationProcedureType.
_OML_TYPE_TO_PROCEDURE: dict[str, EstimationProcedureType] = {
    "crossvalidation": EstimationProcedureType.CROSSVALIDATION,
    "holdout": EstimationProcedureType.HOLDOUT,
    "holdout_ordered": EstimationProcedureType.HOLDOUT_ORDERED,
    "leaveoneout": EstimationProcedureType.LEAVEONEOUT,
    "testontrainingdata": EstimationProcedureType.TESTONTRAININGDATA,
    "testonthetrainingdata": EstimationProcedureType.TESTONTRAININGDATA,
    "learningcurve": EstimationProcedureType.LEARNINGCURVE_CV,
}


@dataclass(frozen=True)
class EstimationProcedure:
    """Estimation-procedure configuration read by the Java dispatcher.

    ``folds`` / ``repeats`` / ``percentage`` correspond to the procedure fields
    consumed by ``GenerateFolds.java``; ``percentage`` is a test-set size in the
    range 0..100.
    """

    type: EstimationProcedureType
    folds: int | None = None
    repeats: int | None = None
    percentage: float | None = None
