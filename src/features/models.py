from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# Dataset metadata
# ============================================================================


@dataclass()
class DatasetDownloadInfo:
    file_path: str
    default_target_attribute: Optional[str]


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

    number_of_distinct_values: Optional[int] = None
    number_of_unique_values: Optional[int] = None
    number_of_missing_values: Optional[int] = None
    number_of_integer_values: Optional[int] = None
    number_of_real_values: Optional[int] = None
    number_of_nominal_values: Optional[int] = None
    number_of_values: Optional[int] = None

    maximum_value: Optional[float] = None
    minimum_value: Optional[float] = None
    mean_value: Optional[float] = None
    standard_deviation: Optional[float] = None

    class_distribution: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.index} - {self.name}"


@dataclass()
class DataFeature:
    did: Optional[int] = None
    evaluation_engine_id: Optional[int] = None
    features: list[Feature] = field(default_factory=list)
    error: Optional[str] = None

    def feature_map(
        self,
        *,
        sorted_names: bool = False,
    ) -> dict[str, Feature]:
        result = {f.name: f for f in self.features}
        return dict(sorted(result.items())) if sorted_names else result


# ============================================================================
# Constants
# ============================================================================

_NUMERIC_TYPES = frozenset({"NUMERIC", "REAL", "INTEGER"})


_OML_BOOL_FIELDS = (
    "is_target",
    "is_ignore",
    "is_row_identifier",
)

_OML_INT_FIELDS = (
    "number_of_missing_values",
    "number_of_distinct_values",
    "number_of_unique_values",
    "number_of_integer_values",
    "number_of_real_values",
    "number_of_nominal_values",
    "number_of_values",
)

_OML_FLOAT_FIELDS = (
    "maximum_value",
    "minimum_value",
    "mean_value",
    "standard_deviation",
)
