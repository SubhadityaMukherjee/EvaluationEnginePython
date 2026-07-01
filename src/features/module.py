from collections import Counter
from dataclasses import dataclass

import numpy as np

from models import Feature


def _fill_numeric_feature(col, feat: Feature) -> None:
    feat.number_of_values = len(col)

    present = np.asarray(
        [v for v in col if v is not None],
        dtype=np.float64,
    )

    feat.number_of_missing_values = len(col) - present.size

    if present.size == 0:
        return

    int_mask = present == np.floor(present)

    feat.number_of_integer_values = int(int_mask.sum())
    feat.number_of_real_values = int((~int_mask).sum())

    feat.maximum_value = float(np.max(present))
    feat.minimum_value = float(np.min(present))
    feat.mean_value = float(np.mean(present))
    feat.standard_deviation = float(np.std(present, ddof=0))

    unique_vals, counts = np.unique(
        present,
        return_counts=True,
    )

    feat.number_of_distinct_values = len(unique_vals)
    feat.number_of_unique_values = int(np.sum(counts == 1))


def _fill_nominal_feature(col, feat: Feature, schema_values) -> None:
    feat.nominal_values = sorted(schema_values)
    feat.number_of_values = len(col)

    counts = Counter(v for v in col if v is not None)

    present = sum(counts.values())

    feat.number_of_missing_values = len(col) - present
    feat.number_of_nominal_values = present
    feat.number_of_distinct_values = len(counts)
    feat.number_of_unique_values = sum(c == 1 for c in counts.values())

    feat.class_distribution = ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))
