from __future__ import annotations

import numpy as np
import pandas as pd

from src.helpers import get_row_index, get_row_index_multi


class FoldsPredictionCounter:
    def __init__(
        self,
        splits: pd.DataFrame,
        type_name: str = "TEST",
        shadow_type: str = "TRAIN",
    ) -> None:
        cols = list(splits.columns)

        self._col_type = get_row_index("type", cols)
        self._col_rowid = get_row_index_multi(["rowid", "row_id"], cols)
        self._col_repeat = get_row_index_multi(["repeat", "repeat_nr"], cols)
        self._col_fold = get_row_index_multi(["fold", "fold_nr"], cols)
        try:
            self._col_sample = get_row_index_multi(["sample", "sample_nr"], cols)
        except ValueError:
            self._col_sample = -1

        # Dimensions: max + 1, matching Weka's ``attributeStats(...).numericStats.max + 1``
        self._num_repeats = (
            int(splits.iloc[:, self._col_repeat].max()) + 1
            if "repeat" in cols or "repeat_nr" in cols
            else 1
        )
        self._num_folds = (
            int(splits.iloc[:, self._col_fold].max()) + 1
            if "fold" in cols or "fold_nr" in cols
            else 1
        )
        self._num_samples = (
            int(splits.iloc[:, self._col_sample].max()) + 1
            if self._col_sample >= 0
            else 1
        )

        self.expected = [
            [[[] for _ in range(self._num_samples)] for _ in range(self._num_folds)]
            for _ in range(self._num_repeats)
        ]
        self.actual = [
            [[[] for _ in range(self._num_samples)] for _ in range(self._num_folds)]
            for _ in range(self._num_repeats)
        ]
        self.shadow_type_size = np.zeros(
            (self._num_repeats, self._num_folds, self._num_samples), dtype=int
        )
        self.expected_total = 0
        self.error_message = ""

        for _, row in splits.iterrows():
            row_type = row.iloc[self._col_type]
            repeat = int(row.iloc[self._col_repeat])
            fold = int(row.iloc[self._col_fold])
            sample = int(row.iloc[self._col_sample]) if self._col_sample >= 0 else 0
            rowid = int(row.iloc[self._col_rowid])

            if row_type == type_name:
                self.expected[repeat][fold][sample].append(rowid)
                self.expected_total += 1
            elif row_type == shadow_type:
                self.shadow_type_size[repeat][fold][sample] += 1

        for i in range(self._num_repeats):
            for j in range(self._num_folds):
                for k in range(self._num_samples):
                    self.expected[i][j][k].sort()

    def add_prediction(self, repeat: int, fold: int, sample: int, rowid: int) -> None:
        if repeat >= len(self.actual):
            raise RuntimeError(f"Repeat #{repeat} not defined by task.")
        if fold >= len(self.actual[repeat]):
            raise RuntimeError(f"Fold #{fold} not defined by task.")
        self.actual[repeat][fold][sample].append(rowid)

    def check(self) -> bool:
        for i in range(self._num_repeats):
            for j in range(self._num_folds):
                for k in range(self._num_samples):
                    self.actual[i][j][k].sort()
                    if self.actual[i][j][k] != self.expected[i][j][k]:
                        self.error_message = (
                            f"Repeat {i} fold {j} sample {k} expected predictions "
                            f"with row id's {self.expected[i][j][k]}, but got "
                            f"predictions with row id's {self.actual[i][j][k]}"
                        )
                        return False
        return True

    def get_expected_rowids(self, i: int, j: int, k: int) -> list[int]:
        return list(self.expected[i][j][k])

    def get_shadow_type_size(self, i: int, j: int, k: int) -> int:
        return int(self.shadow_type_size[i][j][k])

    def get_error_message(self) -> str:
        return self.error_message

    @property
    def repeats(self) -> int:
        return self._num_repeats

    @property
    def folds(self) -> int:
        return self._num_folds

    @property
    def samples(self) -> int:
        return self._num_samples

    def get_expected_total(self) -> int:
        return self.expected_total
