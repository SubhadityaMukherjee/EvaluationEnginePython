from __future__ import annotations

import arff
import pandas as pd


def splits_to_arff(splits: pd.DataFrame, relation: str = "splits") -> str:
    """Serialize a splits DataFrame to an OpenML-format ARFF string.

    Java's ``ArffMapping`` emits a 4- or 5-column ARFF (``type``, ``rowid``,
    ``repeat``, ``fold``, optional ``sample``). We reproduce that schema with
    ``liac-arff`` so the output is byte-compatible with what the OpenML server
    accepts. ``type`` is nominal ``{TRAIN, TEST}``; every other column is
    ``NUMERIC``. Columns are emitted in declaration order regardless of
    DataFrame column order.
    """
    attributes: list[tuple[str, object]] = [("type", ["TRAIN", "TEST"])]
    for col in ("rowid", "repeat", "fold", "sample"):
        if col in splits.columns:
            attributes.append((col, "NUMERIC"))

    ordered = splits[[name for name, _ in attributes]]
    data = ordered.astype(object).to_numpy().tolist()

    return arff.dumps(
        {
            "relation": relation,
            "attributes": attributes,
            "data": data,
        }
    )


def save_splits_arff(
    splits: pd.DataFrame,
    path: str,
    relation: str = "splits",
) -> None:
    """Write a splits DataFrame to ``path`` as OpenML-format ARFF."""
    text = splits_to_arff(splits, relation=relation)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def arff_head(text: str, n: int = 15) -> str:
    """First ``n`` lines of an ARFF string — handy for quick inspection."""
    return "\n".join(text.splitlines()[:n])
