"""Shared fail-closed column-schema checks."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_unique_column_names(
    columns: Iterable[object],
    *,
    context: str,
) -> None:
    """Reject ambiguous schemas without echoing potentially sensitive names."""

    index = pd.Index(list(columns))
    duplicate_count = int(index.duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"{context} has duplicate column names; count={duplicate_count}"
        )
