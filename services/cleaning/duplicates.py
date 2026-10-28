"""Remove repeated rows."""

from __future__ import annotations

import pandas as pd

from services.cleaning.checks import require_columns
from services.cleaning.records import StepRecord


def remove_duplicate_rows(
    df: pd.DataFrame, subset: list[str] | None = None
) -> tuple[pd.DataFrame, StepRecord]:
    """Remove repeated rows, keeping the first occurrence."""
    subset = list(subset or [])
    require_columns(df, subset)

    # An empty subset means "compare every column".
    columns_to_compare = subset or None
    out = df.drop_duplicates(subset=columns_to_compare, keep="first")
    removed = len(df) - len(out)
    if subset:
        scope = f"columns {', '.join(subset)}"
    else:
        scope = "all columns"
    record = StepRecord(
        operation="Remove duplicate rows",
        description=f"Removed {removed} duplicate row(s) (compared on {scope}).",
        columns=subset,
        rows_removed=removed,
    )
    return out, record
