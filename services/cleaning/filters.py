"""Keep rows that match a rule, and sort the rows that remain."""

from __future__ import annotations

import pandas as pd

from services.cleaning.checks import require_columns
from services.cleaning.records import CleaningError, StepRecord
from services.profiling_service import is_numeric_column


def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[pd.DataFrame, StepRecord]:
    """Keep rows whose ``column`` value lies within ``[minimum, maximum]``.

    Rows with a missing value in ``column`` are removed as well.
    """
    require_columns(df, [column])
    if minimum is None and maximum is None:
        raise CleaningError("Enter a minimum, a maximum or both.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CleaningError("The minimum must not be greater than the maximum.")

    series = df[column]
    if not is_numeric_column(series):
        raise CleaningError(f"'{column}' is not numeric; convert it to a number type first.")

    keep = series.notna()
    if minimum is not None:
        keep = keep & (series >= minimum)
    if maximum is not None:
        keep = keep & (series <= maximum)

    out = df[keep].copy()
    bounds = []
    if minimum is not None:
        bounds.append(f">= {minimum:g}")
    if maximum is not None:
        bounds.append(f"<= {maximum:g}")
    removed = len(df) - len(out)
    record = StepRecord(
        operation="Filter by numeric range",
        description=(
            f"Kept rows where '{column}' {' and '.join(bounds)}; removed {removed} row(s)."
        ),
        columns=[column],
        rows_removed=removed,
        details={"minimum": minimum, "maximum": maximum},
    )
    return out, record


def filter_text_equals(df: pd.DataFrame, column: str, value: str) -> tuple[pd.DataFrame, StepRecord]:
    """Keep rows whose ``column`` value (as text) exactly equals ``value``."""
    require_columns(df, [column])
    series = df[column]
    keep = series.notna() & (series.astype(str) == value)
    out = df[keep].copy()
    removed = len(df) - len(out)
    record = StepRecord(
        operation="Filter by exact text match",
        description=f"Kept rows where '{column}' equals '{value}'; removed {removed} row(s).",
        columns=[column],
        rows_removed=removed,
        details={"value": value},
    )
    return out, record


def sort_rows(
    df: pd.DataFrame, columns: list[str], ascending: bool = True
) -> tuple[pd.DataFrame, StepRecord]:
    """Sort rows by ``columns``. Missing values are placed last."""
    require_columns(df, columns)
    if not columns:
        raise CleaningError("Choose at least one column to sort by.")
    try:
        out = df.sort_values(by=columns, ascending=ascending, na_position="last", kind="mergesort")
    except TypeError as error:
        raise CleaningError(
            "a sort column mixes numbers and text; convert its type first"
        ) from error

    if ascending:
        direction = "ascending"
    else:
        direction = "descending"
    record = StepRecord(
        operation="Sort rows",
        description=f"Sorted by {', '.join(columns)} ({direction}).",
        columns=list(columns),
        details={"ascending": ascending},
    )
    return out, record
