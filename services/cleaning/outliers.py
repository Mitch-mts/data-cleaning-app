"""Find unusual numeric values with the interquartile range (IQR) rule."""

from __future__ import annotations

from typing import Any

import pandas as pd

from services.cleaning.checks import require_columns, row_number
from services.cleaning.constants import MAX_EXAMPLES
from services.cleaning.records import CleaningError, StepRecord
from services.profiling_service import is_numeric_column, to_native


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float, float, float]:
    """Return ``(q1, q3, lower_bound, upper_bound)`` using the IQR rule.

    Q1 and Q3 are the values at 25% and 75% of the column. The gap between
    them is the IQR. A value is an outlier when it sits more than
    ``multiplier`` IQRs below Q1 or above Q3.
    """
    values = series.dropna().astype("float64")
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return q1, q3, lower_bound, upper_bound


def _outlier_examples(flagged: pd.Series) -> list[dict[str, Any]]:
    """The first few outlier values, with their original row numbers."""
    return [
        {"row": row_number(label), "value": to_native(value)}
        for label, value in flagged.head(MAX_EXAMPLES).items()
    ]


def detect_outliers(
    df: pd.DataFrame, columns: list[str], multiplier: float = 1.5, remove: bool = False
) -> tuple[pd.DataFrame, StepRecord]:
    """Flag IQR outliers in numeric columns. Remove those rows only when ``remove`` is true."""
    if not 0 < multiplier <= 10:
        raise CleaningError("The IQR multiplier must be greater than 0 and at most 10.")
    require_columns(df, columns)

    outliers: dict[str, Any] = {}
    row_is_outlier = pd.Series(False, index=df.index)
    issues = []
    for column in columns:
        series = df[column]
        if not is_numeric_column(series):
            issues.append(f"'{column}' is not numeric; convert it to a number type first.")
            continue
        if series.notna().sum() < 4:
            issues.append(f"'{column}' has fewer than 4 values; outliers were not checked.")
            continue

        q1, q3, lower_bound, upper_bound = iqr_bounds(series, multiplier)
        is_outlier = series.notna() & ((series < lower_bound) | (series > upper_bound))
        row_is_outlier = row_is_outlier | is_outlier
        outliers[column] = {
            "q1": q1,
            "q3": q3,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "count": int(is_outlier.sum()),
            "examples": _outlier_examples(series[is_outlier]),
        }

    total = int(row_is_outlier.sum())
    if remove:
        out = df[~row_is_outlier].copy()
    else:
        out = df.copy()
    removed = len(df) - len(out)
    if remove:
        action = f"removed {removed} row(s)"
    else:
        action = "rows were kept (detection only)"
    record = StepRecord(
        operation="Detect outliers (IQR)",
        description=f"Found {total} row(s) with outliers; {action}.",
        columns=list(outliers),
        rows_removed=removed,
        details={"multiplier": multiplier, "removed": remove, "outliers": outliers},
        issues=issues,
    )
    return out, record
