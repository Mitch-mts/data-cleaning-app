"""Drop or fill missing values."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from services.cleaning.checks import require_columns
from services.cleaning.records import CleaningError, StepRecord
from services.profiling_service import is_numeric_column, is_text_column, to_native


def missing_value_counts(df: pd.DataFrame) -> dict[str, int]:
    """Missing values per column."""
    return {str(name): int(count) for name, count in df.isna().sum().items()}


def drop_columns_over_threshold(
    df: pd.DataFrame, columns: list[str], threshold_pct: float
) -> tuple[pd.DataFrame, StepRecord]:
    """Drop columns (among ``columns``) whose missing share is above ``threshold_pct``."""
    if not 0 <= threshold_pct <= 100:
        raise CleaningError("The missing-value threshold must be between 0 and 100.")
    require_columns(df, columns)

    total_rows = len(df)
    shares = {}
    for column in columns:
        missing_count = float(df[column].isna().sum())
        shares[column] = (missing_count / total_rows * 100) if total_rows else 0.0

    to_drop = [column for column, share in shares.items() if share > threshold_pct]
    record = StepRecord(
        operation="Remove columns with many missing values",
        description=(
            f"Removed {len(to_drop)} column(s) with more than {threshold_pct:g}% missing values."
        ),
        columns=to_drop,
        columns_removed=to_drop,
        details={"missing_pct": {column: round(shares[column], 1) for column in to_drop}},
    )
    return df.drop(columns=to_drop), record


def fill_numeric_missing(
    df: pd.DataFrame, columns: list[str], strategy: str
) -> tuple[pd.DataFrame, StepRecord]:
    """Fill missing values in numeric columns with the column mean or median."""
    if strategy not in ("mean", "median"):
        raise CleaningError(f"Unknown numeric fill strategy: {strategy}")
    require_columns(df, columns)

    out = df.copy()
    filled: dict[str, int] = {}
    values: dict[str, Any] = {}
    issues = []
    for column in columns:
        series = out[column]
        if not is_numeric_column(series):
            continue

        missing = int(series.isna().sum())
        if not missing:
            continue

        fill_value = series.mean() if strategy == "mean" else series.median()
        if pd.isna(fill_value):
            issues.append(f"'{column}' has no values to compute a {strategy} from.")
            continue

        # A whole-number column cannot store 1.5. Widen it to decimals first.
        if ptypes.is_integer_dtype(series) and float(fill_value) != int(fill_value):
            series = series.astype("float64")
            issues.append(f"'{column}' was changed to decimal so the {strategy} fits.")

        out[column] = series.fillna(fill_value)
        filled[column] = missing
        values[column] = to_native(fill_value)

    record = StepRecord(
        operation=f"Fill missing numbers ({strategy})",
        description=f"Filled {sum(filled.values())} missing numeric value(s) with the {strategy}.",
        columns=list(filled),
        missing_filled=sum(filled.values()),
        details={"filled_per_column": filled, "fill_values": values},
        issues=issues,
    )
    return out, record


def fill_text_missing(
    df: pd.DataFrame, columns: list[str], strategy: str, value: str = ""
) -> tuple[pd.DataFrame, StepRecord]:
    """Fill missing values in text columns with a fixed value or the most frequent value."""
    if strategy not in ("value", "mode"):
        raise CleaningError(f"Unknown text fill strategy: {strategy}")
    if strategy == "value" and not value:
        raise CleaningError("Please provide the text to fill missing values with.")
    require_columns(df, columns)

    out = df.copy()
    filled: dict[str, int] = {}
    values: dict[str, Any] = {}
    issues = []
    for column in columns:
        series = out[column]
        if not is_text_column(series):
            issues.append(f"'{column}' is not a text column and was skipped.")
            continue

        missing = int(series.isna().sum())
        if not missing:
            continue

        if strategy == "mode":
            most_common = series.mode(dropna=True)
            if most_common.empty:
                issues.append(f"'{column}' has no values to find the most frequent from.")
                continue
            fill_value = most_common.iloc[0]
        else:
            fill_value = value

        out[column] = series.fillna(fill_value)
        filled[column] = missing
        values[column] = to_native(fill_value)

    if strategy == "mode":
        label = "most frequent value"
    else:
        label = f"'{value}'"
    record = StepRecord(
        operation="Fill missing text",
        description=f"Filled {sum(filled.values())} missing text value(s) with the {label}.",
        columns=list(filled),
        missing_filled=sum(filled.values()),
        details={"filled_per_column": filled, "fill_values": values},
        issues=issues,
    )
    return out, record


def drop_rows_with_missing(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, StepRecord]:
    """Remove rows that still have a missing value in any of ``columns``."""
    require_columns(df, columns)
    if columns:
        out = df.dropna(subset=columns)
    else:
        out = df.copy()
    removed = len(df) - len(out)
    record = StepRecord(
        operation="Remove rows with missing values",
        description=f"Removed {removed} row(s) that had missing values in the selected columns.",
        columns=list(columns),
        rows_removed=removed,
    )
    return out, record
