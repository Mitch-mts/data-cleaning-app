"""Reusable data cleaning operations and the pipeline that combines them.

Every operation takes a DataFrame and returns ``(new_df, StepRecord)``; the
input DataFrame is never modified. Operations raise ``CleaningError`` when they
cannot be applied, and the pipeline reports those errors instead of crashing.
"""

from __future__ import annotations

import operator
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from services.profiling_service import (
    is_datetime_column,
    is_numeric_column,
    is_text_column,
    to_native,
)

if TYPE_CHECKING:
    from services.options_service import CleaningOptions

TEXT_CASES = ("none", "lower", "upper", "title")
NUMERIC_FILL_STRATEGIES = ("none", "mean", "median")
TEXT_FILL_STRATEGIES = ("none", "value", "mode")
MISSING_ROW_ACTIONS = ("keep", "drop")
TARGET_TYPES = ("keep", "integer", "float", "string", "datetime")
CONVERSION_ERROR_ACTIONS = ("coerce", "skip")
SORT_DIRECTIONS = ("asc", "desc")
TRANSFORM_OPERATIONS: dict[str, Any] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}
MAX_EXAMPLES = 10


class CleaningError(ValueError):
    """Raised when a cleaning operation cannot be performed."""


@dataclass
class StepRecord:
    """What a cleaning step did, used to build the cleaning report."""

    operation: str
    description: str
    columns: list[str] = field(default_factory=list)
    rows_removed: int = 0
    columns_removed: list[str] = field(default_factory=list)
    missing_filled: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)


@dataclass
class PipelineResult:
    """Output of :func:`run_cleaning_pipeline`."""

    df: pd.DataFrame
    steps: list[StepRecord]
    errors: list[str]
    warnings: list[str]


def row_number(label: Any) -> Any:
    """1-based row number in the original data for an index label."""
    return int(label) + 1 if isinstance(label, (int, np.integer)) else str(label)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError(
            "Column(s) not found (they may have been removed by an earlier step): "
            + ", ".join(missing)
        )


# --------------------------------------------------------------------------- #
# Column names
# --------------------------------------------------------------------------- #
def make_unique_names(names: list[str]) -> list[str]:
    """Append ``_2``, ``_3``… to repeated names so every name is unique."""
    used: set[str] = set()
    result = []
    for name in names:
        candidate, counter = name, 2
        while candidate in used:
            candidate = f"{name}_{counter}"
            counter += 1
        used.add(candidate)
        result.append(candidate)
    return result


def standardize_column_names(
    df: pd.DataFrame,
    strip: bool = True,
    lowercase: bool = True,
    underscores: bool = True,
) -> tuple[pd.DataFrame, StepRecord]:
    """Trim, lower-case and underscore column names, resolving duplicates safely."""
    proposed = []
    for position, name in enumerate(df.columns, start=1):
        new = str(name)
        if strip:
            new = new.strip()
        if lowercase:
            new = new.lower()
        if underscores:
            new = re.sub(r"\s+", "_", new.strip() if strip else new)
        proposed.append(new or f"column_{position}")

    unique = make_unique_names(proposed)
    renamed = {str(old): new for old, new in zip(df.columns, unique) if str(old) != new}
    duplicates_resolved = [n for n, p in zip(unique, proposed) if n != p]

    out = df.copy()
    out.columns = unique
    record = StepRecord(
        operation="Standardize column names",
        description=f"Renamed {len(renamed)} column(s).",
        columns=list(renamed),
        details={"renamed": renamed, "duplicates_resolved": duplicates_resolved},
    )
    return out, record


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def clean_text_columns(
    df: pd.DataFrame,
    columns: list[str],
    trim: bool = True,
    case: str = "none",
    empty_to_missing: bool = True,
) -> tuple[pd.DataFrame, StepRecord]:
    """Trim whitespace, change capitalisation and blank-to-missing in text columns."""
    if case not in TEXT_CASES:
        raise CleaningError(f"Unknown capitalisation option: {case}")
    _require_columns(df, columns)

    out = df.copy()
    changed: dict[str, int] = {}
    blanks: dict[str, int] = {}
    issues = []
    for column in columns:
        series = out[column]
        if not is_text_column(series):
            issues.append(f"'{column}' is not a text column and was skipped.")
            continue
        is_str = series.map(lambda v: isinstance(v, str))
        original = series[is_str]
        new = original
        if trim:
            new = new.str.strip()
        if case != "none":
            new = getattr(new.str, case)()
        blank_count = 0
        if empty_to_missing:
            blank_mask = new.str.strip() == ""
            blank_count = int(blank_mask.sum())
            new = new.mask(blank_mask, np.nan)
        updated = series.copy()
        updated[is_str] = new
        out[column] = updated
        changed[column] = int(original.ne(new).sum())
        blanks[column] = blank_count

    actions = [name for flag, name in (
        (trim, "trimmed whitespace"),
        (case != "none", f"converted to {case} case"),
        (empty_to_missing, "turned blank values into missing values"),
    ) if flag]
    record = StepRecord(
        operation="Clean text",
        description=(", ".join(actions) or "no text changes selected").capitalize()
        + f"; {sum(changed.values())} cell(s) changed.",
        columns=list(changed),
        details={"cells_changed": changed, "blanks_to_missing": blanks},
        issues=issues,
    )
    return out, record


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def missing_value_counts(df: pd.DataFrame) -> dict[str, int]:
    """Missing values per column."""
    return {str(k): int(v) for k, v in df.isna().sum().items()}


def drop_columns_over_threshold(
    df: pd.DataFrame, columns: list[str], threshold_pct: float
) -> tuple[pd.DataFrame, StepRecord]:
    """Drop columns (among ``columns``) whose missing share is above ``threshold_pct``."""
    if not 0 <= threshold_pct <= 100:
        raise CleaningError("The missing-value threshold must be between 0 and 100.")
    _require_columns(df, columns)
    total = len(df)
    shares = {
        c: (float(df[c].isna().sum()) / total * 100 if total else 0.0) for c in columns
    }
    to_drop = [c for c, share in shares.items() if share > threshold_pct]
    record = StepRecord(
        operation="Remove columns with many missing values",
        description=(
            f"Removed {len(to_drop)} column(s) with more than {threshold_pct:g}% missing values."
        ),
        columns=to_drop,
        columns_removed=to_drop,
        details={"missing_pct": {c: round(shares[c], 1) for c in to_drop}},
    )
    return df.drop(columns=to_drop), record


def fill_numeric_missing(
    df: pd.DataFrame, columns: list[str], strategy: str
) -> tuple[pd.DataFrame, StepRecord]:
    """Fill missing values in numeric columns with the column mean or median."""
    if strategy not in ("mean", "median"):
        raise CleaningError(f"Unknown numeric fill strategy: {strategy}")
    _require_columns(df, columns)
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
    _require_columns(df, columns)
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
            modes = series.mode(dropna=True)
            if modes.empty:
                issues.append(f"'{column}' has no values to find the most frequent from.")
                continue
            fill_value = modes.iloc[0]
        else:
            fill_value = value
        out[column] = series.fillna(fill_value)
        filled[column] = missing
        values[column] = to_native(fill_value)
    label = "most frequent value" if strategy == "mode" else f"'{value}'"
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
    _require_columns(df, columns)
    out = df.dropna(subset=columns) if columns else df.copy()
    removed = len(df) - len(out)
    record = StepRecord(
        operation="Remove rows with missing values",
        description=f"Removed {removed} row(s) that had missing values in the selected columns.",
        columns=list(columns),
        rows_removed=removed,
    )
    return out, record


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def remove_duplicate_rows(
    df: pd.DataFrame, subset: list[str] | None = None
) -> tuple[pd.DataFrame, StepRecord]:
    """Remove repeated rows, keeping the first occurrence."""
    subset = list(subset or [])
    _require_columns(df, subset)
    out = df.drop_duplicates(subset=subset or None, keep="first")
    removed = len(df) - len(out)
    scope = f"columns {', '.join(subset)}" if subset else "all columns"
    record = StepRecord(
        operation="Remove duplicate rows",
        description=f"Removed {removed} duplicate row(s) (compared on {scope}).",
        columns=subset,
        rows_removed=removed,
    )
    return out, record


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
def _strip_strings(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v.strip() if isinstance(v, str) else v)


def convert_series(series: pd.Series, target: str) -> tuple[pd.Series, pd.Series]:
    """Convert ``series`` to ``target`` type.

    Returns ``(converted, invalid_mask)`` where ``invalid_mask`` marks values that
    were present before but could not be converted (they become missing).
    """
    present = series.notna()
    if target in ("integer", "float"):
        source = _strip_strings(series) if is_text_column(series) else series
        if ptypes.is_bool_dtype(source):
            source = source.astype("float64")
        numeric = pd.to_numeric(source, errors="coerce")
        if is_datetime_column(series):
            numeric = pd.Series(np.nan, index=series.index)
        invalid = present & numeric.isna()
        if target == "integer":
            fractional = numeric.notna() & (numeric % 1 != 0)
            invalid = invalid | fractional
            numeric = numeric.mask(fractional)
            try:
                return numeric.astype("Int64"), invalid
            except (TypeError, ValueError, OverflowError) as exc:
                raise CleaningError("values are too large to store as integers") from exc
        return numeric.astype("float64"), invalid

    if target == "string":
        return series.astype("string"), pd.Series(False, index=series.index)

    if target == "datetime":
        if is_datetime_column(series):
            return series, pd.Series(False, index=series.index)
        source = _strip_strings(series) if is_text_column(series) else series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(source, errors="coerce", format="mixed")
            if not is_datetime_column(converted):  # mixed time zones
                converted = pd.to_datetime(source, errors="coerce", format="mixed", utc=True)
        return converted, present & converted.isna()

    raise CleaningError(f"Unsupported target type: {target}")


def convert_column_types(
    df: pd.DataFrame, conversions: dict[str, str], on_invalid: str = "coerce"
) -> tuple[pd.DataFrame, StepRecord]:
    """Convert columns to new types and report every value that failed.

    ``on_invalid="coerce"`` turns invalid values into missing values;
    ``on_invalid="skip"`` leaves a column unchanged when any value is invalid.
    """
    if on_invalid not in CONVERSION_ERROR_ACTIONS:
        raise CleaningError(f"Unknown invalid-value action: {on_invalid}")
    _require_columns(df, list(conversions))

    out = df.copy()
    converted_columns: dict[str, str] = {}
    failures: dict[str, Any] = {}
    issues = []
    for column, target in conversions.items():
        if target not in TARGET_TYPES or target == "keep":
            continue
        try:
            converted, invalid = convert_series(out[column], target)
        except CleaningError as exc:
            issues.append(f"'{column}' could not be converted to {target}: {exc}.")
            continue
        invalid_count = int(invalid.sum())
        skipped = bool(invalid_count) and on_invalid == "skip"
        if invalid_count:
            examples = out[column][invalid].head(MAX_EXAMPLES)
            failures[column] = {
                "target_type": target,
                "invalid_count": invalid_count,
                "action": "column left unchanged" if skipped else "invalid values set to missing",
                "examples": [
                    {"row": row_number(label), "value": str(value)}
                    for label, value in examples.items()
                ],
            }
        if not skipped:
            out[column] = converted
            converted_columns[column] = target

    total_invalid = sum(f["invalid_count"] for f in failures.values())
    record = StepRecord(
        operation="Convert data types",
        description=(
            f"Converted {len(converted_columns)} column(s); "
            f"{total_invalid} value(s) could not be converted."
        ),
        columns=list(converted_columns),
        details={"converted": converted_columns, "conversion_failures": failures},
        issues=issues,
    )
    return out, record


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float, float, float]:
    """Return ``(q1, q3, lower_bound, upper_bound)`` using the IQR rule."""
    values = series.dropna().astype("float64")
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    return q1, q3, q1 - multiplier * iqr, q3 + multiplier * iqr


def detect_outliers(
    df: pd.DataFrame, columns: list[str], multiplier: float = 1.5, remove: bool = False
) -> tuple[pd.DataFrame, StepRecord]:
    """Flag IQR outliers in numeric columns; remove rows only if ``remove`` is True."""
    if not 0 < multiplier <= 10:
        raise CleaningError("The IQR multiplier must be greater than 0 and at most 10.")
    _require_columns(df, columns)

    outliers: dict[str, Any] = {}
    combined = pd.Series(False, index=df.index)
    issues = []
    for column in columns:
        series = df[column]
        if not is_numeric_column(series):
            issues.append(f"'{column}' is not numeric; convert it to a number type first.")
            continue
        if series.notna().sum() < 4:
            issues.append(f"'{column}' has fewer than 4 values; outliers were not checked.")
            continue
        q1, q3, lower, upper = iqr_bounds(series, multiplier)
        mask = series.notna() & ((series < lower) | (series > upper))
        combined |= mask
        flagged = series[mask]
        outliers[column] = {
            "q1": q1,
            "q3": q3,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": int(mask.sum()),
            "examples": [
                {"row": row_number(label), "value": to_native(value)}
                for label, value in flagged.head(MAX_EXAMPLES).items()
            ],
        }

    total = int(combined.sum())
    out = df[~combined].copy() if remove else df.copy()
    removed = len(df) - len(out)
    action = f"removed {removed} row(s)" if remove else "rows were kept (detection only)"
    record = StepRecord(
        operation="Detect outliers (IQR)",
        description=f"Found {total} row(s) with outliers; {action}.",
        columns=list(outliers),
        rows_removed=removed,
        details={"multiplier": multiplier, "removed": remove, "outliers": outliers},
        issues=issues,
    )
    return out, record


# --------------------------------------------------------------------------- #
# Filtering and sorting
# --------------------------------------------------------------------------- #
def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[pd.DataFrame, StepRecord]:
    """Keep rows whose ``column`` value lies within ``[minimum, maximum]``.

    Rows with a missing value in ``column`` are removed as well.
    """
    _require_columns(df, [column])
    if minimum is None and maximum is None:
        raise CleaningError("Enter a minimum, a maximum or both.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CleaningError("The minimum must not be greater than the maximum.")
    series = df[column]
    if not is_numeric_column(series):
        raise CleaningError(f"'{column}' is not numeric; convert it to a number type first.")
    keep = series.notna()
    if minimum is not None:
        keep &= series >= minimum
    if maximum is not None:
        keep &= series <= maximum
    out = df[keep].copy()
    bounds = " and ".join(
        part for part in (
            f">= {minimum:g}" if minimum is not None else "",
            f"<= {maximum:g}" if maximum is not None else "",
        ) if part
    )
    removed = len(df) - len(out)
    record = StepRecord(
        operation="Filter by numeric range",
        description=f"Kept rows where '{column}' {bounds}; removed {removed} row(s).",
        columns=[column],
        rows_removed=removed,
        details={"minimum": minimum, "maximum": maximum},
    )
    return out, record


def filter_text_equals(df: pd.DataFrame, column: str, value: str) -> tuple[pd.DataFrame, StepRecord]:
    """Keep rows whose ``column`` value (as text) exactly equals ``value``."""
    _require_columns(df, [column])
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
    """Sort rows by ``columns``; missing values are placed last."""
    _require_columns(df, columns)
    if not columns:
        raise CleaningError("Choose at least one column to sort by.")
    try:
        out = df.sort_values(by=columns, ascending=ascending, na_position="last", kind="mergesort")
    except TypeError as exc:
        raise CleaningError(
            "a sort column mixes numbers and text; convert its type first"
        ) from exc
    direction = "ascending" if ascending else "descending"
    record = StepRecord(
        operation="Sort rows",
        description=f"Sorted by {', '.join(columns)} ({direction}).",
        columns=list(columns),
        details={"ascending": ascending},
    )
    return out, record


# --------------------------------------------------------------------------- #
# Transformation
# --------------------------------------------------------------------------- #
def add_computed_column(
    df: pd.DataFrame, source: str, operation: str, constant: float, new_name: str
) -> tuple[pd.DataFrame, StepRecord]:
    """Create ``new_name`` = ``source`` (add|subtract|multiply|divide) ``constant``.

    Only the fixed operations in ``TRANSFORM_OPERATIONS`` are supported; no user
    expression is ever evaluated.
    """
    _require_columns(df, [source])
    func = TRANSFORM_OPERATIONS.get(operation)
    if func is None:
        raise CleaningError(f"Unsupported operation: {operation}")
    if new_name in df.columns:
        raise CleaningError(f"A column named '{new_name}' already exists.")
    if operation == "divide" and constant == 0:
        raise CleaningError("Cannot divide by zero.")
    series = df[source]
    if not is_numeric_column(series):
        raise CleaningError(f"'{source}' is not numeric; convert it to a number type first.")
    out = df.copy()
    out[new_name] = func(series.astype("float64"), constant)
    symbol = {"add": "+", "subtract": "-", "multiply": "×", "divide": "÷"}[operation]
    record = StepRecord(
        operation="Create computed column",
        description=f"Created '{new_name}' = '{source}' {symbol} {constant:g}.",
        columns=[new_name],
        details={"source": source, "operation": operation, "constant": constant},
    )
    return out, record


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_cleaning_pipeline(df: pd.DataFrame, options: CleaningOptions) -> PipelineResult:
    """Apply the selected operations in a fixed, documented order.

    Order: text → data types → missing values → duplicates → outliers →
    filters → computed column → sorting → column names. Column renaming runs
    last so every other option can refer to the original column names shown on
    the overview page.
    """
    steps: list[StepRecord] = []
    errors: list[str] = []
    current = df.copy()

    def apply(label: str, func, *args, **kwargs) -> None:
        nonlocal current
        try:
            current, record = func(current, *args, **kwargs)
            steps.append(record)
        except CleaningError as exc:
            errors.append(f"{label}: {exc}")

    def present(columns: list[str]) -> list[str]:
        return [c for c in columns if c in current.columns]

    if options.text_columns and (
        options.text_trim or options.text_case != "none" or options.text_empty_to_missing
    ):
        apply(
            "Text cleaning",
            clean_text_columns,
            options.text_columns,
            trim=options.text_trim,
            case=options.text_case,
            empty_to_missing=options.text_empty_to_missing,
        )

    conversions = {c: t for c, t in options.conversions.items() if t != "keep"}
    if conversions:
        apply("Data type conversion", convert_column_types, conversions, options.conversion_errors)

    if options.missing_column_threshold is not None:
        apply(
            "Remove columns with missing values",
            drop_columns_over_threshold,
            present(options.scope),
            options.missing_column_threshold,
        )
    if options.missing_numeric_fill != "none":
        apply(
            "Fill missing numbers",
            fill_numeric_missing,
            present(options.scope),
            options.missing_numeric_fill,
        )
    if options.missing_text_fill != "none":
        apply(
            "Fill missing text",
            fill_text_missing,
            present(options.text_columns),
            options.missing_text_fill,
            options.missing_text_fill_value,
        )
    if options.missing_row_action == "drop":
        apply("Remove rows with missing values", drop_rows_with_missing, present(options.scope))

    if options.duplicates_remove:
        apply("Remove duplicates", remove_duplicate_rows, options.duplicates_subset)

    if options.outlier_columns:
        apply(
            "Outlier detection",
            detect_outliers,
            options.outlier_columns,
            options.outlier_multiplier,
            options.outlier_remove,
        )

    if options.numeric_filter_column:
        apply(
            "Numeric filter",
            filter_numeric_range,
            options.numeric_filter_column,
            options.numeric_filter_min,
            options.numeric_filter_max,
        )
    if options.text_filter_column:
        apply(
            "Text filter",
            filter_text_equals,
            options.text_filter_column,
            options.text_filter_value,
        )

    if options.transform_source:
        apply(
            "Computed column",
            add_computed_column,
            options.transform_source,
            options.transform_operation,
            options.transform_value,
            options.transform_new_name,
        )

    if options.sort_columns:
        apply("Sorting", sort_rows, options.sort_columns, options.sort_ascending)

    if options.rename_strip or options.rename_lowercase or options.rename_underscores:
        apply(
            "Standardize column names",
            standardize_column_names,
            strip=options.rename_strip,
            lowercase=options.rename_lowercase,
            underscores=options.rename_underscores,
        )

    warnings = [f"{r.operation}: {issue}" for r in steps for issue in r.issues]
    return PipelineResult(df=current, steps=steps, errors=errors, warnings=warnings)
