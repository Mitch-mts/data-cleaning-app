"""Apply the selected cleaning operations in a fixed order.

Order:

1. Text
2. Data types
3. Missing values
4. Duplicates
5. Outliers
6. Filters
7. Computed column
8. Sorting
9. Column names

Column renaming runs last so every earlier step can use the original column
names shown on the overview page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from services.cleaning.duplicates import remove_duplicate_rows
from services.cleaning.filters import filter_numeric_range, filter_text_equals, sort_rows
from services.cleaning.missing import (
    drop_columns_over_threshold,
    drop_rows_with_missing,
    fill_numeric_missing,
    fill_text_missing,
)
from services.cleaning.names import standardize_column_names
from services.cleaning.outliers import detect_outliers
from services.cleaning.records import PipelineResult, StepRecord
from services.cleaning.steps import run_step
from services.cleaning.text import clean_text_columns
from services.cleaning.transform import add_computed_column
from services.cleaning.types import convert_column_types

if TYPE_CHECKING:
    from services.options_service import CleaningOptions


def columns_still_present(data: pd.DataFrame, column_names: list[str]) -> list[str]:
    """Skip columns an earlier step already removed."""
    return [name for name in column_names if name in data.columns]


def _warnings_from(steps: list[StepRecord]) -> list[str]:
    """Turn each step's notes into ``Operation: note`` lines for the report."""
    warnings = []
    for record in steps:
        for issue in record.issues:
            warnings.append(f"{record.operation}: {issue}")
    return warnings


def run_cleaning_pipeline(df: pd.DataFrame, options: CleaningOptions) -> PipelineResult:
    """Apply only the operations the user turned on, in the order above.

    A step that cannot run is recorded in ``errors`` and the remaining steps
    still run. The uploaded table is copied first and never modified.
    """
    data = df.copy()
    steps: list[StepRecord] = []
    errors: list[str] = []

    user_wants_text_cleaning = options.text_columns and (
        options.text_trim or options.text_case != "none" or options.text_empty_to_missing
    )
    if user_wants_text_cleaning:
        data = run_step(
            data,
            steps,
            errors,
            "Text cleaning",
            clean_text_columns,
            options.text_columns,
            trim=options.text_trim,
            case=options.text_case,
            empty_to_missing=options.text_empty_to_missing,
        )

    conversions = {
        column: target
        for column, target in options.conversions.items()
        if target != "keep"
    }
    if conversions:
        data = run_step(
            data,
            steps,
            errors,
            "Data type conversion",
            convert_column_types,
            conversions,
            options.conversion_errors,
        )

    if options.missing_column_threshold is not None:
        data = run_step(
            data,
            steps,
            errors,
            "Remove columns with missing values",
            drop_columns_over_threshold,
            columns_still_present(data, options.scope),
            options.missing_column_threshold,
        )
    if options.missing_numeric_fill != "none":
        data = run_step(
            data,
            steps,
            errors,
            "Fill missing numbers",
            fill_numeric_missing,
            columns_still_present(data, options.scope),
            options.missing_numeric_fill,
        )
    if options.missing_text_fill != "none":
        data = run_step(
            data,
            steps,
            errors,
            "Fill missing text",
            fill_text_missing,
            columns_still_present(data, options.text_columns),
            options.missing_text_fill,
            options.missing_text_fill_value,
        )
    if options.missing_row_action == "drop":
        data = run_step(
            data,
            steps,
            errors,
            "Remove rows with missing values",
            drop_rows_with_missing,
            columns_still_present(data, options.scope),
        )

    if options.duplicates_remove:
        data = run_step(
            data,
            steps,
            errors,
            "Remove duplicates",
            remove_duplicate_rows,
            options.duplicates_subset,
        )

    if options.outlier_columns:
        data = run_step(
            data,
            steps,
            errors,
            "Outlier detection",
            detect_outliers,
            options.outlier_columns,
            options.outlier_multiplier,
            options.outlier_remove,
        )

    if options.numeric_filter_column:
        data = run_step(
            data,
            steps,
            errors,
            "Numeric filter",
            filter_numeric_range,
            options.numeric_filter_column,
            options.numeric_filter_min,
            options.numeric_filter_max,
        )
    if options.text_filter_column:
        data = run_step(
            data,
            steps,
            errors,
            "Text filter",
            filter_text_equals,
            options.text_filter_column,
            options.text_filter_value,
        )

    if options.transform_source:
        data = run_step(
            data,
            steps,
            errors,
            "Computed column",
            add_computed_column,
            options.transform_source,
            options.transform_operation,
            options.transform_value,
            options.transform_new_name,
        )

    if options.sort_columns:
        data = run_step(
            data,
            steps,
            errors,
            "Sorting",
            sort_rows,
            options.sort_columns,
            options.sort_ascending,
        )

    if options.rename_strip or options.rename_lowercase or options.rename_underscores:
        data = run_step(
            data,
            steps,
            errors,
            "Standardize column names",
            standardize_column_names,
            strip=options.rename_strip,
            lowercase=options.rename_lowercase,
            underscores=options.rename_underscores,
        )

    return PipelineResult(df=data, steps=steps, errors=errors, warnings=_warnings_from(steps))
