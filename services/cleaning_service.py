"""Public entry point for data cleaning.

The operations live in ``services/cleaning/``. This module re-exports them so
existing imports keep working. Start with :func:`run_cleaning_pipeline`.
"""

from services.cleaning.checks import row_number
from services.cleaning.constants import (
    CONVERSION_ERROR_ACTIONS,
    MAX_EXAMPLES,
    MISSING_ROW_ACTIONS,
    NUMERIC_FILL_STRATEGIES,
    SORT_DIRECTIONS,
    TARGET_TYPES,
    TEXT_CASES,
    TEXT_FILL_STRATEGIES,
    TRANSFORM_OPERATIONS,
)
from services.cleaning.duplicates import remove_duplicate_rows
from services.cleaning.filters import filter_numeric_range, filter_text_equals, sort_rows
from services.cleaning.missing import (
    drop_columns_over_threshold,
    drop_rows_with_missing,
    fill_numeric_missing,
    fill_text_missing,
    missing_value_counts,
)
from services.cleaning.names import make_unique_names, standardize_column_names
from services.cleaning.outliers import detect_outliers, iqr_bounds
from services.cleaning.pipeline import run_cleaning_pipeline
from services.cleaning.records import CleaningError, PipelineResult, StepRecord
from services.cleaning.text import clean_text_columns
from services.cleaning.transform import add_computed_column
from services.cleaning.types import convert_column_types, convert_series

__all__ = [
    "CONVERSION_ERROR_ACTIONS",
    "MAX_EXAMPLES",
    "MISSING_ROW_ACTIONS",
    "NUMERIC_FILL_STRATEGIES",
    "SORT_DIRECTIONS",
    "TARGET_TYPES",
    "TEXT_CASES",
    "TEXT_FILL_STRATEGIES",
    "TRANSFORM_OPERATIONS",
    "CleaningError",
    "PipelineResult",
    "StepRecord",
    "add_computed_column",
    "clean_text_columns",
    "convert_column_types",
    "convert_series",
    "detect_outliers",
    "drop_columns_over_threshold",
    "drop_rows_with_missing",
    "fill_numeric_missing",
    "fill_text_missing",
    "filter_numeric_range",
    "filter_text_equals",
    "iqr_bounds",
    "make_unique_names",
    "missing_value_counts",
    "remove_duplicate_rows",
    "row_number",
    "run_cleaning_pipeline",
    "sort_rows",
    "standardize_column_names",
]
