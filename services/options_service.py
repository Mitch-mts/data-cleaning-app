"""Server-side validation of the cleaning form.

The browser form is never trusted: every column name must exist in the
dataset, every choice must come from a fixed list and every number is range
checked. Validation errors are collected so the user sees all of them at once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from services.cleaning_service import (
    CONVERSION_ERROR_ACTIONS,
    MISSING_ROW_ACTIONS,
    NUMERIC_FILL_STRATEGIES,
    SORT_DIRECTIONS,
    TARGET_TYPES,
    TEXT_CASES,
    TEXT_FILL_STRATEGIES,
    TRANSFORM_OPERATIONS,
)

MAX_TEXT_INPUT = 200
NEW_COLUMN_PATTERN = re.compile(r"^[A-Za-z0-9_ \-]{1,64}$")


class FormData(Protocol):
    """Minimal interface shared by Flask's ``request.form`` and ``MultiDict``."""

    def get(self, key: str, default: str | None = None) -> str | None: ...

    def getlist(self, key: str) -> list[str]: ...


class OptionsError(ValueError):
    """Raised with a list of user-facing validation messages."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class CleaningOptions:
    """Validated cleaning settings. Column names refer to the original dataset."""

    scope: list[str]
    rename_strip: bool = False
    rename_lowercase: bool = False
    rename_underscores: bool = False
    text_columns: list[str] = field(default_factory=list)
    text_trim: bool = False
    text_case: str = "none"
    text_empty_to_missing: bool = False
    missing_row_action: str = "keep"
    missing_column_threshold: float | None = None
    missing_numeric_fill: str = "none"
    missing_text_fill: str = "none"
    missing_text_fill_value: str = ""
    duplicates_remove: bool = False
    duplicates_subset: list[str] = field(default_factory=list)
    conversions: dict[str, str] = field(default_factory=dict)
    conversion_errors: str = "coerce"
    outlier_columns: list[str] = field(default_factory=list)
    outlier_multiplier: float = 1.5
    outlier_remove: bool = False
    numeric_filter_column: str | None = None
    numeric_filter_min: float | None = None
    numeric_filter_max: float | None = None
    text_filter_column: str | None = None
    text_filter_value: str = ""
    sort_columns: list[str] = field(default_factory=list)
    sort_ascending: bool = True
    transform_source: str | None = None
    transform_operation: str = "multiply"
    transform_value: float = 1.0
    transform_new_name: str = ""

    def has_operations(self) -> bool:
        """True when at least one cleaning operation is selected."""
        return any(
            [
                self.rename_strip,
                self.rename_lowercase,
                self.rename_underscores,
                self.text_columns
                and (self.text_trim or self.text_case != "none" or self.text_empty_to_missing),
                self.missing_row_action != "keep",
                self.missing_column_threshold is not None,
                self.missing_numeric_fill != "none",
                self.missing_text_fill != "none",
                self.duplicates_remove,
                any(t != "keep" for t in self.conversions.values()),
                self.outlier_columns,
                self.numeric_filter_column,
                self.text_filter_column,
                self.sort_columns,
                self.transform_source,
            ]
        )


class _Parser:
    """Small helper that reads form fields and collects validation errors."""

    def __init__(self, form: FormData):
        self.form = form
        self.errors: list[str] = []

    def flag(self, name: str) -> bool:
        return self.form.get(name) == "on"

    def text(self, name: str) -> str:
        return (self.form.get(name) or "").strip()

    def choice(self, name: str, allowed: tuple[str, ...], label: str) -> str:
        value = self.text(name) or allowed[0]
        if value not in allowed:
            self.errors.append(f"{label}: '{value}' is not a valid option.")
            return allowed[0]
        return value

    def columns(self, name: str, allowed: list[str], label: str) -> list[str]:
        selected = self.form.getlist(name)
        unknown = [c for c in selected if c not in allowed]
        if unknown:
            self.errors.append(f"{label}: unknown column(s) {', '.join(unknown)}.")
        seen: list[str] = []
        for column in selected:
            if column in allowed and column not in seen:
                seen.append(column)
        return seen

    def column(self, name: str, allowed: list[str], label: str) -> str | None:
        value = self.form.get(name) or ""
        if not value:
            return None
        if value not in allowed:
            self.errors.append(f"{label}: unknown column {value}.")
            return None
        return value

    def number(
        self,
        name: str,
        label: str,
        required: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        raw = self.text(name)
        if not raw:
            if required:
                self.errors.append(f"{label} is required.")
            return None
        try:
            value = float(raw)
        except ValueError:
            self.errors.append(f"{label} must be a number.")
            return None
        if not math.isfinite(value):
            self.errors.append(f"{label} must be a finite number.")
            return None
        if minimum is not None and value < minimum:
            self.errors.append(f"{label} must be at least {minimum:g}.")
            return None
        if maximum is not None and value > maximum:
            self.errors.append(f"{label} must be at most {maximum:g}.")
            return None
        return value

    def limited_text(self, name: str, label: str, required: bool = False) -> str:
        value = self.text(name)
        if required and not value:
            self.errors.append(f"{label} is required.")
        if len(value) > MAX_TEXT_INPUT:
            self.errors.append(f"{label} must be at most {MAX_TEXT_INPUT} characters.")
            value = value[:MAX_TEXT_INPUT]
        return value


def parse_cleaning_options(
    form: FormData, all_columns: list[str], scope: list[str]
) -> CleaningOptions:
    """Validate the cleaning form and return :class:`CleaningOptions`.

    Raises ``OptionsError`` listing every problem found.
    """
    p = _Parser(form)
    scope = [c for c in scope if c in all_columns] or list(all_columns)

    options = CleaningOptions(scope=scope)
    options.rename_strip = p.flag("rename_strip")
    options.rename_lowercase = p.flag("rename_lowercase")
    options.rename_underscores = p.flag("rename_underscores")

    options.text_columns = p.columns("text_columns", scope, "Text columns")
    options.text_trim = p.flag("text_trim")
    options.text_case = p.choice("text_case", TEXT_CASES, "Capitalisation")
    options.text_empty_to_missing = p.flag("text_empty_to_missing")

    options.missing_row_action = p.choice(
        "missing_row_action", MISSING_ROW_ACTIONS, "Rows with missing values"
    )
    if p.flag("missing_threshold_enabled"):
        options.missing_column_threshold = p.number(
            "missing_threshold", "Missing-value threshold", required=True, minimum=0, maximum=100
        )
    options.missing_numeric_fill = p.choice(
        "missing_numeric_fill", NUMERIC_FILL_STRATEGIES, "Numeric fill"
    )
    options.missing_text_fill = p.choice("missing_text_fill", TEXT_FILL_STRATEGIES, "Text fill")
    if options.missing_text_fill == "value":
        options.missing_text_fill_value = p.limited_text(
            "missing_text_fill_value", "Text fill value", required=True
        )
    if options.missing_text_fill != "none" and not options.text_columns:
        p.errors.append("Text fill: choose at least one text column in the Text section.")

    options.duplicates_remove = p.flag("duplicates_remove")
    options.duplicates_subset = p.columns(
        "duplicates_subset", list(all_columns), "Duplicate check columns"
    )

    # Conversion fields are keyed by column position so names never appear in field names.
    for position, column in enumerate(all_columns):
        if column in scope:
            target = p.choice(f"convert_type_{position}", TARGET_TYPES, f"Type for '{column}'")
            if target != "keep":
                options.conversions[column] = target
    options.conversion_errors = p.choice(
        "conversion_errors", CONVERSION_ERROR_ACTIONS, "Invalid values"
    )

    options.outlier_columns = p.columns("outlier_columns", scope, "Outlier columns")
    if options.outlier_columns:
        multiplier = p.number(
            "outlier_multiplier", "IQR multiplier", required=True, minimum=0.1, maximum=10
        )
        options.outlier_multiplier = multiplier if multiplier is not None else 1.5
    options.outlier_remove = p.flag("outlier_remove")
    if options.outlier_remove and not options.outlier_columns:
        p.errors.append("Outlier removal: choose the columns to check for outliers.")

    options.numeric_filter_column = p.column(
        "filter_numeric_column", list(all_columns), "Numeric filter"
    )
    if options.numeric_filter_column:
        errors_before = len(p.errors)
        options.numeric_filter_min = p.number("filter_numeric_min", "Filter minimum")
        options.numeric_filter_max = p.number("filter_numeric_max", "Filter maximum")
        low, high = options.numeric_filter_min, options.numeric_filter_max
        if low is None and high is None and len(p.errors) == errors_before:
            p.errors.append("Numeric filter: enter a minimum, a maximum or both.")
        if low is not None and high is not None and low > high:
            p.errors.append("Numeric filter: the minimum must not be greater than the maximum.")

    options.text_filter_column = p.column("filter_text_column", list(all_columns), "Text filter")
    if options.text_filter_column:
        options.text_filter_value = p.limited_text(
            "filter_text_value", "Text filter value", required=True
        )

    primary = p.column("sort_primary", list(all_columns), "Sort by")
    secondary = p.column("sort_secondary", list(all_columns), "Then sort by")
    if secondary and not primary:
        p.errors.append("Sorting: choose the first sort column before a second one.")
    options.sort_columns = [c for c in (primary, secondary) if c]
    if primary and secondary == primary:
        options.sort_columns = [primary]
    options.sort_ascending = p.choice("sort_direction", SORT_DIRECTIONS, "Sort direction") == "asc"

    options.transform_source = p.column("transform_source", list(all_columns), "Computed column")
    if options.transform_source:
        options.transform_operation = p.choice(
            "transform_operation", tuple(TRANSFORM_OPERATIONS), "Computed column operation"
        )
        value = p.number("transform_value", "Computed column value", required=True)
        options.transform_value = value if value is not None else 1.0
        if options.transform_operation == "divide" and value == 0:
            p.errors.append("Computed column: cannot divide by zero.")
        name = p.text("transform_new_name")
        if not NEW_COLUMN_PATTERN.fullmatch(name):
            p.errors.append(
                "Computed column name must be 1–64 characters using letters, numbers, "
                "spaces, underscores or hyphens."
            )
        elif name in all_columns:
            p.errors.append(f"Computed column name '{name}' already exists.")
        options.transform_new_name = name

    if not p.errors and not options.has_operations():
        p.errors.append("Select at least one cleaning operation.")
    if p.errors:
        raise OptionsError(p.errors)
    return options
