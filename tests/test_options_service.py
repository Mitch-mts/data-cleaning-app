"""Tests for server-side validation of the cleaning form."""

from __future__ import annotations

import pytest
from werkzeug.datastructures import MultiDict

from services.options_service import OptionsError, parse_cleaning_options

COLUMNS = ["Name", "Age", "City"]


def parse(pairs, scope=None):
    return parse_cleaning_options(MultiDict(pairs), COLUMNS, scope or COLUMNS)


def test_valid_options():
    options = parse(
        [
            ("text_columns", "Name"),
            ("text_trim", "on"),
            ("text_case", "lower"),
            ("missing_threshold_enabled", "on"),
            ("missing_threshold", "40"),
            ("duplicates_remove", "on"),
            ("duplicates_subset", "Name"),
        ]
    )
    assert options.text_columns == ["Name"]
    assert options.text_case == "lower"
    assert options.missing_column_threshold == 40
    assert options.duplicates_subset == ["Name"]


def test_no_operation_selected():
    with pytest.raises(OptionsError, match="at least one"):
        parse([])


def test_unknown_column_rejected():
    with pytest.raises(OptionsError) as info:
        parse([("text_trim", "on"), ("text_columns", "<script>")])
    assert "unknown column" in info.value.errors[0]


def test_text_columns_limited_to_scope():
    with pytest.raises(OptionsError):
        parse([("text_trim", "on"), ("text_columns", "City")], scope=["Name"])


def test_invalid_choice_rejected():
    with pytest.raises(OptionsError, match="not a valid option"):
        parse([("missing_numeric_fill", "eval(1)")])


@pytest.mark.parametrize("threshold", ["abc", "-1", "101", "nan", "inf", ""])
def test_invalid_threshold(threshold):
    with pytest.raises(OptionsError):
        parse([("missing_threshold_enabled", "on"), ("missing_threshold", threshold)])


def test_text_fill_value_required_and_limited():
    with pytest.raises(OptionsError, match="required"):
        parse([("text_columns", "Name"), ("missing_text_fill", "value")])
    with pytest.raises(OptionsError, match="at most"):
        parse([("text_columns", "Name"), ("missing_text_fill", "value"),
               ("missing_text_fill_value", "x" * 500)])


def test_text_fill_needs_text_columns():
    with pytest.raises(OptionsError, match="text column"):
        parse([("missing_text_fill", "mode")])


def test_conversions_use_column_positions():
    options = parse([("convert_type_1", "integer"), ("convert_type_2", "keep")])
    assert options.conversions == {"Age": "integer"}


def test_conversion_outside_scope_is_ignored():
    with pytest.raises(OptionsError, match="at least one"):
        parse([("convert_type_1", "integer")], scope=["Name"])


def test_invalid_conversion_type():
    with pytest.raises(OptionsError):
        parse([("convert_type_0", "pickle")])


def test_outlier_options():
    options = parse([("outlier_columns", "Age"), ("outlier_multiplier", "3"), ("outlier_remove", "on")])
    assert options.outlier_columns == ["Age"]
    assert options.outlier_multiplier == 3
    assert options.outlier_remove is True
    with pytest.raises(OptionsError):
        parse([("outlier_remove", "on")])
    with pytest.raises(OptionsError):
        parse([("outlier_columns", "Age"), ("outlier_multiplier", "0")])


def test_numeric_filter_validation():
    options = parse([("filter_numeric_column", "Age"), ("filter_numeric_min", "18")])
    assert (options.numeric_filter_min, options.numeric_filter_max) == (18, None)
    with pytest.raises(OptionsError, match="minimum, a maximum"):
        parse([("filter_numeric_column", "Age")])
    with pytest.raises(OptionsError, match="greater than"):
        parse([("filter_numeric_column", "Age"), ("filter_numeric_min", "9"),
               ("filter_numeric_max", "1")])
    with pytest.raises(OptionsError, match="must be a number"):
        parse([("filter_numeric_column", "Age"), ("filter_numeric_min", "1; DROP TABLE")])


def test_sort_options():
    options = parse([("sort_primary", "City"), ("sort_secondary", "Age"), ("sort_direction", "desc")])
    assert options.sort_columns == ["City", "Age"]
    assert options.sort_ascending is False
    with pytest.raises(OptionsError):
        parse([("sort_secondary", "Age")])


@pytest.mark.parametrize(
    "name, value, operation, message",
    [
        ("", "2", "multiply", "1–64 characters"),
        ("__import__('os')", "2", "multiply", "1–64 characters"),
        ("Age", "2", "multiply", "already exists"),
        ("ok", "0", "divide", "divide by zero"),
        ("ok", "two", "multiply", "must be a number"),
        ("ok", "2", "exec", "not a valid option"),
    ],
)
def test_transform_validation(name, value, operation, message):
    with pytest.raises(OptionsError, match=message):
        parse([("transform_source", "Age"), ("transform_operation", operation),
               ("transform_value", value), ("transform_new_name", name)])


def test_collects_multiple_errors():
    with pytest.raises(OptionsError) as info:
        parse([("text_case", "bad"), ("missing_row_action", "bad")])
    assert len(info.value.errors) == 2
