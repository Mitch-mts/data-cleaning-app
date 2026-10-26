"""Tests for the cleaning report."""

from __future__ import annotations

import json

from services.cleaning_service import run_cleaning_pipeline
from services.options_service import CleaningOptions
from services.report_service import build_report, compare, report_to_json, summarize


def test_summarize(messy_df):
    assert summarize(messy_df) == {
        "rows": 6,
        "columns": 4,
        "missing_values": 4,
        "duplicate_rows": 1,
    }


def test_report_matches_pipeline(messy_df):
    options = CleaningOptions(
        scope=list(messy_df.columns),
        missing_numeric_fill="median",
        missing_column_threshold=99,
        duplicates_remove=True,
        rename_lowercase=True,
    )
    result = run_cleaning_pipeline(messy_df, options)
    report = build_report("people.csv", messy_df, result)

    assert report["source_file"] == "people.csv"
    assert report["summary"]["before"]["rows"] == 6
    assert report["summary"]["after"]["rows"] == 5
    assert report["summary"]["rows_removed"] == 1
    assert report["summary"]["missing_values_handled"] == 1
    assert report["summary"]["after"]["duplicate_rows"] == 0
    assert [op["operation"] for op in report["operations"]] == [
        "Remove columns with many missing values",
        "Fill missing numbers (median)",
        "Remove duplicate rows",
        "Standardize column names",
    ]
    assert report["errors"] == []


def test_report_is_json_serialisable(messy_df):
    options = CleaningOptions(scope=list(messy_df.columns), missing_numeric_fill="mean")
    report = build_report("x.csv", messy_df, run_cleaning_pipeline(messy_df, options))
    parsed = json.loads(report_to_json(report))
    fill_value = parsed["operations"][0]["details"]["fill_values"]["Age"]
    assert isinstance(fill_value, float)


def test_compare_rows():
    rows = compare(
        {"rows": 10, "columns": 3, "missing_values": 4, "duplicate_rows": 2},
        {"rows": 8, "columns": 3, "missing_values": 0, "duplicate_rows": 0},
    )
    assert rows[0] == {"metric": "Rows", "before": 10, "after": 8, "change": -2}
