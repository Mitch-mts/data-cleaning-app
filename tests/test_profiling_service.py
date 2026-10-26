"""Tests for dataset profiling and previews."""

from __future__ import annotations

import pandas as pd
import pytest

from services import file_service
from services.profiling_service import (
    build_preview,
    detect_mixed_types,
    format_cell,
    profile_dataset,
)


def test_profile_counts(messy_df):
    profile = profile_dataset(messy_df)
    assert profile["rows"] == 6
    assert profile["columns"] == 4
    assert profile["duplicate_rows"] == 1
    by_name = {c["name"]: c for c in profile["column_profiles"]}
    assert by_name["Age"]["missing"] == 1
    assert by_name["Age"]["dtype"] == "decimal"
    assert by_name["Age"]["stats"]["median"] == pytest.approx(30)
    assert by_name["City"]["stats"] is None
    assert by_name["City"]["unique"] == 4


def test_duplicate_rows_detected():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert profile_dataset(df)["duplicate_rows"] == 1


def test_sample_file_profile(sample_bytes):
    df = file_service.load_dataset(sample_bytes, ".csv")
    profile = profile_dataset(df)
    assert profile["duplicate_rows"] == 2
    warned = {c["name"] for c in profile["column_profiles"] if c["warning"]}
    assert {"Age", "Purchase Amount"} <= warned


def test_mixed_type_detection():
    assert detect_mixed_types(pd.Series(["1", "2", "three"], name="n")) is not None
    assert detect_mixed_types(pd.Series(["a", "b"], name="t")) is None
    assert detect_mixed_types(pd.Series([1, 2], name="i")) is None


def test_preview_limits_rows_and_marks_missing():
    df = pd.DataFrame({"a": range(30), "b": [None] * 30})
    preview = build_preview(df, max_rows=10)
    assert preview["shown"] == 10
    assert preview["total"] == 30
    assert preview["rows"][0]["row_number"] == 1
    assert preview["rows"][0]["cells"] == ["0", None]


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), (float("nan"), None), (1.5, "1.5"), (2.0, "2"), ("x" * 100, "x" * 79 + "…"),
     (pd.Timestamp("2024-01-02"), "2024-01-02")],
)
def test_format_cell(value, expected):
    assert format_cell(value) == expected
