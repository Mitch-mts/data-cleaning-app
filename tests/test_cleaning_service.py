"""Tests for individual cleaning operations and the pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from services import cleaning_service as cs
from services.cleaning_service import CleaningError
from services.options_service import CleaningOptions


class TestColumnNames:
    def test_standardize(self):
        df = pd.DataFrame(columns=[" First Name ", "AGE", "Home  City"])
        out, record = cs.standardize_column_names(df)
        assert list(out.columns) == ["first_name", "age", "home_city"]
        assert record.details["renamed"][" First Name "] == "first_name"

    def test_duplicate_names_get_suffix(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["City", "city ", "CITY"])
        out, record = cs.standardize_column_names(df)
        assert list(out.columns) == ["city", "city_2", "city_3"]
        assert record.details["duplicates_resolved"] == ["city_2", "city_3"]
        assert out.iloc[0].tolist() == [1, 2, 3]

    def test_blank_name_gets_placeholder(self):
        df = pd.DataFrame([[1, 2]], columns=["   ", "a"])
        out, _ = cs.standardize_column_names(df)
        assert list(out.columns) == ["column_1", "a"]

    def test_options_are_independent(self):
        df = pd.DataFrame(columns=[" My Col "])
        out, _ = cs.standardize_column_names(df, strip=True, lowercase=False, underscores=False)
        assert list(out.columns) == ["My Col"]


class TestText:
    def test_trim_case_and_blanks(self, messy_df):
        out, record = cs.clean_text_columns(
            messy_df, [" First Name ", "City"], trim=True, case="title", empty_to_missing=True
        )
        assert out[" First Name "].tolist()[:3] == ["Alice", "Bob", "Bob"]
        assert pd.isna(out[" First Name "].iloc[3])
        assert pd.isna(out["City"].iloc[3])
        assert record.details["blanks_to_missing"] == {" First Name ": 1, "City": 1}
        assert messy_df[" First Name "].iloc[0] == " Alice "  # input not modified

    def test_non_text_column_is_skipped_with_issue(self, messy_df):
        out, record = cs.clean_text_columns(messy_df, ["Age"], trim=True)
        assert out["Age"].equals(messy_df["Age"])
        assert record.issues

    def test_invalid_case_raises(self, messy_df):
        with pytest.raises(CleaningError):
            cs.clean_text_columns(messy_df, ["City"], case="shout")


class TestMissingValues:
    def test_counts(self, messy_df):
        counts = cs.missing_value_counts(messy_df)
        assert counts == {" First Name ": 1, "Age": 1, "City": 1, "Score": 1}

    def test_drop_columns_over_threshold(self):
        df = pd.DataFrame({"mostly_empty": [None, None, None, 1], "full": [1, 2, 3, 4]})
        out, record = cs.drop_columns_over_threshold(df, ["mostly_empty", "full"], 50)
        assert list(out.columns) == ["full"]
        assert record.columns_removed == ["mostly_empty"]

    def test_threshold_is_strictly_above(self):
        df = pd.DataFrame({"half": [None, 1]})
        out, _ = cs.drop_columns_over_threshold(df, ["half"], 50)
        assert list(out.columns) == ["half"]

    def test_invalid_threshold(self, messy_df):
        with pytest.raises(CleaningError):
            cs.drop_columns_over_threshold(messy_df, ["Age"], 150)

    def test_fill_numeric_mean_and_median(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, 10.0]})
        mean_df, record = cs.fill_numeric_missing(df, ["x"], "mean")
        assert mean_df["x"].iloc[1] == pytest.approx(14 / 3)
        assert record.missing_filled == 1
        median_df, _ = cs.fill_numeric_missing(df, ["x"], "median")
        assert median_df["x"].iloc[1] == pytest.approx(3.0)

    def test_fill_numeric_skips_text_columns(self, messy_df):
        out, record = cs.fill_numeric_missing(messy_df, ["City", "Age"], "median")
        assert record.columns == ["Age"]
        assert out["City"].isna().sum() == 1

    def test_fill_int_column_with_fractional_mean(self):
        df = pd.DataFrame({"n": pd.array([1, None, 2], dtype="Int64")})
        out, record = cs.fill_numeric_missing(df, ["n"], "mean")
        assert out["n"].iloc[1] == pytest.approx(1.5)
        assert record.issues

    def test_fill_text_value_and_mode(self):
        df = pd.DataFrame({"c": ["a", "b", "b", None]})
        value_df, _ = cs.fill_text_missing(df, ["c"], "value", "Unknown")
        assert value_df["c"].iloc[3] == "Unknown"
        mode_df, record = cs.fill_text_missing(df, ["c"], "mode")
        assert mode_df["c"].iloc[3] == "b"
        assert record.missing_filled == 1

    def test_fill_text_value_required(self):
        with pytest.raises(CleaningError):
            cs.fill_text_missing(pd.DataFrame({"c": [None]}), ["c"], "value", "")

    def test_drop_rows_with_missing(self, messy_df):
        out, record = cs.drop_rows_with_missing(messy_df, ["Age", "City"])
        assert len(out) == 4
        assert record.rows_removed == 2


class TestDuplicates:
    def test_remove_full_row_duplicates(self, messy_df):
        out, record = cs.remove_duplicate_rows(messy_df)
        assert len(out) == 5
        assert record.rows_removed == 1

    def test_remove_duplicates_on_subset(self):
        df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "b", "c"]})
        out, record = cs.remove_duplicate_rows(df, ["id"])
        assert out["v"].tolist() == ["a", "c"]
        assert record.rows_removed == 1

    def test_unknown_subset_column(self, messy_df):
        with pytest.raises(CleaningError):
            cs.remove_duplicate_rows(messy_df, ["nope"])


class TestTypeConversion:
    def test_integer_conversion_reports_failures(self):
        df = pd.DataFrame({"age": ["34", " 29 ", "thirty", None, "36.5"]})
        out, record = cs.convert_column_types(df, {"age": "integer"})
        assert str(out["age"].dtype) == "Int64"
        assert out["age"].tolist()[:2] == [34, 29]
        failure = record.details["conversion_failures"]["age"]
        assert failure["invalid_count"] == 2
        assert {e["value"] for e in failure["examples"]} == {"thirty", "36.5"}
        assert failure["examples"][0]["row"] == 3
        assert out["age"].isna().sum() == 3

    def test_float_conversion(self):
        df = pd.DataFrame({"amount": ["1.5", "abc", "", "2"]})
        out, record = cs.convert_column_types(df, {"amount": "float"})
        assert out["amount"].dtype == "float64"
        assert out["amount"].iloc[0] == pytest.approx(1.5)
        assert record.details["conversion_failures"]["amount"]["invalid_count"] == 2

    def test_skip_leaves_column_unchanged(self):
        df = pd.DataFrame({"n": ["1", "x"]})
        out, record = cs.convert_column_types(df, {"n": "integer"}, on_invalid="skip")
        assert out["n"].tolist() == ["1", "x"]
        assert record.columns == []
        assert record.details["conversion_failures"]["n"]["action"] == "column left unchanged"

    def test_datetime_conversion_with_malformed_dates(self):
        df = pd.DataFrame({"d": ["2024-01-15", "2024/02/20", "not a date", "2024-13-45", None]})
        out, record = cs.convert_column_types(df, {"d": "datetime"})
        assert pd.api.types.is_datetime64_any_dtype(out["d"])
        assert out["d"].iloc[0] == pd.Timestamp("2024-01-15")
        assert out["d"].iloc[1] == pd.Timestamp("2024-02-20")
        assert record.details["conversion_failures"]["d"]["invalid_count"] == 2

    def test_string_conversion_keeps_missing(self):
        out, _ = cs.convert_column_types(pd.DataFrame({"n": [1, None]}), {"n": "string"})
        assert out["n"].iloc[0] == "1.0"
        assert pd.isna(out["n"].iloc[1])

    def test_invalid_action(self):
        with pytest.raises(CleaningError):
            cs.convert_column_types(pd.DataFrame({"n": [1]}), {"n": "float"}, on_invalid="drop")


class TestOutliers:
    def test_iqr_detection_does_not_remove_by_default(self):
        df = pd.DataFrame({"x": [10, 12, 11, 13, 12, 11, 500]})
        out, record = cs.detect_outliers(df, ["x"])
        assert len(out) == 7
        info = record.details["outliers"]["x"]
        assert info["count"] == 1
        assert info["examples"] == [{"row": 7, "value": 500}]
        assert record.rows_removed == 0

    def test_explicit_removal(self):
        df = pd.DataFrame({"x": [10, 12, 11, 13, 12, 11, 500, -400]})
        out, record = cs.detect_outliers(df, ["x"], remove=True)
        assert 500 not in out["x"].tolist()
        assert record.rows_removed == 2

    def test_non_numeric_column_is_reported(self, messy_df):
        _, record = cs.detect_outliers(messy_df, ["City"])
        assert record.issues and "not numeric" in record.issues[0]

    def test_invalid_multiplier(self, messy_df):
        with pytest.raises(CleaningError):
            cs.detect_outliers(messy_df, ["Age"], multiplier=0)


class TestFilterSortTransform:
    def test_numeric_range_filter(self, messy_df):
        out, record = cs.filter_numeric_range(messy_df, "Age", 25, 40)
        assert out["Age"].tolist() == [30, 25, 25, 40]
        assert record.rows_removed == 2

    def test_numeric_filter_on_text_column(self, messy_df):
        with pytest.raises(CleaningError, match="not numeric"):
            cs.filter_numeric_range(messy_df, "City", 0, 1)

    def test_numeric_filter_min_greater_than_max(self, messy_df):
        with pytest.raises(CleaningError):
            cs.filter_numeric_range(messy_df, "Age", 50, 10)

    def test_text_filter(self, messy_df):
        out, record = cs.filter_text_equals(messy_df, "City", "Rome")
        assert len(out) == 2
        assert record.rows_removed == 4

    def test_sort_with_missing_last(self, messy_df):
        out, _ = cs.sort_rows(messy_df, ["Age"], ascending=False)
        assert out["Age"].tolist()[:2] == [1000, 40]
        assert pd.isna(out["Age"].iloc[-1])

    def test_sort_mixed_types_is_reported(self):
        df = pd.DataFrame({"m": [1, "a", 2]})
        with pytest.raises(CleaningError, match="mixes"):
            cs.sort_rows(df, ["m"])

    def test_computed_column(self, messy_df):
        out, record = cs.add_computed_column(messy_df, "Age", "multiply", 2, "age_x2")
        assert out["age_x2"].iloc[0] == 60
        assert "age_x2" not in messy_df.columns
        assert record.columns == ["age_x2"]

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"source": "Age", "operation": "divide", "constant": 0, "new_name": "z"}, "zero"),
            ({"source": "Age", "operation": "power", "constant": 2, "new_name": "z"}, "Unsupported"),
            ({"source": "City", "operation": "add", "constant": 1, "new_name": "z"}, "not numeric"),
            ({"source": "Age", "operation": "add", "constant": 1, "new_name": "City"}, "exists"),
        ],
    )
    def test_computed_column_errors(self, messy_df, kwargs, message):
        with pytest.raises(CleaningError, match=message):
            cs.add_computed_column(messy_df, **kwargs)


class TestPipeline:
    def test_conversion_enables_numeric_steps(self):
        df = pd.DataFrame({"amount": ["10", "12", "11", "13", "x", "900", "12"]})
        options = CleaningOptions(
            scope=["amount"],
            conversions={"amount": "float"},
            outlier_columns=["amount"],
            transform_source="amount",
            transform_operation="add",
            transform_value=1,
            transform_new_name="amount_plus_one",
            sort_columns=["amount"],
        )
        result = cs.run_cleaning_pipeline(df, options)
        assert result.errors == []
        assert [s.operation for s in result.steps] == [
            "Convert data types",
            "Detect outliers (IQR)",
            "Create computed column",
            "Sort rows",
        ]
        assert result.steps[1].details["outliers"]["amount"]["count"] == 1
        assert result.df["amount_plus_one"].iloc[0] == 11


    def test_only_selected_operations_run(self, messy_df):
        options = CleaningOptions(scope=list(messy_df.columns), duplicates_remove=True)
        result = cs.run_cleaning_pipeline(messy_df, options)
        assert [s.operation for s in result.steps] == ["Remove duplicate rows"]
        assert list(result.df.columns) == list(messy_df.columns)

    def test_rename_runs_last(self, messy_df):
        options = CleaningOptions(
            scope=list(messy_df.columns),
            text_columns=[" First Name "],
            text_trim=True,
            rename_strip=True,
            rename_lowercase=True,
            rename_underscores=True,
        )
        result = cs.run_cleaning_pipeline(messy_df, options)
        assert "first_name" in result.df.columns
        assert result.df["first_name"].iloc[0] == "Alice"
        assert result.steps[-1].operation == "Standardize column names"

    def test_original_is_preserved(self, messy_df):
        snapshot = messy_df.copy()
        options = CleaningOptions(
            scope=list(messy_df.columns), missing_row_action="drop", duplicates_remove=True
        )
        cs.run_cleaning_pipeline(messy_df, options)
        pd.testing.assert_frame_equal(messy_df, snapshot)

    def test_step_errors_are_reported_not_raised(self, messy_df):
        options = CleaningOptions(
            scope=list(messy_df.columns), duplicates_remove=True, duplicates_subset=["gone"]
        )
        result = cs.run_cleaning_pipeline(messy_df, options)
        assert result.errors and "gone" in result.errors[0]
        assert len(result.df) == len(messy_df)
