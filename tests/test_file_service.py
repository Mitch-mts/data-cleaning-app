"""Tests for file validation, loading, export and workspaces."""

from __future__ import annotations

import io
import os
import time

import pandas as pd
import pytest

from services import file_service
from services.file_service import ExportError, FileValidationError

MAX = 10 * 1024 * 1024


class TestValidateUpload:
    def test_accepts_csv_and_xlsx(self, sample_bytes, xlsx_bytes):
        assert file_service.validate_upload("a.CSV", sample_bytes, MAX) == ".csv"
        assert file_service.validate_upload("b.xlsx", xlsx_bytes, MAX) == ".xlsx"

    @pytest.mark.parametrize("name", ["data.txt", "data.xls", "data.csv.exe", "data", ""])
    def test_rejects_unsupported_extensions(self, name):
        with pytest.raises(FileValidationError):
            file_service.validate_upload(name, b"a,b\n1,2\n", MAX)

    def test_rejects_empty_file(self):
        with pytest.raises(FileValidationError, match="empty"):
            file_service.validate_upload("a.csv", b"", MAX)

    def test_rejects_oversized_file(self):
        with pytest.raises(FileValidationError, match="too large"):
            file_service.validate_upload("a.csv", b"a\n" * 20, 10)

    def test_rejects_fake_excel(self):
        with pytest.raises(FileValidationError, match="Excel"):
            file_service.validate_upload("a.xlsx", b"a,b\n1,2\n", MAX)

    def test_rejects_binary_csv(self):
        with pytest.raises(FileValidationError):
            file_service.validate_upload("a.csv", b"\x00\x01\x02binary", MAX)


class TestLoadDataset:
    def test_loads_sample_csv(self, sample_bytes):
        df = file_service.load_dataset(sample_bytes, ".csv")
        assert df.shape == (15, 9)
        assert " Customer ID " in df.columns
        assert "city " in df.columns

    def test_loads_semicolon_csv(self):
        df = file_service.load_dataset(b"a;b\n1;2\n3;4\n", ".csv")
        assert list(df.columns) == ["a", "b"]
        assert df.shape == (2, 2)

    def test_loads_latin1_csv(self):
        df = file_service.load_dataset("name\nJosé\n".encode("cp1252"), ".csv")
        assert df.iloc[0, 0] == "José"

    def test_loads_excel(self, xlsx_bytes):
        df = file_service.load_dataset(xlsx_bytes, ".xlsx")
        assert list(df.columns) == ["Name", "Amount"]
        assert df["Amount"].sum() == pytest.approx(4.0)

    def test_header_only_csv_is_rejected(self):
        with pytest.raises(FileValidationError, match="no data rows"):
            file_service.load_dataset(b"a,b,c\n", ".csv")

    def test_whitespace_only_csv_is_rejected(self):
        with pytest.raises(FileValidationError, match="empty"):
            file_service.load_dataset(b"   \n\n", ".csv")

    def test_malformed_csv_is_rejected(self):
        with pytest.raises(FileValidationError, match="malformed"):
            file_service.load_dataset(b"a,b\n1,2\n3,4,5,6\n", ".csv")

    def test_corrupted_excel_is_rejected(self):
        with pytest.raises(FileValidationError, match="could not be read"):
            file_service.load_dataset(b"PK\x03\x04garbage-not-a-workbook", ".xlsx")

    def test_excel_formulas_are_not_evaluated(self):
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["value", "formula"])
        sheet.append([1, "=1+1"])
        buffer = io.BytesIO()
        workbook.save(buffer)
        df = file_service.load_dataset(buffer.getvalue(), ".xlsx")
        # No cached value exists because nothing ever calculated the formula.
        assert df["formula"].isna().all()


class TestExport:
    def test_csv_round_trip(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
        restored = file_service.load_dataset(file_service.export_csv(df), ".csv")
        assert restored["a"].tolist() == [1, 2]
        assert restored["b"].isna().tolist() == [False, True]

    def test_excel_round_trip(self):
        df = pd.DataFrame(
            {
                "n": pd.array([1, None], dtype="Int64"),
                "when": pd.to_datetime(["2024-01-01", None]).tz_localize("UTC"),
                "label": ["a", "b"],
            }
        )
        restored = file_service.load_dataset(file_service.export_excel(df), ".xlsx")
        assert restored.shape == (2, 3)
        assert restored["n"].iloc[0] == 1
        assert pd.isna(restored["n"].iloc[1])
        assert restored["when"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_formula_like_text_is_neutralized(self):
        df = pd.DataFrame({"t": ["=SUM(A1:A2)", "+cmd", "@x", "-5", "normal"]})
        out = file_service.neutralize_formulas(df)
        assert out["t"].tolist() == ["'=SUM(A1:A2)", "'+cmd", "'@x", "-5", "normal"]
        assert df["t"].iloc[0] == "=SUM(A1:A2)"  # input left untouched

    def test_excel_export_row_limit(self, monkeypatch):
        monkeypatch.setattr(file_service, "EXCEL_MAX_ROWS", 2)
        with pytest.raises(ExportError):
            file_service.export_excel(pd.DataFrame({"a": [1, 2, 3]}))


class TestWorkspaces:
    def test_create_get_delete(self, tmp_path):
        workspace_id, path = file_service.create_workspace(tmp_path)
        assert file_service.get_workspace(tmp_path, workspace_id) == path
        file_service.delete_workspace(tmp_path, workspace_id)
        assert not path.exists()

    @pytest.mark.parametrize("bad_id", ["../etc", "..", "abc", "/tmp", "0" * 31 + "/", None, ""])
    def test_invalid_ids_are_rejected(self, tmp_path, bad_id):
        assert file_service.get_workspace(tmp_path, bad_id) is None

    def test_cleanup_removes_only_expired(self, tmp_path):
        old_id, old_path = file_service.create_workspace(tmp_path)
        _, new_path = file_service.create_workspace(tmp_path)
        past = time.time() - 7200
        os.utime(old_path, (past, past))
        assert file_service.cleanup_expired_workspaces(tmp_path, 3600) == 1
        assert not old_path.exists()
        assert new_path.exists()

    def test_display_filename_strips_paths(self):
        assert file_service.display_filename("../../etc/passwd.csv") == "passwd.csv"
        assert file_service.display_filename("C:\\Users\\me\\data.csv") == "data.csv"
