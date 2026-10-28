# Data Cleaning App

A small Flask web app for cleaning CSV and Excel files. Upload a file, review its quality, pick
cleaning options, compare before and after, and download the cleaned data plus a JSON report
of every change.

Built with Flask, pandas, NumPy, openpyxl, Bootstrap 5 and pytest. No database, no login,
no JavaScript framework.

## Features

- **Upload** `.csv` and `.xlsx` files (max 10 MB) with validation of type, size and content.
- **Overview**: row and column counts, data types, missing values, duplicate rows, unique
  counts, min/max/mean/median, mixed-type warnings, a 15-row preview, and a choice of which
  columns to clean.
- **Cleaning options** (only what you tick is applied):
  1. Text: trim, change capitalisation, blank to missing
  2. Data types: integer, decimal, text or datetime, with invalid values reported
  3. Missing values: drop columns above a % threshold, fill numbers with mean/median, fill
     text with a value or the most frequent value, drop remaining rows
  4. Duplicates: full row or selected columns
  5. Outliers: IQR detection, reported only unless you explicitly tick removal
  6. Filter by numeric range or exact text, sort by one or two columns
  7. New computed column: add, subtract, multiply or divide by a constant
  8. Column names: trim, lower case, underscores, with duplicate names made unique
- **Results**: before/after table, list of operations, conversion failures, outliers and a
  preview of the cleaned data.
- **Downloads**: cleaned data as CSV or Excel, and the cleaning report as JSON.

## Project structure

```text
Data-Cleaning-App/
├── app.py                       # Flask app factory and routes
├── services/
│   ├── file_service.py          # upload validation, reading, exporting, temp workspaces
│   ├── profiling_service.py     # dataset statistics, mixed-type warnings, previews
│   ├── cleaning_service.py      # public imports; the operations live in cleaning/
│   ├── cleaning/
│   │   ├── pipeline.py          # runs the selected operations in order
│   │   ├── text.py  missing.py  duplicates.py  types.py
│   │   ├── outliers.py  filters.py  transform.py  names.py
│   ├── options_service.py       # server-side validation of the cleaning form
│   └── report_service.py        # before/after summary and JSON report
├── templates/
│   ├── base.html  _macros.html  index.html  overview.html
│   ├── clean.html results.html  error.html
├── static/
│   ├── css/style.css
│   └── js/app.js                # loading spinner, confirm dialog, select all/none
├── sample_data/messy_customers.csv
├── tests/                       # pytest suite
├── requirements.txt
└── pytest.ini
```

## Setup

Requires **Python 3.11 or newer**. Check with `python3 --version` (Mac) or `py --version` (Windows).

### macOS / Linux

```bash
cd Data-Cleaning-App
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd Data-Cleaning-App
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt with
`.venv\Scripts\activate.bat`.

## Run the tests

With the virtual environment activated (any OS):

```bash
python -m pytest
```

## Start the server

With the virtual environment activated:

```bash
flask --app app run
```

Then open <http://127.0.0.1:5000>. Add `--debug` for auto-reload while developing, or
`--port 5051` if port 5000 is busy. On macOS, AirPlay Receiver often uses port 5000.

To try it out, upload `sample_data/messy_customers.csv`. It contains inconsistent column
names, blank and whitespace-only values, invalid numbers (`thirty`, `abc`), malformed dates
(`2024-13-45`, `not a date`), duplicate rows, an outlier (`99999`) and a formula-like value.

### Configuration (optional)

| Environment variable | Purpose |
|---|---|
| `SECRET_KEY` | Signs the session cookie. If unset, a random key is generated at startup, so sessions reset when the server restarts. |

The upload limit (10 MB), preview size and temp-file lifetime (1 hour) are set in
`create_app()` in `app.py`.

## How it works

### Upload

1. The browser posts the file to `/upload` with a CSRF token.
2. `file_service.validate_upload` checks the extension, that the file isn't empty, the size,
   and the content (a `.xlsx` must be a ZIP archive, a `.csv` must be text).
3. `file_service.load_dataset` parses it. CSV encoding (UTF-8 or Windows-1252) and the
   delimiter (`,` `;` tab `|`) are detected. For Excel, only cached cell values from the first
   sheet are read, and formulas are never evaluated.
4. The original bytes are saved as `original.csv` or `original.xlsx` in a new private temp
   folder named with a random 32-character ID. Only that ID is kept in the signed session
   cookie. The user's filename is shown on screen but never used as a path.

### Overview

`profiling_service.profile_dataset` computes the statistics on each visit from the stored
original. Ticking columns and clicking **Continue** saves the selection, which limits text
cleaning, type conversion, outliers and missing-value handling to those columns.

### Cleaning

1. `options_service.parse_cleaning_options` validates every form field on the server:
   column names must exist, choices must come from fixed lists, and numbers are range-checked.
   All errors are shown at once.
2. `cleaning_service.run_cleaning_pipeline` applies only the selected operations in a fixed
   order: text, data types, missing values, duplicates, outliers, filters, computed column,
   sorting, column names. That checklist is `services/cleaning/pipeline.py`; each operation
   is its own file beside it. Renaming runs last so every option can refer to the original
   column names.
3. Each operation returns a new DataFrame plus a `StepRecord` saying what changed. If one
   step fails (for example, sorting a column that mixes numbers and text), its error is
   reported and the remaining steps still run.
4. `report_service.build_report` builds the before/after summary, the operations performed,
   affected columns, rows removed, missing values filled, conversion failures and outliers.
   The submitted options and the report are saved as JSON in the workspace.

The original file is never modified, so you can go back and clean again with different
options.

### Download

Downloads re-run the pipeline from the original file and the saved options, so no cleaned
copy is kept on disk. The result is streamed from memory as CSV (UTF-8 with BOM, so Excel
reads it correctly) or `.xlsx`. Text cells starting with `=`, `+`, `-` or `@` get a leading
`'` so spreadsheet apps don't run them as formulas. Plain negative numbers are left alone.
The report is served as JSON.

## Security notes

- No `eval`, `exec`, pickle or formula evaluation. Computed columns use a fixed set of four
  arithmetic operations.
- CSRF token on every POST. Jinja auto-escaping is on and user values are never marked safe.
  A Content-Security-Policy and other security headers are sent with every response.
- Workspace IDs are validated against `^[0-9a-f]{32}$`, which rules out path traversal.
- Uploaded files live in the OS temp folder with `0700` permissions. They are deleted when
  you click **Start over** or upload a new file, and after one hour of inactivity.
- Logs record file type, size and shape only, never cell values. Full tracebacks are
  logged only in debug mode because exception messages can quote data.

## Limitations

- Only the first sheet of an Excel workbook is read. Legacy `.xls` files are not supported.
- Excel exports are limited to 1,048,575 rows. Use CSV for larger results.
- A row with no values at all can't be stored in `.xlsx`, so it's dropped from Excel exports.
- This is a development server setup. For production, run it behind a WSGI server
  (for example `gunicorn "app:create_app()"`) with a fixed `SECRET_KEY` and a single worker,
  or shared temp storage.
