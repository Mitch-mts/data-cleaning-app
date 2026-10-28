"""The fixed choices the cleaning form is allowed to send.

Each tuple is the complete list for that setting. A value that is not in the
list is rejected before any data is changed.
"""

TEXT_CASES = ("none", "lower", "upper", "title")
NUMERIC_FILL_STRATEGIES = ("none", "mean", "median")
TEXT_FILL_STRATEGIES = ("none", "value", "mode")
MISSING_ROW_ACTIONS = ("keep", "drop")
TARGET_TYPES = ("keep", "integer", "float", "string", "datetime")
CONVERSION_ERROR_ACTIONS = ("coerce", "skip")
SORT_DIRECTIONS = ("asc", "desc")

# Keys are the only math operations a computed column can use.
TRANSFORM_OPERATIONS = {
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
}

# How many bad values to show in a report. The rest are counted, not listed.
MAX_EXAMPLES = 10
