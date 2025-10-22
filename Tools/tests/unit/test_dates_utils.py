"""
Unit tests for date parsing utilities.
"""

import sys
from pathlib import Path

# Ensure package src is on sys.path when running tests directly
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
from climaxtreme.utils import parse_mixed_date_column, add_date_parts  # type: ignore


def test_parse_mixed_date_column_basic():
    s = pd.Series(["1743-12-01", "3/1/1969", "01/03/1969", None, "invalid"])
    parsed = parse_mixed_date_column(s)

    # Expected: first is 1743-12-01, second interpreted as 1969-03-01 (M/D/Y),
    # third is parsed as 1969-01-03 because M/D/Y is attempted before D/M/Y
    assert pd.notna(parsed.iloc[0])
    assert parsed.iloc[0].year == 1743 and parsed.iloc[0].month == 12 and parsed.iloc[0].day == 1

    assert pd.notna(parsed.iloc[1])
    assert parsed.iloc[1].year == 1969 and parsed.iloc[1].month == 3 and parsed.iloc[1].day == 1

    assert pd.notna(parsed.iloc[2])
    assert parsed.iloc[2].year == 1969 and parsed.iloc[2].month == 1 and parsed.iloc[2].day == 3

    # None/invalid become NaT
    assert pd.isna(parsed.iloc[3])
    assert pd.isna(parsed.iloc[4])


def test_add_date_parts_creates_columns():
    df = pd.DataFrame({
        'dt': ["1743-12-01", "3/1/1969", "01/03/1969"],
        'temperature': [10.5, 12.3, 5.8]
    })

    out = add_date_parts(df)

    assert 'date' in out.columns
    assert 'year' in out.columns and 'month' in out.columns and 'day' in out.columns
    assert out.loc[0, 'year'] == 1743
    assert out.loc[1, 'month'] == 3
    assert out.loc[2, 'month'] == 1
    assert out.loc[2, 'day'] == 3
