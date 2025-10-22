"""
Utilities for robust parsing and normalization of date/time columns.

Handles mixed-format date strings like:
- ISO: "1743-12-01"
- Slash formats: "3/1/1969" (month/day/year) or "01/03/1969" (day/month/year)

The main entry point is `parse_mixed_date_column`, which returns a pandas
Series of dtype datetime64[ns] suitable for downstream modeling.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def parse_mixed_date_column(
    series: pd.Series,
    *,
    prefer_dayfirst: Optional[bool] = None,
) -> pd.Series:
    """
    Parse a pandas Series containing dates in mixed formats into datetime.

    Strategy:
    1) First try strict ISO format (YYYY-MM-DD), which is unambiguous and fast.
    2) For remaining values, try pandas' general parser with dayfirst=False.
    3) For any still-unparsed values, try again with dayfirst=True.

    If `prefer_dayfirst` is provided, we'll bias step 2 accordingly by trying
    that option first, then the opposite for the fallback.

    Args:
        series: Pandas Series of strings/object date-like values.
        prefer_dayfirst: If True, prefer D/M/Y interpretations first; if False,
            prefer M/D/Y first; if None, default to M/D/Y first (common in many
            global CSV exports found online). Regardless, both paths are tried.

    Returns:
        A pandas Series of dtype datetime64[ns] with invalid/unknown values as NaT.
    """
    if series.empty:
        return pd.to_datetime(series, errors="coerce")

    # Ensure object/string dtype for parsing logic
    s = series.astype("string")

    # Step 1: try strict ISO first
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    # Identify which entries are still NaT
    mask_unparsed = parsed.isna()
    if mask_unparsed.any():
        remaining = s[mask_unparsed]

        # Decide order based on preference
        try_orders = []
        if prefer_dayfirst is None:
            try_orders = [False, True]  # Try M/D/Y first, then D/M/Y
        else:
            try_orders = [prefer_dayfirst, not prefer_dayfirst]

        for df_flag in try_orders:
            if not remaining.empty:
                parsed_remaining = pd.to_datetime(
                    remaining, errors="coerce", dayfirst=df_flag
                )
                # Fill only where we still have NaT
                fill_mask = parsed_remaining.notna()
                if fill_mask.any():
                    parsed.loc[fill_mask.index] = parsed_remaining[fill_mask]
                    remaining = remaining[~fill_mask]

    return parsed


def add_date_parts(
    df: pd.DataFrame,
    date_col: str = "dt",
    *,
    prefer_dayfirst: Optional[bool] = None,
    drop_invalid: bool = False,
    in_place: bool = False,
) -> pd.DataFrame:
    """
    Ensure a mixed-format date column is parsed and add standard date parts.

    Adds columns: `date` (datetime64[ns]), `year`, `month`, `day`.

    Args:
        df: Input DataFrame containing a date column (default name 'dt').
        date_col: Name of the column to parse.
        prefer_dayfirst: See `parse_mixed_date_column`.
        drop_invalid: If True, drop rows where the date could not be parsed.

    Returns:
        DataFrame with added/updated columns. If `in_place=False` (default),
        the original df is not modified. If `in_place=True`, the input df is
        modified and returned to avoid duplicating memory for large datasets.
    """
    if date_col not in df.columns:
        return df if in_place else df.copy()

    # Choose working frame
    out = df if in_place else df.copy()

    parsed = parse_mixed_date_column(out[date_col], prefer_dayfirst=prefer_dayfirst)
    out["date"] = parsed

    if drop_invalid:
        # When in-place, filter in place
        out = out[out["date"].notna()].copy()

    # Add date parts for modeling and analysis
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day

    return out
