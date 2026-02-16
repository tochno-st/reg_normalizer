"""Indicator descriptions and attach_indicators merge logic for normalized region data."""

import os
import yaml
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_indicators_yaml_path = os.path.join(_script_dir, "data", "interim", "indicators_descriptions.yaml")
_normalizers_csv_path = os.path.join(_script_dir, "data", "interim", "normalizers.csv")

_normalizers_cache = None


def _read_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_indicator_descriptions() -> dict:
    """Return indicator code -> Russian description mapping.

    Loads from indicators_descriptions.yaml in package data.
    Returns dict[str, str] (e.g. 'pop_total' -> 'Численность населения — всего').
    """
    return _read_yaml(_indicators_yaml_path) or {}


def _load_normalizers() -> pd.DataFrame:
    global _normalizers_cache
    if _normalizers_cache is None:
        _normalizers_cache = pd.read_csv(_normalizers_csv_path, sep=";", dtype={"year": "Int64"})
    return _normalizers_cache


def attach_indicators(
    df: pd.DataFrame,
    indicators,
    name_col: str = "object_name",
    year_col: str = None,
    year: int = None,
    how: str = "left",
) -> pd.DataFrame:
    """Attach one or more indicator columns from normalizers by region (and optionally year).

    Three merge scenarios:
    1. With year in data: set year_col → merge on (name_col, year_col) ↔ (object_name, year).
    2. Without year in data: set year → take values for that year only, merge on name_col ↔ object_name.
    3. Either way, indicators can be a single code (str) or list of codes.

    Args:
        df: DataFrame with normalized region names in name_col.
        indicators: One indicator code (str) or list of codes (e.g. 'pop_total' or ['pop_total', 'ibr']).
        name_col: Column with normalized region name. Default 'object_name'.
        year_col: Column with year in df; if set, merge on (name_col, year_col).
        year: Single year when df has no year column; required if year_col is not set.
        how: Merge type ('left' or 'outer'). Default 'left'.

    Returns:
        DataFrame with requested indicator columns added.

    Raises:
        ValueError: If neither year_col nor year is provided, or indicators are invalid.
    """
    if year_col is None and year is None:
        raise ValueError("Either year_col (column name in df) or year (single year) must be provided.")

    if isinstance(indicators, str):
        indicators = [indicators]

    norm = _load_normalizers()
    available = [c for c in norm.columns if c not in ("object_name", "object_level", "oktmo", "year")]
    missing = [i for i in indicators if i not in available]
    if missing:
        raise ValueError(f"Unknown indicator(s): {missing}. Available: {available}")

    cols = ["object_name", "year"] + list(indicators)
    right = norm[cols].copy()

    if year_col is not None:
        # Merge on (name_col, year_col) <-> (object_name, year)
        left_on = [name_col, year_col]
        right_on = ["object_name", "year"]
    else:
        # Merge by year only: filter normalizers to that year, then merge on name
        right = right[right["year"] == year].drop(columns=["year"])
        right_on = ["object_name"]
        left_on = [name_col]

    df = df.merge(right, left_on=left_on, right_on=right_on, how=how)
    return df
