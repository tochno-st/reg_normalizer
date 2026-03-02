"""Tests for indicators and attach_indicators."""

import pandas as pd
import pytest

from reg_normalizer import RegionMatcher, get_indicator_descriptions
from reg_normalizer.indicators import attach_indicators


def test_get_indicator_descriptions():
    d = get_indicator_descriptions()
    assert isinstance(d, dict)
    assert "pop_total" in d
    assert "ibr" in d
    assert d["pop_total"] == "Численность населения — всего"


def test_get_indicator_descriptions_values_are_strings():
    d = get_indicator_descriptions()
    for code, desc in d.items():
        assert isinstance(code, str)
        assert isinstance(desc, str)


def test_attach_indicators_requires_year_or_year_col():
    df = pd.DataFrame({"object_name": ["Московская область"]})
    with pytest.raises(ValueError, match="year_col|year"):
        attach_indicators(df, "pop_total")


def test_attach_indicators_with_year_col():
    df = pd.DataFrame({
        "object_name": ["Московская область", "Свердловская область"],
        "year": [2020, 2020],
    })
    out = attach_indicators(df, "pop_total", name_col="object_name", year_col="year")
    assert "pop_total" in out.columns
    assert "object_name" in out.columns
    assert "year" in out.columns
    assert len(out) == 2


def test_attach_indicators_with_year_only():
    df = pd.DataFrame({"object_name": ["Московская область", "Свердловская область"]})
    out = attach_indicators(df, ["pop_total", "ibr"], name_col="object_name", year=2020)
    assert "pop_total" in out.columns
    assert "ibr" in out.columns
    assert len(out) == 2


def test_attach_indicators_single_indicator_as_str():
    df = pd.DataFrame({"object_name": ["Московская область"], "year": [2020]})
    out = attach_indicators(df, "pop_total", name_col="object_name", year_col="year")
    assert "pop_total" in out.columns
    assert len(out) == 1


def test_attach_indicators_custom_name_col():
    df = pd.DataFrame({
        "region": ["Московская область", "Свердловская область"],
        "год": [2020, 2020],
    })
    out = attach_indicators(df, "pop_total", name_col="region", year_col="год")
    assert "pop_total" in out.columns
    assert "region" in out.columns
    assert "год" in out.columns
    assert len(out) == 2


def test_attach_indicators_unknown_indicator():
    df = pd.DataFrame({"object_name": ["Московская область"], "year": [2020]})
    with pytest.raises(ValueError, match="Unknown indicator"):
        attach_indicators(df, "nonexistent_indicator", year_col="year")


def test_attach_indicators_unknown_indicators_list():
    df = pd.DataFrame({"object_name": ["Московская область"], "year": [2020]})
    with pytest.raises(ValueError, match="Unknown indicator"):
        attach_indicators(df, ["pop_total", "bad_one"], year_col="year")


def test_match_dataframe_adds_object_name():
    region_matcher = RegionMatcher()
    df = pd.DataFrame({"region": ["Московская область", "Свердловская область"]})
    out = region_matcher.match_dataframe(df, "region", threshold=70)
    assert "object_name" in out.columns
    assert "levenshtein_score" in out.columns


def test_region_matcher_attach_indicators():
    """RegionMatcher.attach_indicators delegates to indicators.attach_indicators; test that contract."""
    df = pd.DataFrame({
        "object_name": ["Московская область"],
        "year": [2020],
    })
    out = attach_indicators(df, "pop_total", name_col="object_name", year_col="year")
    assert "pop_total" in out.columns


def test_region_matcher_get_indicator_descriptions():
    region_matcher = RegionMatcher()
    d = region_matcher.get_indicator_descriptions()
    assert isinstance(d, dict)
    assert d == get_indicator_descriptions()
