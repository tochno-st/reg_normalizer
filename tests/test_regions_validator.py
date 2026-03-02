"""Tests for RegionMatcher, preprocess_name, stem_region_name, match_dataframe, attach_fields."""

import pandas as pd
import pytest

from reg_normalizer.regions_validator import RegionMatcher, read_yaml


# --- preprocess_name (static) ---


def test_preprocess_name_latin_to_cyrillic():
    assert RegionMatcher.preprocess_name("Mосковская область") == "московская область"


def test_preprocess_name_dashes_to_spaces():
    assert RegionMatcher.preprocess_name("Ханты–Мансийский — округ") == "ханты мансийский округ"


def test_preprocess_name_lowercase_and_whitespace():
    assert RegionMatcher.preprocess_name("  Московская   Область  ") == "московская область"


def test_preprocess_name_extra_data_removed():
    name = "Тюменская область без учета новых субъектов (с 01.01.2023)"
    out = RegionMatcher.preprocess_name(name)
    assert "без учета новых субъектов" not in out
    assert "с 01.01.2023" not in out
    assert "тюменская область" in out


def test_preprocess_name_non_string_returns_empty():
    assert RegionMatcher.preprocess_name(123) == ""
    assert RegionMatcher.preprocess_name(None) == ""


# --- stem_region_name (static) ---


def test_stem_region_name_basic():
    out = RegionMatcher.stem_region_name("московская область")
    assert isinstance(out, str)
    assert len(out) > 0
    # Snowball stemmer reduces words
    assert "московск" in out or "област" in out


def test_stem_region_name_empty_returns_empty():
    assert RegionMatcher.stem_region_name("") == ""


# --- RegionMatcher init and custom etalon/abbreviations ---


def test_region_matcher_default_etalon():
    matcher = RegionMatcher()
    assert len(matcher.etalon) > 0
    assert "Московская область" in matcher.etalon or any(
        "Москва" in r or "москов" in r.lower() for r in matcher.etalon
    )


def test_region_matcher_custom_etalon():
    custom = ["Москва", "Санкт-Петербург", "Татарстан"]
    matcher = RegionMatcher(etalon_regions=custom)
    assert matcher.etalon == custom
    assert len(matcher.preprocessed_etalon) == 3


def test_region_matcher_custom_abbreviations():
    custom_abbr = {"мск": "Москва", "спб": "Санкт-Петербург"}
    matcher = RegionMatcher(etalon_regions=["Москва", "Санкт-Петербург"], abbreviations=custom_abbr)
    assert matcher.abbreviations == custom_abbr


# --- find_best_match ---


def test_find_best_match_exact_returns_high_score():
    matcher = RegionMatcher(etalon_regions=["Московская область", "Свердловская область"])
    match, score = matcher.find_best_match("Московская область", threshold=0)
    assert match == "Московская область"
    assert score is not None
    assert score >= 90


def test_find_best_match_below_threshold_returns_none_score():
    matcher = RegionMatcher(etalon_regions=["Московская область"])
    match, score = matcher.find_best_match("XyzUnknownRegion123", threshold=99)
    assert score is None
    assert match is not None or score is None


def test_find_best_match_abbreviation_resolved():
    matcher = RegionMatcher(
        etalon_regions=["Санкт-Петербург"],
        abbreviations={"спб": "Санкт-Петербург"},
    )
    match, score = matcher.find_best_match("спб", threshold=50)
    assert match == "Санкт-Петербург"
    assert score is not None


def test_find_best_match_custom_weights():
    matcher = RegionMatcher(etalon_regions=["Московская область"])
    match1, s1 = matcher.find_best_match(
        "московская область",
        weights={"levenshtein": 1.0, "token_set": 0.0},
        threshold=0,
    )
    match2, s2 = matcher.find_best_match(
        "московская область",
        weights={"levenshtein": 0.0, "token_set": 1.0},
        threshold=0,
    )
    assert match1 == "Московская область"
    assert match2 == "Московская область"


# --- match_dataframe ---


def test_match_dataframe_adds_object_name_and_levenshtein_score():
    matcher = RegionMatcher(etalon_regions=["Московская область", "Свердловская область"])
    df = pd.DataFrame({"region": ["Московская область", "Свердловская область", "Московская область"]})
    out = matcher.match_dataframe(df, "region", threshold=70)
    assert "object_name" in out.columns
    assert "levenshtein_score" in out.columns
    assert len(out) == 3
    assert out["region"].tolist() == ["Московская область", "Свердловская область", "Московская область"]


def test_match_dataframe_unique_values_mapped_once():
    matcher = RegionMatcher(etalon_regions=["Московская область"])
    df = pd.DataFrame({"region": ["Московская область"] * 5})
    out = matcher.match_dataframe(df, "region", threshold=70)
    assert out["object_name"].nunique() == 1
    assert out["object_name"].iloc[0] == "Московская область"


# --- attach_fields ---


def test_attach_fields_adds_requested_columns():
    matcher = RegionMatcher(etalon_regions=["Белгородская область"])
    df = pd.DataFrame({"region": ["Белгородская область"]})
    out = matcher.attach_fields(df, "region", ["name_eng", "okato", "iso_code"], threshold=70)
    assert "name_eng" in out.columns
    assert "okato" in out.columns
    assert "iso_code" in out.columns
    assert out["name_eng"].iloc[0] == "Belgorod"
    assert out["okato"].iloc[0] == "14000000"
    assert out["iso_code"].iloc[0] == "RU-BEL"


def test_attach_fields_single_field():
    matcher = RegionMatcher(etalon_regions=["Москва"])
    df = pd.DataFrame({"region": ["Москва"]})
    out = matcher.attach_fields(df, "region", ["name_eng"], threshold=70)
    assert "name_eng" in out.columns
    assert out["name_eng"].iloc[0] == "Moskva (city)"


def test_attach_fields_no_match_sets_none():
    matcher = RegionMatcher(etalon_regions=["Московская область"])
    df = pd.DataFrame({"region": ["НесуществующийРегионХхх"]})
    out = matcher.attach_fields(df, "region", ["name_eng", "okato"], threshold=99)
    assert out["name_eng"].isna().all()
    assert out["okato"].isna().all()


# --- read_yaml ---


def test_read_yaml_returns_dict():
    import os
    from reg_normalizer.regions_validator import yaml_path
    assert os.path.isfile(yaml_path)
    data = read_yaml(yaml_path)
    assert isinstance(data, dict)
    assert "dict" in data
    assert "version" in data or "title" in data


# --- edge cases ---
# TODO:
# Камчатская (область) — не распознает как край
# Ханты-Мансийский ФО — не распознает, надо править
# Объединенные регионы (Москва, Московская область) — проверить