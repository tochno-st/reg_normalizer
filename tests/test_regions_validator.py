"""Tests for RegionMatcher, preprocess_name, stem_region_name, match_dataframe, attach_fields."""

import pandas as pd
import pytest

from reg_normalizer.regions_validator import RegionMatcher, read_yaml, ETALON_REGIONS


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


def test_preprocess_name_empty_string():
    assert RegionMatcher.preprocess_name("") == ""


def test_preprocess_name_multiple_extra_data():
    """Multiple EXTRA_DATA phrases should all be removed."""
    name = "Тюменская область в границах (с 2010 года)"
    out = RegionMatcher.preprocess_name(name)
    assert "в границах" not in out
    assert "с 2010 года" not in out
    assert "тюменская область" in out


def test_preprocess_name_footnotes_no_space():
    """Footnote markers directly after name should be removed."""
    assert RegionMatcher.preprocess_name('Московская область1);2)') == 'московская область'
    assert RegionMatcher.preprocess_name('Московская область2);3)') == 'московская область'
    assert RegionMatcher.preprocess_name('Московская область3);4)') == 'московская область'


def test_preprocess_name_footnotes_with_space():
    """Footnote markers with leading space should be removed."""
    assert RegionMatcher.preprocess_name('Московская область 1);2)') == 'московская область'
    assert RegionMatcher.preprocess_name('Южный федеральный округ 2);3)') == 'южный федеральный округ'


def test_preprocess_name_footnotes_comma_separator():
    """Footnote markers with comma separator should be removed."""
    assert RegionMatcher.preprocess_name('Московская область1),2)') == 'московская область'


def test_preprocess_name_units_mln():
    """Units after comma (млн) should be removed."""
    assert RegionMatcher.preprocess_name('Российская Федерация, млн плотных м3') == 'российская федерация'
    assert RegionMatcher.preprocess_name('Российская Федерация, млн га') == 'российская федерация'
    assert RegionMatcher.preprocess_name('Российская Федерация, млн т') == 'российская федерация'


def test_preprocess_name_units_mlrd():
    """Units after comma (млрд) should be removed."""
    assert RegionMatcher.preprocess_name('Российская Федерация, млрд. руб.') == 'российская федерация'
    assert RegionMatcher.preprocess_name('Российская Федерация, млрд руб.') == 'российская федерация'


def test_preprocess_name_units_with_latin_chars():
    """Units removal should work even after Latin→Cyrillic conversion."""
    # Latin 'p' in Федерация → Cyrillic 'р' after conversion
    assert RegionMatcher.preprocess_name('Российская Федеpация, млрд руб.') == 'российская федерация'


def test_preprocess_name_resp_prefix():
    """'Респ. X' prefix should become 'республика X'."""
    assert RegionMatcher.preprocess_name('Респ. Татарстан') == 'республика татарстан'
    assert RegionMatcher.preprocess_name('Респ. Адыгея') == 'республика адыгея'
    assert RegionMatcher.preprocess_name('Респ. Бурятия') == 'республика бурятия'


def test_preprocess_name_respubl_prefix():
    """'Республ. X' prefix should become 'республика X'."""
    assert RegionMatcher.preprocess_name('Республ. Дагестан') == 'республика дагестан'
    assert RegionMatcher.preprocess_name('Республ. Карелия') == 'республика карелия'


def test_preprocess_name_resp_suffix():
    """'X Респ.' suffix should become 'республика X'."""
    assert RegionMatcher.preprocess_name('Татарстан Респ.') == 'республика татарстан'
    assert RegionMatcher.preprocess_name('Адыгея Респ.') == 'республика адыгея'
    assert RegionMatcher.preprocess_name('Башкортостан Респ.') == 'республика башкортостан'


def test_preprocess_name_respubl_suffix():
    """'X Республ.' suffix should become 'республика X'."""
    assert RegionMatcher.preprocess_name('Дагестан Республ.') == 'республика дагестан'


# --- stem_region_name (static) ---


def test_stem_region_name_basic():
    out = RegionMatcher.stem_region_name("московская область")
    assert isinstance(out, str)
    assert len(out) > 0
    assert "московск" in out or "област" in out


def test_stem_region_name_empty_returns_empty():
    assert RegionMatcher.stem_region_name("") == ""


def test_stem_region_name_single_word():
    out = RegionMatcher.stem_region_name("москва")
    assert isinstance(out, str)
    assert len(out) > 0


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


def test_region_matcher_preprocessed_etalon_format():
    """Each element of preprocessed_etalon should be a tuple (original, preprocessed, stemmed)."""
    matcher = RegionMatcher(etalon_regions=["Москва", "Санкт-Петербург"])
    for item in matcher.preprocessed_etalon:
        assert len(item) == 3
        original, preprocessed, stemmed = item
        assert isinstance(original, str)
        assert isinstance(preprocessed, str)
        assert isinstance(stemmed, str)


# --- etalon data integrity ---


def test_etalon_contains_renamed_regions():
    """Etalon should contain the renamed Архангельская and Тюменская regions."""
    assert "Архангельская область (с автономным округом)" in ETALON_REGIONS
    assert "Архангельская область (без автономного округа)" in ETALON_REGIONS
    assert "Тюменская область (с автономными округами)" in ETALON_REGIONS
    assert "Тюменская область (без автономных округов)" in ETALON_REGIONS


def test_etalon_contains_autonomous_okrugs():
    """Etalon should contain autonomous okrugs as separate entries."""
    assert "Ненецкий автономный округ" in ETALON_REGIONS
    assert "Ханты-Мансийский автономный округ — Югра" in ETALON_REGIONS
    assert "Ямало-Ненецкий автономный округ" in ETALON_REGIONS


def test_etalon_no_old_names():
    """Old names without '(с ...) / (без ...)' should NOT be in etalon."""
    # Проверяем что нет голых "Архангельская область" и "Тюменская область" без скобок
    for name in ETALON_REGIONS:
        if "Архангельская область" in name:
            assert "(" in name, f"Unexpected bare name: {name}"
        if "Тюменская область" in name:
            assert "(" in name, f"Unexpected bare name: {name}"


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


# --- _find_best_match_core ---


def test_find_best_match_core_same_as_find_best_match_for_simple_input():
    """_find_best_match_core should return same result as find_best_match for non-compound input."""
    matcher = RegionMatcher(etalon_regions=["Московская область", "Свердловская область"])
    core_result = matcher._find_best_match_core("Московская область", threshold=0)
    full_result = matcher.find_best_match("Московская область", threshold=0)
    assert core_result == full_result


# --- abbreviations: Архангельская, Тюменская, Камчатская ---


class TestAbbreviations:
    """Tests for abbreviation resolution to correct canonical names."""

    @pytest.fixture
    def matcher(self):
        return RegionMatcher()

    def test_arkhangelsk_plain_resolves_with_ao(self, matcher):
        """'Архангельская область' without qualifier -> с автономным округом."""
        match, score = matcher.find_best_match("Архангельская область")
        assert match == "Архангельская область (с автономным округом)"
        assert score is not None

    def test_tyumen_plain_resolves_with_ao(self, matcher):
        """'Тюменская область' without qualifier -> с автономными округами."""
        match, score = matcher.find_best_match("Тюменская область")
        assert match == "Тюменская область (с автономными округами)"
        assert score is not None

    def test_arkhangelsk_bez_ao(self, matcher):
        """'Архангельская область (без АО)' -> без автономного округа."""
        match, score = matcher.find_best_match("Архангельская область (без АО)")
        assert match == "Архангельская область (без автономного округа)"
        assert score is not None

    def test_tyumen_bez_ao(self, matcher):
        """'Тюменская область (без АО)' -> без автономных округов."""
        match, score = matcher.find_best_match("Тюменская область (без АО)")
        assert match == "Тюменская область (без автономных округов)"
        assert score is not None

    def test_tyumen_krome_long_form(self, matcher):
        """Long parenthetical exclusion form should resolve correctly."""
        match, score = matcher.find_best_match(
            "Тюменская область (кроме Ханты-Мансийского автономного округа-Югры "
            "и Ямало-Ненецкого автономного округа)"
        )
        assert match == "Тюменская область (без автономных округов)"
        assert score is not None

    def test_arkhangelsk_krome(self, matcher):
        """'Архангельская область (кроме Ненецкого АО)' -> без АО."""
        match, score = matcher.find_best_match(
            "Архангельская область (кроме Ненецкого автономного округа)"
        )
        assert match == "Архангельская область (без автономного округа)"
        assert score is not None

    def test_kamchatskaya_oblast_resolved_to_kray(self, matcher):
        """Камчатская область is a common error, should resolve to Камчатский край."""
        match, score = matcher.find_best_match("Камчатская область", threshold=50)
        assert match == "Камчатский край"
        assert score is not None

    def test_khmao_abbreviation(self, matcher):
        """ХМАО abbreviation should resolve."""
        match, score = matcher.find_best_match("ХМАО", threshold=50)
        assert "Ханты-Мансийский" in match
        assert score is not None

    def test_yanao_abbreviation(self, matcher):
        """ЯНАО abbreviation should resolve."""
        match, score = matcher.find_best_match("ЯНАО", threshold=50)
        assert match == "Ямало-Ненецкий автономный округ"
        assert score is not None

    def test_nao_abbreviation(self, matcher):
        """НАО abbreviation should resolve."""
        match, score = matcher.find_best_match("НАО", threshold=50)
        assert match == "Ненецкий автономный округ"
        assert score is not None

    def test_resp_prefix_matches(self, matcher):
        """'Респ. X' should match the corresponding republic."""
        match, score = matcher.find_best_match('Респ. Татарстан')
        assert match == 'Республика Татарстан'
        assert score is not None

    def test_resp_suffix_matches(self, matcher):
        """'X Респ.' should match the corresponding republic."""
        match, score = matcher.find_best_match('Татарстан Респ.')
        assert match == 'Республика Татарстан'
        assert score is not None

    def test_resp_various_republics(self, matcher):
        """Респ. abbreviation should work for multiple republics."""
        cases = [
            ('Респ. Адыгея', 'Республика Адыгея'),
            ('Респ. Бурятия', 'Республика Бурятия'),
            ('Башкортостан Респ.', 'Республика Башкортостан'),
        ]
        for input_name, expected in cases:
            match, score = matcher.find_best_match(input_name)
            assert match == expected, f"Expected '{expected}' for '{input_name}', got '{match}'"
            assert score is not None


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


def test_match_dataframe_no_match_gives_zero_score():
    """Unmatched rows should have levenshtein_score == 0."""
    matcher = RegionMatcher(etalon_regions=["Московская область"])
    df = pd.DataFrame({"region": ["АбсолютноНеизвестныйРегион"]})
    out = matcher.match_dataframe(df, "region", threshold=99)
    assert out["levenshtein_score"].iloc[0] == 0


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


def test_attach_fields_with_resolve_parent_regions():
    """attach_fields should also apply _resolve_parent_regions."""
    matcher = RegionMatcher()
    df = pd.DataFrame({
        "region": [
            "Архангельская область",
            "Ненецкий автономный округ",
        ]
    })
    out = matcher.attach_fields(df, "region", ["name_eng"], threshold=70)
    # Архангельская should be resolved to "без АО" because НАО is present
    assert out["name_eng"].iloc[0] == "Arkhangelskaya without Nenets Autonomous Okrug"


# --- read_yaml ---


def test_read_yaml_returns_dict():
    import os
    from reg_normalizer.regions_validator import yaml_path
    assert os.path.isfile(yaml_path)
    data = read_yaml(yaml_path)
    assert isinstance(data, dict)
    assert "dict" in data
    assert "version" in data or "title" in data


# --- compound region handling ---


class TestCompoundRegions:
    """Tests for compound region strings containing 'и', ',' or ';'."""

    @pytest.fixture
    def matcher(self):
        return RegionMatcher()

    def test_compound_moscow_returns_none(self, matcher):
        """город Москва и Московская область -> no match."""
        match, score = matcher.find_best_match("город Москва и Московская область")
        assert match is None
        assert score is None

    def test_compound_spb_returns_none(self, matcher):
        """Санкт-Петербург и Ленинградская область -> no match."""
        match, score = matcher.find_best_match(
            "город Санкт-Петербург и Ленинградская область"
        )
        assert match is None
        assert score is None

    def test_compound_arkhangelsk_with_nenets(self, matcher):
        """Архангельская область и Ненецкий АО -> Архангельская область (с автономным округом)."""
        match, score = matcher.find_best_match(
            "Архангельская область и Ненецкий автономный округ"
        )
        assert match == "Архангельская область (с автономным округом)"
        assert score is not None
        assert score >= 70

    def test_compound_arkhangelsk_nenets_reversed_order(self, matcher):
        """Order should not matter for exception rules."""
        match, score = matcher.find_best_match(
            "Ненецкий автономный округ и Архангельская область"
        )
        assert match == "Архангельская область (с автономным округом)"

    def test_compound_tyumen_with_khmao(self, matcher):
        """Тюменская область и ХМАО -> Тюменская область (с автономными округами)."""
        match, score = matcher.find_best_match(
            "Тюменская область и Ханты-Мансийский автономный округ — Югра"
        )
        assert match == "Тюменская область (с автономными округами)"

    def test_compound_tyumen_with_yanao(self, matcher):
        """Тюменская область и ЯНАО -> Тюменская область (с автономными округами)."""
        match, score = matcher.find_best_match(
            "Тюменская область и Ямало-Ненецкий автономный округ"
        )
        assert match == "Тюменская область (с автономными округами)"

    def test_compound_tyumen_with_both_okrugs(self, matcher):
        """Тюменская with both KHMAO and YANAO -> Тюменская область (с автономными округами)."""
        match, score = matcher.find_best_match(
            "Тюменская область, Ханты-Мансийский автономный округ и Ямало-Ненецкий автономный округ"
        )
        assert match == "Тюменская область (с автономными округами)"

    def test_no_false_split_on_region_with_и(self, matcher):
        """'и' inside region names should not trigger compound logic."""
        match, score = matcher.find_best_match("Республика Марий Эл", threshold=50)
        assert match == "Республика Марий Эл"

    def test_compound_arbitrary_two_regions_returns_none(self, matcher):
        """Two arbitrary regions joined by 'и' -> no match."""
        match, score = matcher.find_best_match(
            "Краснодарский край и Ростовская область"
        )
        assert match is None
        assert score is None

    def test_compound_with_semicolon(self, matcher):
        """Two regions joined by ';' -> no match."""
        match, score = matcher.find_best_match(
            "Московская область; Свердловская область"
        )
        assert match is None
        assert score is None

    def test_compound_with_comma(self, matcher):
        """Two regions joined by ',' -> no match."""
        match, score = matcher.find_best_match(
            "Московская область, Свердловская область"
        )
        assert match is None
        assert score is None

    def test_compound_abbreviation_not_split(self, matcher):
        """Input matching full abbreviation should NOT be treated as compound."""
        match, score = matcher.find_best_match(
            "тюменская область (кроме ханты мансийского автономного округа югры "
            "и ямало ненецкого автономного округа)",
            threshold=50,
        )
        assert match == "Тюменская область (без автономных округов)"
        assert score is not None


# --- post-analysis: resolve parent regions ---


class TestResolveParentRegions:
    """Tests for _resolve_parent_regions post-analysis on DataFrames."""

    @pytest.fixture
    def matcher(self):
        return RegionMatcher()

    def test_arkhangelsk_with_nenets_in_dataset(self, matcher):
        """If НАО is separate in dataset, Архангельская -> без АО."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область",
                "Ненецкий автономный округ",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Архангельская область (без автономного округа)"
        assert result["object_name"].iloc[1] == "Ненецкий автономный округ"
        assert result["object_name"].iloc[2] == "Москва"

    def test_arkhangelsk_without_nenets_in_dataset(self, matcher):
        """If НАО is NOT in dataset, Архангельская stays as с АО."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Архангельская область (с автономным округом)"
        assert result["object_name"].iloc[1] == "Москва"

    def test_tyumen_with_okrugs_in_dataset(self, matcher):
        """If ХМАО or ЯНАО is separate in dataset, Тюменская -> без АО."""
        df = pd.DataFrame({
            "region": [
                "Тюменская область",
                "Ханты-Мансийский автономный округ",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Тюменская область (без автономных округов)"

    def test_tyumen_with_yanao_in_dataset(self, matcher):
        """If ЯНАО is separate in dataset, Тюменская -> без АО."""
        df = pd.DataFrame({
            "region": [
                "Тюменская область",
                "Ямало-Ненецкий автономный округ",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Тюменская область (без автономных округов)"

    def test_tyumen_with_both_okrugs_in_dataset(self, matcher):
        """If both ХМАО and ЯНАО are in dataset, Тюменская -> без АО."""
        df = pd.DataFrame({
            "region": [
                "Тюменская область",
                "Ханты-Мансийский автономный округ",
                "ЯНАО",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Тюменская область (без автономных округов)"

    def test_tyumen_without_okrugs_in_dataset(self, matcher):
        """If no okrugs in dataset, Тюменская stays as с АО."""
        df = pd.DataFrame({
            "region": [
                "Тюменская область",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Тюменская область (с автономными округами)"

    def test_compound_in_dataframe(self, matcher):
        """Compound handling propagates through match_dataframe."""
        df = pd.DataFrame({
            "region": [
                "город Москва и Московская область",
                "Архангельская область и Ненецкий автономный округ",
                "Московская область",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert pd.isna(result["object_name"].iloc[0])
        assert result["object_name"].iloc[1] == "Архангельская область (с автономным округом)"
        assert result["object_name"].iloc[2] == "Московская область"

    def test_resolve_logs_parent_resolved_when_children_present(self, matcher):
        """Should log parent_resolved event when parent region is redefined."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область",
                "Ненецкий автономный округ",
            ]
        })
        matcher.match_dataframe(df, "region", threshold=70)
        log = matcher.get_match()
        resolved = log[log["event"] == "parent_resolved"]
        assert len(resolved) == 1
        assert "переопределена" in resolved.iloc[0]["note"] or "→" in resolved.iloc[0]["note"]
        assert "Ненецкий" in resolved.iloc[0]["note"]

    def test_resolve_logs_parent_kept_when_no_children(self, matcher):
        """Should log parent_kept event when parent region stays as-is."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область",
                "Москва",
            ]
        })
        matcher.match_dataframe(df, "region", threshold=70)
        log = matcher.get_match()
        kept = log[log["event"] == "parent_kept"]
        assert len(kept) == 1
        assert "оставлена" in kept.iloc[0]["note"]

    def test_no_parent_events_when_no_parent_regions(self, matcher):
        """No parent_resolved/parent_kept events when no Архангельская/Тюменская in data."""
        df = pd.DataFrame({
            "region": [
                "Москва",
                "Свердловская область",
            ]
        })
        matcher.match_dataframe(df, "region", threshold=70)
        log = matcher.get_match()
        assert "parent_resolved" not in log["event"].values
        assert "parent_kept" not in log["event"].values

    def test_both_parents_resolved_independently(self, matcher):
        """Архангельская and Тюменская should be resolved independently."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область",
                "Ненецкий автономный округ",
                "Тюменская область",
                # No ХМАО/ЯНАО, so Тюменская stays "с АО"
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Архангельская область (без автономного округа)"
        assert result["object_name"].iloc[2] == "Тюменская область (с автономными округами)"

    def test_resolve_sets_score_100_for_redefined(self, matcher):
        """When parent is redefined, its score should be set to 100.0."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область",
                "Ненецкий автономный округ",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["levenshtein_score"].iloc[0] == 100.0

    def test_explicit_bez_ao_not_overridden_by_resolve(self, matcher):
        """Явно указанная 'без АО' не должна быть перезаписана пост-анализом."""
        df = pd.DataFrame({
            "region": [
                "Тюменская область (кроме Ханты-Мансийского автономного округа-Югры "
                "и Ямало-Ненецкого автономного округа)",
                "Ханты-Мансийский автономный округ",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        # Явно указано "кроме..." → уже "без АО", пост-анализ не должен менять
        assert result["object_name"].iloc[0] == "Тюменская область (без автономных округов)"

    def test_explicit_bez_ao_alongside_plain_tyumen(self, matcher):
        """Если есть и явное 'без АО', и просто 'Тюменская', пост-анализ меняет только 'просто'."""
        df = pd.DataFrame({
            "region": [
                "Тюменская область (кроме Ханты-Мансийского автономного округа-Югры "
                "и Ямало-Ненецкого автономного округа)",
                "Тюменская область",
                "ХМАО",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        # Явная "кроме..." → остаётся "без АО"
        assert result["object_name"].iloc[0] == "Тюменская область (без автономных округов)"
        # Просто "Тюменская область" → пост-анализ переопределяет на "без АО" (ХМАО есть)
        assert result["object_name"].iloc[1] == "Тюменская область (без автономных округов)"

    def test_explicit_arkhangelsk_bez_ao_not_overridden(self, matcher):
        """Явно указанная 'Архангельская область (кроме НАО)' не перезаписывается."""
        df = pd.DataFrame({
            "region": [
                "Архангельская область (кроме Ненецкого автономного округа)",
                "Ненецкий автономный округ",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert result["object_name"].iloc[0] == "Архангельская область (без автономного округа)"


# --- full integration: sample data from README ---


class TestIntegration:
    """Integration tests with realistic data similar to README examples."""

    @pytest.fixture
    def matcher(self):
        return RegionMatcher()

    def test_readme_sample_regions(self, matcher):
        """Core regions from README should match correctly."""
        cases = [
            ("московск Обл", "Московская область"),
            ("свердловск", "Свердловская область"),
            ("петербург", "Санкт-Петербург"),
            ("Республика     Алтай", "Республика Алтай"),
            ("спб", "Санкт-Петербург"),
            ("мск", "Москва"),
        ]
        for input_name, expected in cases:
            match, score = matcher.find_best_match(input_name, threshold=50)
            assert match == expected, f"Failed for '{input_name}': got '{match}'"
            assert score is not None, f"Score is None for '{input_name}'"

    def test_full_dataframe_pipeline(self, matcher):
        """Full pipeline: match_dataframe + attach_fields should work together."""
        df = pd.DataFrame({
            "region": [
                "Московская область",
                "Свердловская область",
                "Москва",
            ]
        })
        result = matcher.match_dataframe(df, "region", threshold=70)
        assert "object_name" in result.columns
        assert "levenshtein_score" in result.columns
        assert result["object_name"].notna().all()

        result = matcher.attach_fields(result, "object_name", ["name_eng"], threshold=70)
        assert "name_eng" in result.columns
        assert result["name_eng"].notna().all()


# --- edge cases ---
# TODO:
# Ханты-Мансийский ФО — не распознает, надо править
