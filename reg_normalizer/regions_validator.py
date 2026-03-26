import os
import yaml
from fuzzywuzzy import fuzz
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer

from .constants import (
    LATIN_TO_CYRILLIC, DEFAULT_ABBREVIATIONS, EXTRA_DATA,
    COMPOUND_SEPARATORS, COMPOUND_REGION_RULES, PARENT_REGION_RULES,
    FOOTNOTE_PATTERN, UNITS_PATTERN, REPUBLIC_PATTERNS,
)
from . import indicators as indicators_module


def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, 'data/interim/regions_etalon_v2.0.yaml')
etalon_data = read_yaml(yaml_path)
ETALON_REGIONS = [region['name_rus'] for region in etalon_data['dict'].values()]


class RegionMatcher:
    """Find best match for region name from etalon list.

    This class provides fuzzy matching capabilities to normalize Russian region names
    by comparing them against an etalon (reference) list of standardized region names.
    It handles typos, abbreviations, Latin-to-Cyrillic character mixing, and various
    spelling variations.

    Attributes:
        etalon (list): List of standardized region names to match against.
        abbreviations (dict): Dictionary mapping common abbreviations to full region names.
        preprocessed_etalon (list): Precomputed list of tuples containing original,
            preprocessed, and stemmed versions of etalon regions for efficient matching.
    """

    def __init__(self, etalon_regions=None, abbreviations=None):
        """Initialize the RegionMatcher with etalon regions and abbreviations.

        Args:
            etalon_regions (list, optional): Custom list of standardized region names
                to use for matching. If None, uses the default ETALON_REGIONS loaded
                from the YAML file. Defaults to None.
            abbreviations (dict, optional): Custom dictionary mapping abbreviated
                region names (keys) to their full standardized names (values).
                If None, uses DEFAULT_ABBREVIATIONS. Defaults to None.

        Example:
            >>> # Use default regions and abbreviations
            >>> matcher = RegionMatcher()
            >>>
            >>> # Use custom etalon regions
            >>> custom_regions = ['Москва', 'Санкт-Петербург']
            >>> matcher = RegionMatcher(etalon_regions=custom_regions)
            >>>
            >>> # Use custom abbreviations
            >>> custom_abbr = {'спб': 'Санкт-Петербург', 'мск': 'Москва'}
            >>> matcher = RegionMatcher(abbreviations=custom_abbr)
        """
        self.etalon = etalon_regions or ETALON_REGIONS
        self.abbreviations = abbreviations or DEFAULT_ABBREVIATIONS
        # Precompute both preprocessed and stemmed versions
        self.preprocessed_etalon = [
            (region, self.preprocess_name(region), self.stem_region_name(self.preprocess_name(region)))
            for region in self.etalon
        ]
        self._match_log: list = []

    def _process_input(self, input_name: str) -> tuple:
        """Handle preprocessing and abbreviation replacement.

        Preprocesses the input name and checks if it matches any known abbreviations.
        If an abbreviation is found, it's replaced with the full region name before
        returning both processed and stemmed versions.

        Args:
            input_name (str): Raw region name to process.

        Returns:
            tuple: A tuple containing:
                - processed (str): Preprocessed (normalized) region name.
                - stemmed (str): Stemmed version of the processed region name.

        Example:
            >>> matcher = RegionMatcher()
            >>> processed, stemmed = matcher._process_input("спб")
            >>> print(processed)
            'санкт петербург'
        """
        processed = self.preprocess_name(input_name)

        # Check for abbreviation match
        if processed in self.abbreviations:
            full_name = self.abbreviations[processed]
            processed = self.preprocess_name(full_name)

        stemmed = self.stem_region_name(processed)
        return processed, stemmed

    @staticmethod
    def preprocess_name(name: str) -> str:
        """Normalize and clean region names for comparison.

        Performs several normalization steps:
        - Converts Latin characters to Cyrillic equivalents
        - Replaces various dash types with spaces
        - Normalizes whitespace (collapse multiple spaces, trim)
        - Converts to lowercase
        - Removes extra metadata phrases (e.g., "в границах", "без учета новых субъектов")

        Args:
            name (str): Region name to preprocess.

        Returns:
            str: Normalized region name in lowercase with cleaned whitespace,
                or empty string if input is not a string.

        Example:
            >>> RegionMatcher.preprocess_name("Mосковская  Область")
            'московская область'
            >>> RegionMatcher.preprocess_name("Тюменская область (без АО)")
            'тюменская область'
        """
        if not isinstance(name, str):
            return ''

        for latin, cyrillic in LATIN_TO_CYRILLIC.items():
            name = name.replace(latin, cyrillic)

        name = re.sub(r'[-–—]+', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().lower()

        # Expand "Республика" abbreviations: "Респ. X", "X Респ." → "республика X"
        for pattern, replacement in REPUBLIC_PATTERNS:
            name = re.sub(pattern, replacement, name).strip()

        # Remove footnote markers: 1);2), 1),2), 2);3) etc.
        name = re.sub(FOOTNOTE_PATTERN, '', name).strip()
        # Remove measurement units appended after comma: , млн т, , млрд. руб. etc.
        name = re.sub(UNITS_PATTERN, '', name).strip()

        for word in EXTRA_DATA:
            if word in name:
                parts = name.partition(word)
                name = (parts[0] + parts[2]).strip()

        return name

    @staticmethod
    def stem_region_name(name: str) -> str:
        """Stem Russian words using Snowball stemmer.

        Applies stemming to reduce Russian words to their root forms, which helps
        with matching variations of the same word (e.g., "московская" and "московской"
        both stem to a similar root).

        Args:
            name (str): Region name to stem (should be preprocessed/lowercase).

        Returns:
            str: String with all words stemmed, or empty string if input is empty.

        Example:
            >>> RegionMatcher.stem_region_name("московская область")
            'московск област'
            >>> RegionMatcher.stem_region_name("свердловской области")
            'свердловск област'
        """
        if not name:
            return ''
        stemmer = SnowballStemmer('russian')
        words = name.split()
        return ' '.join([stemmer.stem(word) for word in words])

    def _find_best_match_core(self, input_name,
                              weights=None,
                              approach_weights=None,
                              threshold=70):
        """Core fuzzy matching without compound-input detection.

        This is the pure fuzzy matching logic, used internally to avoid
        recursion when resolving compound parts.

        Args:
            input_name (str): The region name to match.
            weights (dict, optional): Algorithm weights. Defaults to 50/50.
            approach_weights (dict, optional): Approach weights. Defaults to 50/50.
            threshold (int, optional): Minimum score. Defaults to 70.

        Returns:
            tuple: (best_match, best_score) or (best_match, None) if below threshold.
        """
        weights = weights or {'levenshtein': 0.5, 'token_set': 0.5}
        approach_weights = approach_weights or {'original': 0.5, 'stemmed': 0.5}

        processed_input, stemmed_input = self._process_input(input_name)

        best_match = None
        best_score = 0

        for etalon_name, etalon_preprocessed, etalon_stemmed in self.preprocessed_etalon:
            lev_original = fuzz.ratio(processed_input, etalon_preprocessed)
            ts_original = fuzz.token_set_ratio(processed_input, etalon_preprocessed)
            original_score = (weights['levenshtein'] * lev_original +
                             weights['token_set'] * ts_original)

            lev_stemmed = fuzz.ratio(stemmed_input, etalon_stemmed)
            ts_stemmed = fuzz.token_set_ratio(stemmed_input, etalon_stemmed)
            stemmed_score = (weights['levenshtein'] * lev_stemmed +
                            weights['token_set'] * ts_stemmed)

            total_score = (approach_weights['original'] * original_score +
                          approach_weights['stemmed'] * stemmed_score)

            if total_score > best_score:
                best_score = total_score
                best_match = etalon_name

        return best_match, best_score if best_score >= threshold else None

    def _handle_compound_input(self, input_name, threshold=70):
        """Detect and handle input strings containing multiple regions.

        If the input contains two or more regions connected by separators
        (e.g., "и", ",", ";"), checks whether the combination matches a known
        compound-region exception rule. If it does, returns the canonical form.
        Otherwise returns (None, None) to indicate an unmatchable compound input.

        Args:
            input_name (str): Raw region name that may contain multiple regions.
            threshold (int): Threshold for sub-part matching.

        Returns:
            tuple or None:
                - None if the input is not a compound string (caller should
                  proceed with normal matching).
                - (str, float) if a compound exception rule matches.
                - (None, None) if compound but no exception rule applies.
        """
        preprocessed = self.preprocess_name(input_name)

        # If the full string matches an abbreviation, it's not compound
        if preprocessed in self.abbreviations:
            return None

        # Build combined split pattern from COMPOUND_SEPARATORS
        split_pattern = '|'.join(COMPOUND_SEPARATORS)

        # Check if any separator is present
        if not re.search(split_pattern, preprocessed):
            return None

        parts = [p.strip() for p in re.split(split_pattern, preprocessed) if p.strip()]

        if len(parts) < 2:
            return None

        # Resolve each part to its canonical etalon name
        resolved_parts = set()
        for part in parts:
            match, score = self._find_best_match_core(part, threshold=threshold)
            if match is None or score is None:
                return (None, None)
            resolved_parts.add(match)

        # If all parts resolved to the same region, not truly compound
        if len(resolved_parts) == 1:
            return None

        # Check if the resolved set matches any compound exception rule
        resolved_frozen = frozenset(resolved_parts)
        for rule_key, canonical_name in COMPOUND_REGION_RULES.items():
            if resolved_frozen == rule_key:
                return (canonical_name, 100.0)

        # Compound input detected but no exception rule — return no match
        return (None, None)

    def _match_single(self, input_name, weights=None, approach_weights=None, threshold=70):
        """Internal matching without logging. Used by match_dataframe and attach_fields."""
        compound_result = self._handle_compound_input(input_name, threshold=threshold)
        if compound_result is not None:
            return compound_result
        return self._find_best_match_core(
            input_name, weights=weights,
            approach_weights=approach_weights, threshold=threshold
        )

    def find_best_match(self, input_name,
                       weights=None,
                       approach_weights=None,
                       threshold=70):
        """Find best match using combined fuzzy matching approaches.

        Uses a combination of Levenshtein distance and token set ratio algorithms,
        applied to both original and stemmed text. The final score is a weighted
        combination of these approaches.

        Handles compound inputs (multiple regions in one string joined by "и", ","
        or ";"). For most compound inputs returns (None, None). For exceptions
        like Архангельская область + НАО, returns the combined canonical name.

        Each call appends an entry to the internal match log, accessible via get_match().

        Args:
            input_name (str): The region name to match against the etalon list.
            weights (dict, optional): Weights for different matching algorithms.
                Keys: 'levenshtein', 'token_set'. Values should sum to 1.0 for
                best results. Defaults to {'levenshtein': 0.5, 'token_set': 0.5}.
            approach_weights (dict, optional): Weights for different text processing
                approaches. Keys: 'original', 'stemmed'. Values should sum to 1.0.
                Defaults to {'original': 0.5, 'stemmed': 0.5}.
            threshold (int, optional): Minimum score (0-100) required to accept a match.
                Scores below this return None. Defaults to 70.

        Returns:
            tuple: A tuple containing:
                - best_match (str or None): The matched etalon region name, or None
                  if no match meets the threshold.
                - best_score (float or None): The matching score (0-100), or None
                  if below threshold.

        Example:
            >>> matcher = RegionMatcher()
            >>> match, score = matcher.find_best_match("свердловск")
            >>> print(f"{match}: {score:.2f}")
            'Свердловская область: 85.50'
        """
        match, score = self._match_single(
            input_name, weights=weights,
            approach_weights=approach_weights, threshold=threshold
        )
        if score is None:
            event, note = 'low_score', 'нет совпадения выше порога — проверьте вручную'
        else:
            event, note = 'match', ''
        self._match_log.append({
            'original': input_name,
            'normalized': match,
            'score': score,
            'event': event,
            'note': note,
        })
        return match, score

    def _resolve_parent_regions(self, value_mapping, score_mapping=None):
        """Post-analysis: disambiguate parent regions based on the full set of matched regions.

        If a dataset contains both a parent region (e.g., "Архангельская область (с автономным округом)")
        and its child autonomous okrug (e.g., "Ненецкий автономный округ"), the parent should be
        replaced with the "without AO" variant.

        Args:
            value_mapping (dict): Mapping of original names to matched etalon names.
            score_mapping (dict, optional): Mapping of original names to scores.
                If provided, scores for replaced regions are set to 100.0.

        Returns:
            tuple: (value_mapping, log_updates) where log_updates is a dict mapping
                original input value → (event, note) for entries affected by parent resolution.
        """
        all_matched = set(value_mapping.values()) - {None}
        log_updates = {}

        for rule in PARENT_REGION_RULES:
            parent = rule['parent']
            child_regions = rule['child_regions']
            without_ao = rule['without_ao']

            if parent not in all_matched:
                continue

            # Check if any child autonomous okrug is also in the dataset
            children_present = child_regions & all_matched

            if children_present:
                # Children are present separately → parent should be "without AO"
                children_str = ", ".join(sorted(children_present))
                for key, val in value_mapping.items():
                    if val == parent:
                        value_mapping[key] = without_ao
                        if score_mapping is not None:
                            score_mapping[key] = 100.0
                        log_updates[key] = (
                            'parent_resolved',
                            f'"{parent}" → "{without_ao}": АО ({children_str}) найдены отдельно в данных',
                        )
            else:
                # No children → parent stays as "with AO"
                for key, val in value_mapping.items():
                    if val == parent:
                        log_updates[key] = (
                            'parent_kept',
                            f'"{parent}" оставлена с АО: автономные округа отдельно не найдены',
                        )

        return value_mapping, log_updates

    def match_dataframe(self, df: pd.DataFrame, column_name: str, **kwargs):
        """Apply matching to an entire DataFrame column by first processing unique values.

        Efficiently matches region names in a DataFrame by processing only unique values
        and mapping the results back to all rows. Adds two new columns: 'object_name' (matched
        region name) and 'levenshtein_score' (matching confidence score).

        After matching, performs post-analysis to disambiguate parent regions
        (Архангельская/Тюменская) based on whether their autonomous okrugs
        are present separately in the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing region names to match.
            column_name (str): Name of the column containing region names.
            **kwargs: Additional keyword arguments passed to find_best_match(),
                such as 'weights', 'approach_weights', and 'threshold'.

        Returns:
            pd.DataFrame: The input DataFrame with two new columns added:
                - 'object_name': The matched etalon region name (or None if no match).
                - 'levenshtein_score': The matching score (0-100, or 0 if no match).

        Example:
            >>> import pandas as pd
            >>> matcher = RegionMatcher()
            >>> df = pd.DataFrame({
            ...     'region_name': ['московск обл', 'свердловск', 'спб']
            ... })
            >>> result = matcher.match_dataframe(
            ...     df,
            ...     'region_name',
            ...     weights={'levenshtein': 0.4, 'token_set': 0.6},
            ...     threshold=70
            ... )
            >>> print(result[['region_name', 'object_name', 'levenshtein_score']])
        """
        self._match_log = []

        # Get unique values and create mapping dictionaries
        unique_values = df[column_name].unique()
        value_mapping = {}
        score_mapping = {}

        for value in unique_values:
            match_result = self._match_single(value, **kwargs)
            value_mapping[value] = match_result[0]
            score_mapping[value] = match_result[1] if match_result[1] is not None else 0

        # Post-analysis: disambiguate parent regions
        value_mapping, log_updates = self._resolve_parent_regions(value_mapping, score_mapping)

        # Build match log
        for value in unique_values:
            if value in log_updates:
                event, note = log_updates[value]
            elif score_mapping[value] == 0:
                event, note = 'low_score', 'нет совпадения выше порога — проверьте вручную'
            else:
                event, note = 'match', ''
            self._match_log.append({
                'original': value,
                'normalized': value_mapping[value],
                'score': score_mapping[value] if score_mapping[value] != 0 else None,
                'event': event,
                'note': note,
            })

        # Apply the mappings to create new columns
        df['object_name'] = df[column_name].map(value_mapping)
        df['levenshtein_score'] = df[column_name].map(score_mapping)

        return df

    def attach_fields(self, df: pd.DataFrame, column_name: str, etalon_fields: list, **kwargs) -> pd.DataFrame:
        """Add one or more fields from etalon data in a single efficient operation.

        Performs fuzzy matching only once per unique value, then attaches all
        requested etalon fields. For a single field, pass a one-element list
        (e.g. ['name_eng']).

        After matching, performs post-analysis to disambiguate parent regions
        based on the full set of matched regions.

        Args:
            df: DataFrame to modify
            column_name: Column containing region names
            etalon_fields: List of field names from etalon data to attach
                          (e.g., ['name_eng', 'okato', 'iso_code'])
            **kwargs: Additional arguments passed to find_best_match

        Returns:
            DataFrame with all requested fields added as new columns

        Example:
            >>> matcher = RegionMatcher()
            >>> df = matcher.attach_fields(df, 'region_name',
            ...                            ['name_eng', 'okato', 'iso_code'])
        """
        self._match_log = []

        # Get unique values to avoid redundant matching
        unique_values = df[column_name].unique()

        # First pass: match all unique values
        match_results = {}
        value_mapping = {}
        for value in unique_values:
            match_result = self._match_single(value, **kwargs)
            match_results[value] = match_result
            value_mapping[value] = match_result[0]

        # Post-analysis: disambiguate parent regions
        value_mapping, log_updates = self._resolve_parent_regions(value_mapping)

        # Update match_results with resolved values
        for value in unique_values:
            old_match = match_results[value][0]
            new_match = value_mapping[value]
            if old_match != new_match:
                match_results[value] = (new_match, 100.0)

        # Build match log
        for value in unique_values:
            score = match_results[value][1]
            if value in log_updates:
                event, note = log_updates[value]
            elif score is None:
                event, note = 'low_score', 'нет совпадения выше порога — проверьте вручную'
            else:
                event, note = 'match', ''
            self._match_log.append({
                'original': value,
                'normalized': value_mapping[value],
                'score': score,
                'event': event,
                'note': note,
            })

        # Create a mapping dictionary for each field
        field_mappings = {field: {} for field in etalon_fields}

        # Build mappings for unique values only (single pass)
        for value in unique_values:
            match_result = match_results[value]

            if not match_result or match_result[1] is None:
                for field in etalon_fields:
                    field_mappings[field][value] = None
                continue

            best_match, score = match_result

            # Find the matching etalon record
            for record in etalon_data['dict'].values():
                if record['name_rus'] == best_match:
                    for field in etalon_fields:
                        field_mappings[field][value] = record.get(field)
                    break
            else:
                for field in etalon_fields:
                    field_mappings[field][value] = None

        # Apply all mappings to create new columns
        for field in etalon_fields:
            df[field] = df[column_name].map(field_mappings[field])

        return df

    def get_match(self) -> pd.DataFrame:
        """Return a DataFrame with all transformations from the last matching run.

        Each row corresponds to one unique input value. Results are sorted by
        complexity: most complex / problematic cases first.

        Event types (sort order):
        - parent_resolved: parent region was reassigned based on children found in data
          (e.g. "Архангельская область (с АО)" → "Архангельская область (без АО)")
        - low_score:       no match found above threshold — needs manual review
        - parent_kept:     parent region kept as-is because no child AOs found separately
        - match:           normal successful match

        Columns:
            original   — исходное название из данных
            normalized — итоговое нормализованное название (None для low_score)
            score      — балл совпадения (None для low_score)
            event      — тип события (см. выше)
            note       — дополнительное пояснение

        Returns:
            pd.DataFrame, empty if no matching run has been performed yet.

        Example:
            >>> matcher = RegionMatcher()
            >>> matcher.match_dataframe(df, 'region')
            >>> matcher.get_match()
        """
        if not self._match_log:
            return pd.DataFrame(columns=['original', 'normalized', 'score', 'event', 'note'])

        event_order = {'parent_resolved': 0, 'low_score': 1, 'parent_kept': 2, 'match': 3}
        result = pd.DataFrame(self._match_log)
        result['_order'] = result['event'].map(event_order)
        result = result.sort_values('_order').drop(columns='_order').reset_index(drop=True)
        return result

    def get_indicator_descriptions(self) -> dict:
        """Return indicator code -> Russian description mapping.

        Loads from indicators_descriptions.yaml in package data.
        Returns dict[str, str] (e.g. 'pop_total' -> 'Численность населения — всего').
        """
        return indicators_module.get_indicator_descriptions()

    def attach_indicators(
        self,
        df: pd.DataFrame,
        indicators,
        name_col: str = "object_name",
        year_col: str = None,
        year: int = None,
        how: str = "left",
    ) -> pd.DataFrame:
        """Attach one or more indicator columns from normalizers by region (and optionally year).

        Three merge scenarios:
        1. With year in data: set year_col -> merge on (name_col, year_col) with (object_name, year).
        2. Without year in data: set year -> take values for that year only, merge on name_col with object_name.
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
        return indicators_module.attach_indicators(
            df, indicators, name_col=name_col, year_col=year_col, year=year, how=how
        )

if __name__ == '__main__':
    matcher = RegionMatcher()

    data = pd.DataFrame({
    'region': [
        'московск Обл',        # Shortened form
        'свердловск',          # Without 'область'
        'петербург',           # Shortened
        'Mосковская област',   # Latin 'M' + typo
        'татарстан респ.',     # Abbreviation
        'Свердлов обл',         # Different ending
        'aлтайский к',
        'Республика     Алтай',
        'ХМао',
        'Юж федеральный округ'
    ]})

    result = matcher.match_dataframe(
        data,
        'region',
        weights={'levenshtein': 0.4, 'token_set': 0.6},
        approach_weights={'original': 0.3, 'stemmed': 0.7},
        threshold=70)

    print(result)
