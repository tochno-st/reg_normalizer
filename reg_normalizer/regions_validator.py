import os
import yaml
from fuzzywuzzy import fuzz
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer


def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, 'data/interim/regions_etalon_v2.0.yaml')
etalon_data = read_yaml(yaml_path)
ETALON_REGIONS = [region['name_rus'] for region in etalon_data['dict'].values()]

LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'I': 'І', 'J': 'Ј', 'K': 'К', 'M': 'М', 'O': 'О',
    'P': 'Р', 'S': 'С', 'T': 'Т', 'X': 'Х', 'Y': 'У',
    'a': 'а', 'b': 'в', 'c': 'с', 'e': 'е', 'i': 'і',
    'j': 'ј', 'k': 'к', 'm': 'м', 'o': 'о', 'p': 'р',
    's': 'с', 't': 'т', 'x': 'х', 'y': 'у'
}

DEFAULT_ABBREVIATIONS = {
    'хмао': 'Ханты-Мансийский автономный округ',
    'хм а о':'Ханты-Мансийский автономный округ — Югра',
    'янао': 'Ямало-Ненецкий автономный округ',
    'я н ао':'Ямало-Ненецкий автономный округ',
    'нао': 'Ненецкий автономный округ',
    'н а о':'Ненецкий автономный округ',
    'чао': 'Чукотский автономный округ',
    'мо': 'Московская область',
    'спб': 'Санкт-Петербург',
    'свердл': 'Свердловская область',
    'рт': 'Республика Татарстан',
    'рб': 'Республика Башкортостан',
    ' фо': 'федеральный округ',
    'кбр': 'Кабардино-Балкарская Республика',
    'кчр': 'Карачаево-Черкесская Республика',
    'еао': 'Еврейская автономная область',
    'рсо': 'Республика Северная Осетия',
    'цао': 'Центральный федеральный округ',
    'сзфо': 'Северо-Западный федеральный округ',
    'юфо': 'Южный федеральный округ',
    'скфо': 'Северо-Кавказский федеральный округ',
    'пфо': 'Приволжский федеральный округ',
    'уфо': 'Уральский федеральный округ',
    'сфо': 'Сибирский федеральный округ',
    'двфо': 'Дальневосточный федеральный округ',
    'россии': 'Российская Федерация',
    'город москва столица российской федерации город федерального значения': 'Москва',
    'город санкт петербург город федерального значения': 'Санкт-Петербург',
    'город федерального значения севастополь': 'Севастополь',
    'иные территории, включая байконур': 'Иные территории, включая Байконур',
    'тюменская область (кроме ханты мансийского автономного округа югры и ямало ненецкого автономного округа)': 'Тюменская область',
    'ненецкий автономный округ (архангельская область)': 'Ненецкий автономный округ',
    'архангельская область (кроме ненецкого автономного округа)': 'Архангельская область',
    'ямало ненецкий автономный округ (тюменская область)': 'Ямало-Ненецкий автономный округ',
    'республика татарстан (татарстан)': 'Республика Татарстан',
    'тюменская область (без ао)': 'Тюменская область',
    'республика адыгея (адыгея)': 'Республика Адыгея',
    'ханты мансийский автономный округ югра (тюменская область)': 'Ханты-Мансийский автономный округ',
}

EXTRA_DATA = ['в границах', 'после', 'без учета новых субъектов (с 01.01.2023)',
              '(по 2009 год)', '(с 2010 года)', '(с 29.07.2016)', '(без АО)',
              '(с 03.11.2018)']


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

    Example:
        >>> matcher = RegionMatcher()
        >>> match, score = matcher.find_best_match("московск область")
        >>> print(match)
        'Московская область'
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

        for word in EXTRA_DATA:
            if word in name:
                name = name.partition(word)[0].strip()

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
    
    def find_best_match(self, input_name,
                       weights=None,
                       approach_weights=None,
                       threshold=65):
        """Find best match using combined fuzzy matching approaches.

        Uses a combination of Levenshtein distance and token set ratio algorithms,
        applied to both original and stemmed text. The final score is a weighted
        combination of these approaches.

        Args:
            input_name (str): The region name to match against the etalon list.
            weights (dict, optional): Weights for different matching algorithms.
                Keys: 'levenshtein', 'token_set'. Values should sum to 1.0 for
                best results. Defaults to {'levenshtein': 0.5, 'token_set': 0.5}.
            approach_weights (dict, optional): Weights for different text processing
                approaches. Keys: 'original', 'stemmed'. Values should sum to 1.0.
                Defaults to {'original': 0.5, 'stemmed': 0.5}.
            threshold (int, optional): Minimum score (0-100) required to accept a match.
                Scores below this return None. Defaults to 65.

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
            >>>
            >>> # Custom weights favoring token matching and stemming
            >>> match, score = matcher.find_best_match(
            ...     "московск обл",
            ...     weights={'levenshtein': 0.3, 'token_set': 0.7},
            ...     approach_weights={'original': 0.2, 'stemmed': 0.8},
            ...     threshold=70
            ... )
        """
        # Set default weights if not provided
        weights = weights or {'levenshtein': 0.5, 'token_set': 0.5}
        approach_weights = approach_weights or {'original': 0.5, 'stemmed': 0.5}

        processed_input, stemmed_input = self._process_input(input_name)

        best_match = None
        best_score = 0

        for etalon_name, etalon_preprocessed, etalon_stemmed in self.preprocessed_etalon:
            # Calculate original approach scores
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

        if best_score < threshold:
            print(f"WARNING: Best match score {best_score} is less than threshold {threshold} for {input_name} and {etalon_name}. Check it manually.")

        return best_match, best_score if best_score >= threshold else None
    
    def match_dataframe(self, df: pd.DataFrame, column_name: str, **kwargs):
        """Apply matching to an entire DataFrame column by first processing unique values.

        Efficiently matches region names in a DataFrame by processing only unique values
        and mapping the results back to all rows. Adds two new columns: 'ter' (matched
        region name) and 'levenshtein_score' (matching confidence score).

        Args:
            df (pd.DataFrame): DataFrame containing region names to match.
            column_name (str): Name of the column containing region names.
            **kwargs: Additional keyword arguments passed to find_best_match(),
                such as 'weights', 'approach_weights', and 'threshold'.

        Returns:
            pd.DataFrame: The input DataFrame with two new columns added:
                - 'ter': The matched etalon region name (or None if no match).
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
            >>> print(result[['region_name', 'ter', 'levenshtein_score']])
        """
        # Get unique values and create mapping dictionaries
        unique_values = df[column_name].unique()
        value_mapping = {}
        score_mapping = {}

        for value in unique_values:
            match_result = self.find_best_match(value, **kwargs)
            value_mapping[value] = match_result[0]
            score_mapping[value] = match_result[1] if match_result[1] is not None else 0

        # Apply the mappings to create new columns
        df['ter'] = df[column_name].map(value_mapping)
        df['levenshtein_score'] = df[column_name].map(score_mapping)

        return df
    
    def attach_field(self, df: pd.DataFrame, column_name: str, etalon_field: str, **kwargs) -> pd.DataFrame:
        """Add additional field to the dataframe based on the best match.

        Optimized to process only unique values in the column.

        Args:
            df: DataFrame to modify
            column_name: Column containing region names
            etalon_field: Field name from etalon data to attach (e.g., 'name_eng', 'okato', 'iso_code')
            **kwargs: Additional arguments passed to find_best_match

        Returns:
            DataFrame with new column added
        """
        # Get unique values to avoid redundant matching
        unique_values = df[column_name].unique()
        field_mapping = {}

        # Build mapping for unique values only
        for value in unique_values:
            match_result = self.find_best_match(value, **kwargs)

            if not match_result or match_result[1] is None:
                field_mapping[value] = None
                continue

            best_match, score = match_result
            # Find the matching etalon record
            for record in etalon_data['dict'].values():
                if record['name_rus'] == best_match:
                    field_mapping[value] = record.get(etalon_field)
                    break
            else:
                field_mapping[value] = None

        # Apply the mapping to create new column
        df[etalon_field] = df[column_name].map(field_mapping)
        return df

    def attach_fields(self, df: pd.DataFrame, column_name: str, etalon_fields: list, **kwargs) -> pd.DataFrame:
        """Add multiple fields from etalon data in a single efficient operation.

        This is much more efficient than calling attach_field() multiple times,
        as it performs fuzzy matching only once per unique value.

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
        # Get unique values to avoid redundant matching
        unique_values = df[column_name].unique()

        # Create a mapping dictionary for each field
        field_mappings = {field: {} for field in etalon_fields}

        # Build mappings for unique values only (single pass)
        for value in unique_values:
            match_result = self.find_best_match(value, **kwargs)

            if not match_result or match_result[1] is None:
                # No match found - set all fields to None
                for field in etalon_fields:
                    field_mappings[field][value] = None
                continue

            best_match, score = match_result

            # Find the matching etalon record
            for record in etalon_data['dict'].values():
                if record['name_rus'] == best_match:
                    # Extract all requested fields from this record
                    for field in etalon_fields:
                        field_mappings[field][value] = record.get(field)
                    break
            else:
                # Match found but not in etalon data (shouldn't happen)
                for field in etalon_fields:
                    field_mappings[field][value] = None

        # Apply all mappings to create new columns
        for field in etalon_fields:
            df[field] = df[column_name].map(field_mappings[field])

        return df

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
