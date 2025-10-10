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
    """Find best match for region name from etalon list."""

    def __init__(self, etalon_regions=None, abbreviations=None):
        self.etalon = etalon_regions or ETALON_REGIONS
        self.abbreviations = abbreviations or DEFAULT_ABBREVIATIONS
        # Precompute both preprocessed and stemmed versions
        self.preprocessed_etalon = [
            (region, self.preprocess_name(region), self.stem_region_name(self.preprocess_name(region))) 
            for region in self.etalon
        ]

    def _process_input(self, input_name: str) -> tuple:
        """Handle preprocessing and abbreviation replacement"""
        processed = self.preprocess_name(input_name)

        # Check for abbreviation match
        if processed in self.abbreviations:
            full_name = self.abbreviations[processed]
            processed = self.preprocess_name(full_name)
        
        stemmed = self.stem_region_name(processed)
        return processed, stemmed
    
    @staticmethod
    def preprocess_name(name: str) -> str:
        """Normalize and clean region names for comparison."""
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
        """Stem Russian words using Snowball stemmer"""
        if not name:
            return ''
        stemmer = SnowballStemmer('russian')
        words = name.split()
        return ' '.join([stemmer.stem(word) for word in words])
    
    def find_best_match(self, input_name, 
                       weights=None, 
                       approach_weights=None,
                       threshold=65):
        """Find best match using combined approaches"""
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
        """Apply matching to an entire DataFrame column by first processing unique values"""
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
        """Add additional field to the dataframe based on the best match."""

        def lookup_etalon_value(region_name: str):
            match_result = self.find_best_match(region_name, **kwargs)
            
            if not match_result or match_result[1] is None:
                return None
            
            best_match, score = match_result
            matched_row = pd.DataFrame([row for row in etalon_data['dict'].values() if row['name_rus'] == best_match])
            if not matched_row.empty:
                return matched_row.iloc[0][etalon_field]
            return None

        df[etalon_field] = df[column_name].apply(lookup_etalon_value)
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
