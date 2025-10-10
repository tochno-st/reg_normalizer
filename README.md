[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY—NC—SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![PyPI version](https://badge.fury.io/py/reg-normalizer.svg)](https://badge.fury.io/py/reg-normalizer)
[![Python Versions](https://img.shields.io/pypi/pyversions/reg-normalizer.svg)](https://pypi.org/project/reg-normalizer/)


# Region Normalizer

**Region Normalizer** — инструмент для нормализации и стандартизации наименований российских регионов. Он помогает распознавать регион даже в случаях, когда в названии встречаются опечатки, латинские буквы или другие особенности написания. Инструмент сопоставляет различные формы написания с [эталонным справочником](https://github.com/tochno-st/reg_normalizer/blob/main/data/interim/regions_etalon_v2.0.yaml) и позволяет извлекать дополнительные атрибуты, такие как коды ОКАТО, ISO, английские названия и многое другое.

Разные наименования одного и того же региона — частая проблема в реальных данных, например, на портале ЕМИСС встречается до 275 различных вариантов написания регионов. Особенно многообразны варианты у Тюменской и Архангельской областей — по 8 и 7 вариантов соответственно. В состав этих регионов входят автономные округа (ХМАО, ЯНАО, НАО), и часть ведомств отмечает, что данные приведены без автономных округов, но сокращения и формулировки используются самые разные.

## Возможности
- Поиск и нормализация региона по произвольному названию (с учетом опечаток, сокращений, аббревиатур, смешения латиницы и кириллицы)
- Пакетная обработка больших таблиц с названиями регионов
- Гибкая настройка весов алгоритмов сопоставления
- Добавление дополнительных полей из эталонного справочника (ОКАТО, ISO, английское название и др.)

## Установка

Установите пакет с помощью pip:

```bash
pip install reg-normalizer
```

### Установка для разработки

Если вы хотите внести изменения в код:

1. Клонируйте репозиторий:

```bash
git clone https://github.com/tochno-st/reg_normalizer.git
cd reg_normalizer
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Импорт и инициализация

```python
from reg_normalizer import RegionMatcher

matcher = RegionMatcher()
```

### 2. Нормализация одного региона

```python
region_name = "московск область"
match, score = matcher.find_best_match(region_name)
print(f"Input: {region_name}")
print(f"Match: {match}")
print(f"Score: {score:.2f}")
```

### 3. Использование с DataFrame

```python
import pandas as pd
sample_data = pd.DataFrame({
    'region_name': [
        'московск Обл',
        'свердловск',
        'петербург',
        'Mосковская област',
        'татарстан респ.',
        'Свердлов обл',
        'aлтайский к',
        'Республика     Алтай',
        'ХМао',
        'Юж федеральный округ',
        'спб',
        'рт',
        'город москва столица российской федерации город федерального значения',
        'тюменская область (кроме ханты мансийского автономного округа югры и ямало ненецкого автономного округа)'
    ]
})

result_df = matcher.match_dataframe(
    sample_data,
    'region_name',
    weights={'levenshtein': 0.4, 'token_set': 0.6},
    approach_weights={'original': 0.3, 'stemmed': 0.7},
    threshold=70
)
```

### 4. Добавление дополнительных полей

```python
# Добавить английские названия
result_df = matcher.attach_field(result_df, 'region_name', 'name_eng')
# Добавить коды ОКАТО
result_df = matcher.attach_field(result_df, 'region_name', 'okato')
# Добавить коды ISO
result_df = matcher.attach_field(result_df, 'region_name', 'iso_code')
print(result_df.head())
```

## Кастомизация

- **weights** — веса для алгоритмов сравнения ('levenshtein', 'token_set')
- **approach_weights** — веса для подходов ('original', 'stemmed')
- **threshold** — пороговое значение для принятия совпадения

Пример:
```python
custom_weights = {'levenshtein': 0.3, 'token_set': 0.7}
custom_approach_weights = {'original': 0.2, 'stemmed': 0.8}
match, score = matcher.find_best_match(
    "свердловск",
    weights=custom_weights,
    approach_weights=custom_approach_weights,
    threshold=60
)
```

## Структура эталонного справочника

Файл [`data/interim/regions_etalon_v2.0.yaml`](https://github.com/tochno-st/reg_normalizer/blob/main/data/interim/regions_etalon_v2.0.yaml) содержит ключи:
- `name_rus` — официальное название региона
- `name_eng` — английское название
- `okato` — код ОКАТО
- `iso_code` — код ISO

## Лицензия

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
Creative Commons License Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
