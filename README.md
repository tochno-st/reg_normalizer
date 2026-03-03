[License: CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[PyPI version](https://badge.fury.io/py/reg-normalizer)
[Python Versions](https://pypi.org/project/reg-normalizer/)

# Region Normalizer

**Region Normalizer** — инструмент для нормализации и стандартизации наименований российских регионов, а также добавления нормализующих переменных. Он помогает распознавать регион даже в случаях, когда в названии встречаются опечатки, латинские буквы или другие особенности написания. Инструмент сопоставляет различные формы написания с [эталонным справочником](https://github.com/tochno-st/reg_normalizer/blob/main/data/interim/regions_etalon_v2.0.yaml) и позволяет извлекать дополнительные атрибуты, такие как коды ОКАТО, ISO, английские названия и многое другое.

Разные наименования одного и того же региона — частая проблема в реальных данных, например, на портале ЕМИСС встречается до 275 различных вариантов написания регионов. Особенно многообразны варианты у Тюменской и Архангельской областей — по 8 и 7 вариантов соответственно. В состав этих регионов входят автономные округа (ХМАО, ЯНАО, НАО), и часть ведомств отмечает, что данные приведены без автономных округов, но сокращения и формулировки используются самые разные.

## Возможности

- Поиск и нормализация региона по произвольному названию (с учетом опечаток, сокращений, аббревиатур, смешения латиницы и кириллицы)
- Пакетная обработка больших таблиц с названиями регионов
- Гибкая настройка весов алгоритмов сопоставления
- Добавление дополнительных полей из эталонного справочника (ОКАТО, ISO, английское название и др.)
- Добавление нормализующих переменных (численность населения, индекс потребительских цен)

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

1. Создайте виртуальное окружение с помощью `uv`:

```bash
uv venv
```

1. Активируйте виртуальное окружение:

**На macOS/Linux:**

```bash
source .venv/bin/activate
```

**На Windows:**

```bash
.venv\Scripts\activate
```

1. Установите зависимости для разработки:

```bash
uv pip install -e ".[dev]"
```

Или используйте традиционный подход с `pip`:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
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
# Одно поле: передайте список из одного элемента
result_df = matcher.attach_fields(result_df, 'region_name', ['name_eng'])

# Несколько полей
result_df = matcher.attach_fields(result_df, 'region_name',
                                  ['name_eng', 'okato', 'iso_code'])

print(result_df.head())
```

### 5. Работа с показателями (индикаторами)

Пакет позволяет присоединять к данным региональные статистические показатели (население, стоимость потребительской корзины, индекс бюджетных расходов и др.) из встроенной таблицы `normalizers.csv`. Данные доступны в разбивке по годам (2000–2025).

#### Список доступных показателей


| Код                     | Описание                                                                   |
| ----------------------- | -------------------------------------------------------------------------- |
| `pop_total`             | Численность населения — всего                                              |
| `pop_men`               | Численность населения — мужчины                                            |
| `pop_women`             | Численность населения — женщины                                            |
| `pop_urban`             | Численность населения — городское население                                |
| `pop_rural`             | Численность населения — сельское население                                 |
| `pop_0_17`              | Численность населения — 0–17 лет                                           |
| `pop_18_plus`           | Численность населения — 18 лет и старше                                    |
| `pop_below_working`     | Численность населения — моложе трудоспособного                             |
| `pop_working`           | Численность населения — в трудоспособном возрасте (муж. 16–59, жен. 16–54) |
| `pop_above_working`     | Численность населения — старше трудоспособного                             |
| `pop_pension`           | Численность населения — пенсионного возраста (66 лет и старше)             |
| `pop_total_avg`         | Численность населения — всего (в среднем за год)                           |
| `pop_men_avg`           | Численность населения — мужчины (в среднем за год)                         |
| `pop_women_avg`         | Численность населения — женщины (в среднем за год)                         |
| `pop_urban_avg`         | Численность населения — городское население (в среднем за год)             |
| `pop_rural_avg`         | Численность населения — сельское население (в среднем за год)              |
| `pop_0_17_avg`          | Численность населения — 0–17 лет (в среднем за год)                        |
| `pop_18_plus_avg`       | Численность населения — 18 лет и старше (в среднем за год)                 |
| `pop_below_working_avg` | Численность населения — моложе трудоспособного (в среднем за год)          |
| `pop_working_avg`       | Численность населения — в трудоспособном возрасте (в среднем за год)       |
| `pop_above_working_avg` | Численность населения — старше трудоспособного (в среднем за год)          |
| `pop_pension_avg`       | Численность населения — пенсионного возраста (в среднем за год)            |
| `fixed_basket`          | Стоимость фиксированного набора потребительских товаров и услуг            |
| `ibr`                   | Индекс бюджетных расходов                                                  |


Получить этот список программно можно так:

```python
from reg_normalizer import RegionMatcher

matcher = RegionMatcher()
descriptions = matcher.get_indicator_descriptions()
for code, description in descriptions.items():
    print(f"{code}: {description}")
```

#### Присоединение показателей к данным

Метод `attach_indicators` поддерживает три сценария:

**Сценарий 1. В данных есть столбец с годом:**

```python
import pandas as pd
from reg_normalizer import RegionMatcher

matcher = RegionMatcher()

df = pd.DataFrame({
    'region': ['Московская область', 'Республика Татарстан'],
    'year': [2023, 2023]
})

# Нормализуем названия
df = matcher.match_dataframe(df, 'region')

# Присоединяем показатели — merge по региону и году
df = matcher.attach_indicators(
    df,
    indicators=['pop_total', 'ibr'],
    name_col='region',
    year_col='year'
)
```

**Сценарий 2. В данных нет столбца с годом — указываем год явно:**

```python
df = pd.DataFrame({
    'region': ['Московская область', 'Республика Татарстан']
})

df = matcher.match_dataframe(df, 'region')

# Присоединяем показатели за конкретный год
df = matcher.attach_indicators(
    df,
    indicators='pop_total',  # можно передать один код строкой
    name_col='region',
    year=2023
)
```

**Сценарий 3. Несколько показателей сразу:**

```python
df = matcher.attach_indicators(
    df,
    indicators=['pop_total', 'pop_urban', 'pop_rural', 'fixed_basket', 'ibr'],
    name_col='region',
    year=2020
)
```

#### Параметры `attach_indicators`


| Параметр     | Тип                   | Описание                                                        |
| ------------ | --------------------- | --------------------------------------------------------------- |
| `df`         | `DataFrame`           | Таблица с нормализованными названиями регионов                  |
| `indicators` | `str` или `list[str]` | Код показателя или список кодов (см. таблицу выше)              |
| `name_col`   | `str`                 | Имя столбца с названием региона. По умолчанию `'object_name'`   |
| `year_col`   | `str` (опц.)          | Столбец с годом в данных. Если указан — merge по региону и году |
| `year`       | `int` (опц.)          | Год для фильтрации. Используется, если `year_col` не задан      |
| `how`        | `str`                 | Тип соединения: `'left'` (по умолчанию) или `'outer'`           |


> Необходимо указать либо `year_col`, либо `year`. Если не указан ни один — будет вызвана ошибка `ValueError`.

#### Требования к формату таблицы

Метод `attach_indicators` ожидает, что входная таблица имеет **«длинный» (long) формат** — регионы и годы должны идти **в строках**, а не в столбцах:


| region               | year | ... |
| -------------------- | ---- | --- |
| Московская область   | 2020 | ... |
| Московская область   | 2021 | ... |
| Республика Татарстан | 2020 | ... |


Если в вашей таблице годы расположены по столбцам («широкий» формат), её нужно предварительно привести к длинному формату, например с помощью `pd.melt`:

```python
# Было: столбцы — годы
#   region              | 2020 | 2021 | 2022
#   Московская область  | ...  | ...  | ...

df_long = df.melt(id_vars='region', var_name='year', value_name='value')
df_long['year'] = df_long['year'].astype(int)

# Теперь можно присоединять показатели
df_long = matcher.attach_indicators(df_long, indicators='pop_total',
                                    name_col='region', year_col='year')
```

## Разработка и тестирование

### Запуск тестов

Проект использует `pytest` для тестирования. Все тесты находятся в директории `tests/`.

**Запустить все тесты:**

```bash
pytest tests/ -v
```

**Запустить тесты с подробным выводом:**

```bash
pytest tests/ -v -s
```

**Запустить конкретный тестовый файл:**

```bash
pytest tests/test_indicators.py -v
pytest tests/test_regions_validator.py -v
```

**Запустить конкретный тест:**

```bash
pytest tests/test_indicators.py::test_get_indicator_descriptions -v
```

**Запустить тесты с покрытием кода:**

```bash
pytest tests/ --cov=reg_normalizer --cov-report=html
```

После выполнения команды отчет о покрытии будет доступен в `htmlcov/index.html`.

### Структура тестов

- `tests/test_indicators.py` — тесты для функций работы с индикаторами (`get_indicator_descriptions`, `attach_indicators`)
- `tests/test_regions_validator.py` — тесты для `RegionMatcher` и вспомогательных функций (`preprocess_name`, `stem_region_name`, `find_best_match`, `match_dataframe`, `attach_fields`)

### Работа с виртуальным окружением `uv`

Если вы используете `uv` для управления зависимостями:

**Создание нового окружения:**

```bash
uv venv
```

**Установка зависимостей:**

```bash
uv pip install -e ".[dev]"
```

**Обновление зависимостей:**

```bash
uv pip install --upgrade -e ".[dev]"
```

**Деактивация окружения:**

```bash
deactivate
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

## TODO

- Добавить обработку сокращений "Республика" (респ., р., Респ.)
- Решить проблему с фразами, в которых встречается "Российская Федерация", но это не регион
- Проверить, что хорошо обрабатываются вложенные автономные округа

## Лицензия  
  
Creative Commons License Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
