# Real Estate Agency — Price Prediction (CatBoost + Feature Engineering)

Проект по прогнозированию стоимости недвижимости на табличных данных.  
Основной упор: очистка “сырых” полей, парсинг вложенных структур (`homeFacts`, `schools`), сравнение baseline-моделей и финальная оптимизация CatBoost через Optuna.

## Краткое резюме
- **Финальная модель:** `CatBoostRegressor` (обучение на `log1p(target)`, обратное преобразование `expm1`)
- **Оценка качества:** **CV-only**, `KFold(n_splits=3, shuffle=True, random_state=42)`
- **Лучший результат Optuna:** **CV MAE = 91 430 ± 3 681**
  - MAE по фолдам: **87 969 / 95 298 / 91 023**
- **Финальное обучение:** `final_cb_optuna` дообучена на всех данных `X_v2/y_v2`
- **Артефакты:** модель и JSON-метаданные сохраняются в `artifacts/`

## Данные
Источник данных: `data/data.csv`.

Ключевые “сырые” поля:
- `target` (цена) — строки, валютные символы, диапазоны значений
- гео/категории: `city`, `state`, `zipcode`, `status`, `propertyType`
- числовые в строках: `beds`, `baths`, `sqft`, `stories`
- вложенные структуры: `homeFacts`, `schools`

> В репозитории может отсутствовать исходный датасет (если есть ограничения).  
> В таком случае положите `data.csv` в `data/` и следуйте разделу “Запуск”.

## Feature Engineering
Сделаны следующие преобразования:
- очистка и приведение `target` к числу (в т.ч. диапазоны `a-b` → среднее)
- нормализация категорий:
  - `state` → `state_clean` (2-буквенный код + `UNK`)
  - `zipcode` → `zip5`, `zip3`
  - `status` → `status_grp` (укрупнение классов)
  - `propertyType` → `property_type_grp`
- извлечение числовых значений из строк: `beds_num`, `baths_num`, `sqft_num`, `stories_num`
- бинарные/агрегаты:
  - `has_private_pool`, `has_fireplace`, `fireplace_n`
  - `street_is_disclosed`, `street_len`
  - `has_mls`
- парсинг `homeFacts`:
  - `year_built`, `remodeled_year`, `lotsize`, `price_sqft`
  - производные: `home_age`, `is_remodeled`, `years_since_remodel`
  - мульти-метки: heating/cooling/parking → ограниченный набор dummy-признаков
- парсинг `schools`:
  - `schools_count`
  - `schools_distance_min`
  - `schools_all_ages` (K-12 / PK-12 / all grades)

## Модели
Сравнивались:
- `LinearRegression` (OHE, log-target) — baseline
- `Ridge`, `RandomForestRegressor` (OHE, log-target) — sanity-check
- `CatBoostRegressor` — основная модель (нативная работа с категориальными)

Финальный тюнинг:
- Optuna (TPE sampler), 25 trials
- 3-fold KFold CV
- оптимизируем по **MAE в исходной шкале target** (предсказания делаются через `expm1`)

## Структура проекта
- `Brif-1-Real-Estate_Agency-final.ipynb` — основной ноутбук
- `data/data.csv` — входные данные
- `artifacts/` — сохранённые модели и метаданные
  - `final_cb_optuna_YYYYMMDD_HHMMSS.joblib`
  - `final_cb_optuna_YYYYMMDD_HHMMSS.json`

## Установка и запуск (macOS / VS Code)
### 1) Создать окружение и поставить зависимости
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn catboost optuna seaborn matplotlib joblib
```

### 2) Запуск ноутбука
Открыть файл:
- `Brif-1-Real-Estate_Agency-final.ipynb`

Далее выполнить ячейки сверху вниз:
1) загрузка данных
2) очистка/feature engineering
3) baseline-сравнения (опционально)
4) Optuna CV-only тюнинг
5) обучение финальной модели на всех данных
6) сохранение в `artifacts/`
7) inference пример (загрузка модели и прогноз)

## Inference (как использовать сохранённую модель)
После выполнения блока сохранения в ноутбуке появятся файлы в `artifacts/`.  
Далее можно загрузить самый свежий `*.joblib` и получить прогнозы (пример есть в ноутбуке).

Важно:
- модель возвращает предсказание в **лог-шкале**, поэтому применяется `np.expm1`
- отрицательные значения клипаются в 0

## Воспроизводимость
- Везде фиксирован `random_state/random_seed = 42`
- CV: `KFold(shuffle=True, random_state=42)`
- Целевая трансформация: `log1p` / `expm1`

## Ограничения и возможные улучшения
- Уточнить обработку пропусков (особенно числовых) строго внутри фолда CV (пайплайн-стиль).
- Поднять качество через:
  - увеличение `n_trials` Optuna
  - дополнительные гео-фичи (агрегации по zip/city без утечки)
  - использование `MAE` как `loss_function` (и сравнение с RMSE в лог-шкале)

## Автор
- maksmick99
