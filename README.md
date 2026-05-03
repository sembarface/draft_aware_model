# draft_aware_model

## ML artifacts

Trained `.cbm` models are stored under `models/patch_*`.

Commit-friendly ML reports are stored under `reports/ml/patch_*`:

- `metrics/` for `*_metrics.json`, comparisons and metric tables.
- `features/` for `*_features.json`.
- `importance/` for feature-importance exports.
- `errors/` for error-analysis reports.
- `recommendations/` for recommendation explanations.

Active datasets: `base`, `interactions`, `players`, `players_smooth`.

Run local draft UI:

```bash
streamlit run src/ui/app.py
```

Курсовой ML-проект по поддержке принятия решений на стадии драфта Dota 2.

Цель проекта - построить модель, которая для каждого состояния драфта ранжирует доступных героев-кандидатов отдельно для действий `pick` и `ban`.

## Текущий ML-пайплайн

Пайплайн строится вокруг candidate table: одна строка соответствует одному доступному герою-кандидату в конкретном состоянии драфта.

Основные шаги:

1. `src/ml/build_draft_events.py` - сбор событий драфта из raw-таблиц.
2. `src/ml/build_draft_states.py` - построение последовательных состояний драфта.
3. `src/ml/build_draft_candidates.py` - построение candidate-таблиц для pick и ban.
4. `src/ml/train_catboost.py` - обучение CatBoost-модели для выбранного действия.
5. `src/ml/evaluate.py` - расчет ranking-метрик на тестовой части.

## Запуск

Обновить локальные данные, сконвертировать parquet и построить ML-таблицы:

```bash
python -m src.data_update --patch-label 7.41 --fetch --convert --build-ml
```

Запустить baseline pipeline:

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset base
```

Запустить pipeline с interaction features:

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset interactions
```

Патч задается через `--patch-label`; соответствие между OpenDota patch label (`7.41`) и JSON patch id (`60`) хранится в `src/config.py`.

## Структура проекта

- `src/ml/` - скрипты построения датасетов, обучения и оценки.
- `data/` - локальные raw-данные и ML-таблицы, не коммитятся, кроме небольших справочников.
- `models/` - локальные обученные модели и артефакты, не коммитятся.
- `reports/` - текстовые отчеты и заметки по проекту.
- `project_status.md` - текущий статус пайплайна и результатов.
- `README.md`, `PROJECT_CONTEXT.md`, `AGENTS.md` - описание проекта и инструкции.

## Данные и модели

Большие данные, parquet-файлы, JSON-дампы, обученные модели и локальные артефакты хранятся только локально и не должны попадать в Git. Небольшой справочник `data/heroes.csv` можно хранить в репозитории.

## Data schema

Raw JSON files, parquet tables and trained models are not stored in the repository.

Instead, the repository contains schema documentation:

- `docs/data_schema/parquet_tables.md`
- `docs/data_schema/raw_match_json_structure.md`
- `docs/data_schema/data_pipeline.md`
- `project_status.md`
- `reports/notebook_report.md`

Цель - сделать репозиторий легким и понятным без загрузки больших данных.

## Текущий статус

Для `patch_60` уже построены:

- `draft_events`
- `draft_states`
- `draft_candidates_pick`
- `draft_candidates_ban`

Также обучены первые CatBoost-модели для ранжирования кандидатов на pick и ban.
