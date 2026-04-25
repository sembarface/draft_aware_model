# draft_aware_model

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



The repository contains schema documentation:

- `docs/data_schema/parquet_tables.md`
- `docs/data_schema/raw_match_json_structure.md`
- `docs/data_schema/data_pipeline.md`
- `project_status.md`
- `reports/notebook_report.md`


## Текущий статус

Для `patch_60` уже построены:

- `draft_events`
- `draft_states`
- `draft_candidates_pick`
- `draft_candidates_ban`

Также обучены первые CatBoost-модели для ранжирования кандидатов на pick и ban.
