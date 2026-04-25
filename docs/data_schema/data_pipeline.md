# Data pipeline

## 1. Raw JSON loading

Источник: OpenDota.

Скрипт: `src/parser_explorer.py`.

Результат: локальные JSON-файлы в:

```text
data/patch_60/matches/
```


## 2. Conversion to raw parquet tables

Скрипт: `src/convert_to_parquet.py`.

Результаты:

- `matches.parquet`
- `players.parquet`
- `picks_bans.parquet`
- `heroes_stats.parquet`

На этом шаге сырые JSON матчей превращаются в плоские таблицы матчей, игроков, действий драфта и агрегированной статистики героев.

## 3. Draft events

Скрипт: `src/ml/build_draft_events.py`.

Из `picks_bans.parquet` и `matches.parquet` создается `draft_events.parquet`.

Добавляются команды, стороны, `action_type` и `state_id`. Неполные матчи фильтруются: остаются матчи с 24 действиями драфта.

## 4. Draft states

Скрипт: `src/ml/build_draft_states.py`.

Из `draft_events.parquet` создается `draft_states.parquet`.

Для каждого действия формируется состояние драфта до этого действия: уже выбранные союзные и вражеские герои, уже сделанные баны и список доступных героев.

## 5. Candidate tables

Скрипт: `src/ml/build_draft_candidates.py`.

Из `draft_states.parquet` создаются:

- `draft_candidates_pick.parquet`
- `draft_candidates_ban.parquet`

Для каждого состояния создается строка для каждого доступного героя-кандидата. `target = 1` только у реально выбранного или забаненного героя.

## 6. Model training

Скрипт: `src/ml/train_catboost.py`.

Отдельно обучаются:

- `pick_model`
- `ban_model`

Текущая модель: `CatBoostClassifier`.

Задача модели - присвоить score каждому герою-кандидату, после чего кандидаты ранжируются внутри каждого `state_id`.

## 7. Evaluation

Скрипт: `src/ml/evaluate.py`.

Метрики:

- `Top-1`
- `Top-3`
- `Top-5`
- `Top-10`
- `Mean Rank`
- `MRR`

## 8. Reports

Отчеты:

- `project_status.md`
- `reports/notebook_report.md`
- `docs/data_schema/*.md`

## Important methodology

- `train` / `valid` / `test` делятся по времени, а не случайно.
- Нельзя использовать post-match признаки как features для драфта.
- Нельзя использовать `acting_team_win`, `radiant_win`, `duration`, `kills`, `gpm` и похожие поля напрямую в модели.
- Текущая модель является baseline и пока сильно опирается на популярность героев.
- Следующий этап исследования: team-specific features, synergy/counter features, `player x hero` features.
