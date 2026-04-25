# Data pipeline

Параметры патча централизованы в `src/config.py`. `PATCH_LABEL` задает OpenDota patch label, например `7.41`; `PATCH_MAP` сопоставляет его с JSON patch id, например `60`; пути `BASE_DIR`, `MATCH_DIR`, `ML_DIR`, `MODEL_DIR` строятся из этого соответствия.

## 1. Raw JSON loading

Источник: OpenDota.

Скрипт: `src/parser_explorer.py`.

Результат: локальные JSON-файлы в:

```text
data/patch_60/matches/
```

Эти файлы не коммитятся.

Запуск:

```bash
python -m src.parser_explorer --patch-label 7.41
python -m src.parser_explorer --patch-label 7.41 --limit 20
python -m src.parser_explorer --patch-label 7.41 --refresh-existing
```

OpenDota SQL фильтрует профессиональные матчи по реальным командам матча: `matches.radiant_team_id` / `matches.dire_team_id` должны входить в ручной список `TEAM_IDS`. Старый фильтр через `notable_players.team_id` не используется как основной, потому что он может добавлять шумные матчи по игрокам.

Перед сохранением JSON проверяется:

- JSON patch id соответствует `PATCH_MAP`;
- одна из команд матча входит в `TEAM_IDS`;
- в драфте 24 действия;
- в матче 10 игроков.

Парсер пишет CSV-логи `match_ids_patch_60.csv` и `parser_log_patch_60.csv` в локальную папку `data/patch_60/`.

## 2. Conversion to raw parquet tables

Скрипт: `src/convert_to_parquet.py`.

Результаты:

- `matches.parquet`
- `players.parquet`
- `picks_bans.parquet`
- `heroes_stats.parquet`

Запуск:

```bash
python -m src.convert_to_parquet --patch-label 7.41
python -m src.convert_to_parquet --patch-label 7.41 --rebuild
python -m src.convert_to_parquet --patch-label 7.41 --only-new
```

По умолчанию конвертация incremental: добавляются только новые JSON, которых еще нет в `matches.parquet`. `--rebuild` полностью пересобирает raw parquet tables.

На этом шаге сырые JSON матчей превращаются в плоские таблицы матчей, игроков, действий драфта и агрегированной статистики героев.

`matches.parquet` теперь дополнительно хранит `league_id`, `series_id`, `series_type`, `game_mode`, `lobby_type`.

`players.parquet` теперь дополнительно хранит `is_radiant`, `side`, `team_id`, `team_name`, `win` для будущих team/player features.

`total_time_taken` может быть пустым в patch 60 и не используется как основной признак модели.

## 3. Draft events

Скрипт: `src/ml/build_draft_events.py`.

Из `picks_bans.parquet` и `matches.parquet` создается `draft_events.parquet`.

Добавляются команды, стороны, `action_type` и `state_id`. Неполные матчи фильтруются: остаются матчи с 24 действиями драфта.

## 4. Draft states

Скрипт: `src/ml/build_draft_states.py`.

Из `draft_events.parquet` создается `draft_states.parquet`.

Для каждого действия формируется состояние драфта до этого действия: уже выбранные союзные и вражеские герои, уже сделанные баны и список доступных героев.

Внутреннее накопление состояния идет по стороне `radiant` / `dire`, а не по `team_id`, потому что `team_id` иногда отсутствует.

## 5. Candidate tables

Скрипт: `src/ml/build_draft_candidates.py`.

Из `draft_states.parquet` создаются:

- `draft_candidates_pick.parquet`
- `draft_candidates_ban.parquet`

Для каждого состояния создается строка для каждого доступного героя-кандидата. `target = 1` только у реально выбранного или забаненного героя.

## 6. Interaction tables

Расчет синергий, контрпиков и условных банов вынесен из Streamlit в чистый ML-модуль:

```bash
python -m src.ml.interaction_tables --patch-label 7.41
python -m src.ml.add_interaction_features --patch-label 7.41
```

Результаты:

- `hero_synergy.parquet`
- `hero_matchups.parquet`
- `hero_conditional_bans.parquet`
- `draft_candidates_pick_interactions.parquet`
- `draft_candidates_ban_interactions.parquet`

Streamlit GUI теперь читает готовые interaction tables или строит их через функции из `src/ml/interaction_tables.py`.

## 7. Model training

Скрипт: `src/ml/train_catboost.py`.

Отдельно обучаются:

- `pick_base_model`
- `ban_base_model`
- `pick_interactions_model`
- `ban_interactions_model`

Текущая модель: `CatBoostClassifier`.

Задача модели - присвоить score каждому герою-кандидату, после чего кандидаты ранжируются внутри каждого `state_id`.

Baseline dataset:

```bash
python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset base
python -m src.ml.train_catboost --patch-label 7.41 --action ban --dataset base
```

Interaction dataset:

```bash
python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset interactions
python -m src.ml.train_catboost --patch-label 7.41 --action ban --dataset interactions
```

## 8. Evaluation

Скрипт: `src/ml/evaluate.py`.

Метрики:

- `Top-1`
- `Top-3`
- `Top-5`
- `Top-10`
- `Mean Rank`
- `MRR`

## 9. Combined commands

```bash
python -m src.data_update --patch-label 7.41 --fetch --convert --build-ml
python -m src.data_update --patch-label 7.41 --convert --rebuild
python -m src.ml.run_pipeline --patch-label 7.41 --dataset base
python -m src.ml.run_pipeline --patch-label 7.41 --dataset interactions
```

## 10. Reports

Отчеты:

- `project_status.md`
- `reports/notebook_report.md`
- `docs/data_schema/*.md`

## Important methodology

- `train` / `valid` / `test` делятся по времени, а не случайно.
- Нельзя использовать post-match признаки как features для драфта.
- Нельзя использовать `acting_team_win`, `radiant_win`, `duration`, `kills`, `gpm` и похожие поля напрямую в модели.
- `total_time_taken` не используется как основной признак модели, потому что в момент рекомендации заранее неизвестно, сколько времени команда потратит на решение.
- Текущая модель является baseline и пока сильно опирается на популярность героев.
- Следующий этап исследования: team-specific features, synergy/counter features, `player x hero` features.
- Для финальной строгой ML-методологии interaction tables должны считаться только на train-части или в rolling/as-of режиме, чтобы не было data leakage. Текущий код подготавливает архитектуру для следующего этапа.
