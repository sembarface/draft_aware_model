# Data Pipeline

Этот документ кратко описывает путь данных от OpenDota JSON до обученной ranking-модели.

## 1. Raw JSON

Матчи загружаются из OpenDota и сохраняются локально:

```text
data/patch_60/matches/
```

JSON-файлы не коммитятся. Загрузка пропускает уже существующие файлы, если не указан `--refresh-existing`.

```bash
python -m src.data_update --patch-label 7.41 --fetch
```

## 2. Raw parquet

`src/convert_to_parquet.py` преобразует JSON в плоские parquet-таблицы:

- `matches.parquet`
- `players.parquet`
- `picks_bans.parquet`
- `heroes_stats.parquet`

```bash
python -m src.convert_to_parquet --patch-label 7.41 --rebuild
```

## 3. Draft events and states

`src/ml/build_draft_events.py` создает последовательность действий драфта, а `src/ml/build_draft_states.py` восстанавливает состояние перед каждым действием:

- уже выбранные союзные и вражеские герои;
- уже сделанные баны;
- доступные герои;
- команда, которая сейчас делает `pick` или `ban`.

## 4. Candidate tables

`src/ml/build_draft_candidates.py` строит candidate tables:

- `draft_candidates_pick.parquet`
- `draft_candidates_ban.parquet`

Для каждого draft state создается строка на каждого доступного героя. `target = 1` стоит только у реально выбранного или забаненного героя.

## 5. Feature layers

Активные feature layers:

- `base`: мета-статистика героя и состояние драфта;
- `interactions`: синергии и матчапы;
- `interactions_role`: role-state признаки для универсальной модели без команд;
- `players_team`: player-hero comfort и team-priority признаки;
- `players_team_role`: empirical role priors и role-fit признаки.

## 6. Training and evaluation

Модели обучаются через `CatBoostRanker` отдельно для `pick` и `ban`.

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset all
```

Train/valid/test делятся по времени. Evaluation всегда считается на полном test split.

## 7. Reports and UI

Отчеты сохраняются в:

```text
reports/ml/patch_60/
```

Streamlit UI использует те же feature builders и обученные модели:

```bash
streamlit run src/ui/app.py
```

## Methodology Notes

- Не использовать post-match статистику текущей игры как draft-feature.
- Не использовать test split для подбора признаков или параметров.
- Роли героев строятся только из локальных OpenDota данных.
- Raw JSON, parquet и модели являются локальными артефактами и не коммитятся.
