# draft_aware_model

ML project for Dota 2 draft recommendations. The model ranks available hero candidates separately for `pick` and `ban` draft actions.

## Current Architecture

The project keeps one compact ML pipeline:

```text
raw patch parquet
-> draft_events
-> draft_states
-> draft_candidates_{pick,ban}
-> interactions
-> players
-> CatBoostRanker
-> reports/ml
-> local Streamlit UI
```

Current patch: `7.41` / `patch_60`.

Active datasets:

- `base`: draft state + candidate hero meta statistics.
- `interactions`: `base` + hero synergy/counter features.
- `players`: `interactions` + roster and player-hero comfort features.

`players` is the main UI dataset.

## Commands

Build tables without training:

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset players --skip-train
```

Train and evaluate a ranker:

```bash
python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset players
python -m src.ml.evaluate --patch-label 7.41 --action pick --dataset players

python -m src.ml.train_catboost --patch-label 7.41 --action ban --dataset players
python -m src.ml.evaluate --patch-label 7.41 --action ban --dataset players
```

Export report tables:

```bash
python -m src.ml.export_reports --patch-label 7.41
```

Run local UI:

```bash
streamlit run src/ui/app.py
```

Download hero icons for local UI cache:

```bash
python -m src.ui.download_hero_icons
```

## Important Files

- `src/config.py`: patch mapping and path helpers.
- `src/ml/feature_sets.py`: single source of truth for datasets and model features.
- `src/ml/build_draft_events.py`: build enriched draft events.
- `src/ml/build_draft_states.py`: build sequential draft states.
- `src/ml/build_draft_candidates.py`: build base candidate tables.
- `src/ml/add_interaction_features.py`: add synergy/counter features.
- `src/ml/add_player_features.py`: add player and player-hero features.
- `src/ml/train_catboost.py`: train CatBoostRanker.
- `src/ml/evaluate.py`: calculate ranking metrics.
- `src/ui/app.py`: Streamlit UI.
- `src/ui/feature_builder.py`: UI inference feature rows.
- `src/ui/recommender.py`: model loading and scoring for UI.

## Artifacts

Generated data and models are local artifacts and should not be committed:

- raw JSON;
- parquet tables;
- `.cbm` models;
- local hero icons under `data/hero_icons/`.

Commit-friendly ML outputs live in:

```text
reports/ml/patch_60/
```

Subdirectories:

- `metrics/`
- `features/`
- `importance/`
- `errors/`
- `recommendations/`

## Cleanup

Dry-run:

```bash
python -m src.ml.cleanup_tables
```

Apply deletion of generated artifacts:

```bash
python -m src.ml.cleanup_tables --apply
```

The cleanup script keeps raw patch parquet tables:

- `matches.parquet`
- `players.parquet`
- `picks_bans.parquet`
- `heroes_stats.parquet`

and hero reference files:

- `data/heroes.csv`
- `data/heroes.json`
