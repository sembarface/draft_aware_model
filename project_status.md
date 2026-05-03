# Project Status

## Current Scope

The project is kept intentionally compact.

Current production comparison datasets:

- `base`
- `interactions`
- `players`

The active model family is `CatBoostRanker`. The main local UI dataset is `players`.

Removed from active scope: classifier models, large `data/stats/**` architecture experiments, and extra history/safe/team-history variants.

## Current Pipeline

```text
build_draft_events
-> build_draft_states
-> build_player_stats
-> build_draft_candidates
-> add_interaction_features
-> add_player_features
-> train_catboost
-> evaluate
-> export_reports
```

Build tables without training:

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset players --skip-train
```

Train/evaluate main models:

```bash
python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset players
python -m src.ml.evaluate --patch-label 7.41 --action pick --dataset players

python -m src.ml.train_catboost --patch-label 7.41 --action ban --dataset players
python -m src.ml.evaluate --patch-label 7.41 --action ban --dataset players
```

## Reports

Commit-friendly outputs are stored under:

```text
reports/ml/patch_60/
```

Models are stored locally under:

```text
models/patch_60/
```

## Local UI

Run:

```bash
streamlit run src/ui/app.py
```

The UI uses:

- `models/patch_60/pick_players_model.cbm`
- `models/patch_60/ban_players_model.cbm`
- `reports/ml/patch_60/features/*_players_features.json`
- `data/patch_60/ml/draft_candidates_*_players.parquet`

Hero icons are cached locally under `data/hero_icons/` and are ignored by git.

## Notes

- `candidate_hero_id` is used by CatBoost as a categorical signal, but UI explanations ignore it as a human-readable reason.
- Recommendations are ranker scores, not win-probability estimates.
- UI explanations are heuristic summaries of readable feature values, not exact SHAP explanations.
