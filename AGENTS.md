# AGENTS.md

Инструкция для Codex/агента, работающего с проектом `draft_aware_model`.

## Цель проекта

Проект строит ML-модель поддержки принятия решений на стадии драфта Dota 2. Модель должна ранжировать доступных героев-кандидатов отдельно для `pick` и `ban` в каждом состоянии драфта.

## Текущий рабочий патч

- `patch_60`

## Основные raw-таблицы

- `matches`
- `players`
- `picks_bans`
- `heroes_stats`

## Основные ML-таблицы

- `draft_events`
- `draft_states`
- `draft_candidates_pick`
- `draft_candidates_ban`

## Основной пайплайн

```text
build_draft_events.py -> build_draft_states.py -> build_draft_candidates.py -> train_catboost.py -> evaluate.py
```

## Важные правила

- Не использовать post-match признаки как draft-features.
- Не допускать target leakage.
- Делить `train` / `valid` / `test` по времени.
- Использовать ranking-метрики: `Top-1`, `Top-3`, `Top-5`, `Top-10`, `Mean Rank`, `MRR`.
- Не добавлять новые сложные признаки без явной задачи.
- Не коммитить raw JSON, parquet, `.cbm`, `.pkl`, `.joblib` и большие локальные данные.

## Стиль кода

- Простой `pandas` / `numpy`.
- Без лишних классов и преждевременных абстракций.
- Понятные функции с небольшим числом обязанностей.
- Сохранять текущую ML-логику, если задача явно не требует ее менять.
