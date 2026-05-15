# draft-aware-model

Курсовой ML-проект для поддержки решений на стадии драфта в Dota 2.

В Dota 2 перед матчем две команды по очереди запрещают (`ban`) и выбирают (`pick`) героев. После запрета герой становится недоступен обеим командам, а после выбора закрепляется за одной из команд. Цель проекта - по текущему состоянию драфта ранжировать доступных героев и подсказать, кого профессиональная команда с наибольшей вероятностью выберет или забанит.

Публичная Streamlit-версия:

```text
https://draft-aware-model.streamlit.app
```

Текущий рабочий патч: `7.41` / `patch_60`.

## Задача

Для каждого состояния драфта строится список доступных героев-кандидатов. Модель решает две отдельные ranking-задачи:

- `pick`: какой герой будет выбран командой;
- `ban`: какой герой будет запрещен командой.

Результат модели - не один класс, а упорядоченный список героев. Поэтому качество оценивается ranking-метриками: `Top-1`, `Top-3`, `Top-5`, `Top-10`, `Mean Rank`, `MRR`.

## Данные

Источник данных - локально сохраненные матчи OpenDota. В репозиторий не коммитятся raw JSON, parquet-таблицы и обученные модели, потому что это локальные генерируемые артефакты.

Основные таблицы:

- `matches.parquet`: матчи, команды, патч, время начала;
- `players.parquet`: игроки, выбранные герои и статистика матча;
- `picks_bans.parquet`: последовательность драфта;
- `heroes_stats.parquet`: агрегированная статистика героев;
- `draft_states.parquet`: состояние драфта перед каждым действием;
- `draft_candidates_pick*.parquet` и `draft_candidates_ban*.parquet`: таблицы кандидатов для обучения.

Важно: post-match поля вроде результата матча, KDA, GPM, XPM текущей игры не используются напрямую как признаки текущего draft state. Train/valid/test делятся по времени.

## Модели

Активные датасеты:

- `base`: базовая статистика героя и состояние драфта.
- `interactions`: `base` + признаки синергии и матчапов героев.
- `interactions_role`: универсальная модель для неизвестных Team A / Team B; добавляет role-state признаки без командных и player-specific признаков.
- `players_team`: `interactions` + статистика игрок-герой и team-priority признаки.
- `players_team_role`: `players_team` + эмпирические role/position-aware признаки.

Основная модель проекта:

```text
players_team_role
```

Для двух неизвестных команд в UI лучше использовать `interactions_role`: она не требует известных roster/team_id, но учитывает уже закрытые и открытые роли в драфте.

## Технологии

- Python
- pandas / numpy
- CatBoostRanker
- Streamlit
- OpenDota match data

## Установка

```bash
pip install -r requirements.txt
```

## Полная пересборка данных

Загрузить недостающие матчи текущего патча, конвертировать JSON в parquet и пересобрать базовые ML-таблицы:

```bash
python -m src.data_update --patch-label 7.41 --fetch --convert --build-ml --rebuild
```

`--fetch` пропускает уже существующие JSON. Для принудительного обновления локальных JSON используется `--refresh-existing`.

## Роли героев

`data/hero_roles.csv` строится из локальных OpenDota-матчей. Роли не задаются вручную и не берутся из сторонних сервисов.

```bash
python -m src.ml.build_hero_roles --patch-labels 7.36 7.37 7.38 7.39 7.40 7.41
python -m src.ml.inspect_hero_roles --patch-label 7.41
python -m src.ml.inspect_inferred_positions --patch-label 7.41
```

Роли оцениваются по lane-полям, стороне Radiant/Dire, статистике игрока, предметам, прошлым позициям игрока и оптимальному назначению пяти игроков на пять позиций.

## Обучение

Обучить и оценить все активные модели:

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset all
```

Обучить только основную модель:

```bash
python -m src.ml.run_pipeline --patch-label 7.41 --dataset players_team_role
```

Отдельные команды для pick/ban:

```bash
python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset players_team_role
python -m src.ml.evaluate --patch-label 7.41 --action pick --dataset players_team_role

python -m src.ml.train_catboost --patch-label 7.41 --action ban --dataset players_team_role
python -m src.ml.evaluate --patch-label 7.41 --action ban --dataset players_team_role
python -m src.ml.export_reports --patch-label 7.41
```

## Отчеты

Основные отчеты сохраняются в:

```text
reports/ml/patch_60/
```

Полезные файлы:

- `metrics/model_metrics.md`: итоговые метрики по моделям;
- `metrics/model_comparison.md`: сравнение соседних моделей;
- `importance/feature_importance.md`: важность признаков;
- `errors/model_error_analysis.md`: состояния, где новая модель улучшила или ухудшила rank реального героя.

Сравнить две модели:

```bash
python -m src.ml.error_analysis --patch-label 7.41 --left-dataset players_team --right-dataset players_team_role
```

## UI

Запуск локального Streamlit UI:

```bash
streamlit run src/ui/app.py
```

В интерфейсе можно:

- выбирать модель;
- собирать текущий драфт;
- получать рекомендации для pick или ban;
- открывать подробности по каждому предложенному герою;
- смотреть факторы рекомендации: мета, синергия, роли, приоритет команды, статистика игроков на герое.

Для локальных hero icons:

```bash
python -m src.ui.download_hero_icons
```

## Структура проекта

```text
src/config.py                    # patch mapping, paths, team ids
src/data_update.py               # общий сценарий загрузки и пересборки таблиц
src/convert_to_parquet.py        # JSON -> parquet
src/ml/                          # ML pipeline, features, training, reports
src/ui/                          # Streamlit-приложение
data/hero_roles.csv              # commit-friendly empirical hero role priors
docs/data_schema/                # краткая схема данных
reports/ml/patch_60/             # commit-friendly отчеты
```

Сырые данные и модели игнорируются `.gitignore`: `data/patch_*`, parquet, JSON, `.cbm`, локальные картинки героев и `catboost_info/`.

## Очистка локальных артефактов

Посмотреть, что будет удалено:

```bash
python -m src.ml.cleanup_tables
```

Удалить сгенерированные ML-таблицы и отчеты, не трогая raw parquet:

```bash
python -m src.ml.cleanup_tables --apply
```
