# Parquet Tables

Parquet-файлы являются локальными артефактами пайплайна и не хранятся в Git.

## Raw Tables

### `matches.parquet`

Одна строка - один матч.

Основные поля:

- `match_id`
- `start_time`
- `patch`
- `radiant_team_id`, `radiant_team_name`
- `dire_team_id`, `dire_team_name`
- `league_id`, `league_name`
- `radiant_win`

`radiant_win` и итоговые показатели матча не используются как признаки текущего draft state.

### `players.parquet`

Одна строка - один игрок в одном матче.

Основные поля:

- `match_id`
- `account_id`
- `nickname`
- `side`
- `team_id`, `team_name`
- `hero_id`, `hero_name`
- `kills`, `deaths`, `assists`
- `last_hits`, `net_worth`, `gold_per_min`, `xp_per_min`
- `hero_damage`, `tower_damage`, `hero_healing`
- `lane`, `lane_role`, `is_roaming`, `lane_efficiency`
- item fields such as `item_0` ... `item_5`
- `purchase_log_json`

Эта таблица используется для построения прошлой статистики игроков и empirical role priors. Post-match поля текущего матча не подаются напрямую в draft model.

### `picks_bans.parquet`

Одна строка - одно действие драфта.

Основные поля:

- `match_id`
- `order`
- `is_pick`
- `hero_id`, `hero_name`
- `team`
- `total_time_taken`, если доступно

### `heroes_stats.parquet`

Одна строка - один герой.

Основные поля:

- `hero_id`
- `hero_name`
- `matches_played`
- `wins`
- `winrate`
- `pick_rate`
- `ban_rate`
- `pick_or_ban_rate`

## ML Tables

### `draft_events.parquet`

Обогащенная последовательность действий драфта.

Основные поля:

- `state_id`
- `match_id`
- `order`
- `action_type`
- `hero_id`
- `acting_side`
- `opponent_side`
- `acting_team_id`
- `opponent_team_id`
- `start_time`

### `draft_states.parquet`

Состояние драфта перед каждым действием.

Основные поля:

- `state_id`
- `match_id`
- `order`
- `action_type`
- `chosen_hero_id`
- `ally_picks_before`
- `enemy_picks_before`
- `ally_bans_before`
- `enemy_bans_before`
- `available_heroes`
- `available_hero_count`

### Candidate Tables

Candidate tables строятся отдельно для pick и ban:

- `draft_candidates_pick.parquet`
- `draft_candidates_ban.parquet`

Расширенные варианты:

- `draft_candidates_*_interactions.parquet`
- `draft_candidates_*_interactions_role.parquet`
- `draft_candidates_*_players_team.parquet`
- `draft_candidates_*_players_team_role.parquet`

Одна строка - один доступный герой-кандидат в одном `state_id`.

Ключевые поля:

- `state_id`
- `candidate_hero_id`
- `candidate_hero_name`
- `target`
- `order`
- `action_type`
- feature columns

Для каждого `state_id` ровно одна строка имеет `target = 1`.
