# Raw Match JSON Structure

OpenDota match JSON сохраняется локально в:

```text
data/patch_60/matches/
```

Файлы не коммитятся. Ниже перечислены поля, которые используются текущим проектом.

## Top-Level Fields

| field | usage |
|---|---|
| `match_id` | ключ матча |
| `start_time` | split по времени и хронологические агрегаты |
| `duration` | raw table only, не draft-feature |
| `radiant_win` | evaluation/aggregates only, не draft-feature |
| `patch` | фильтрация патча |
| `radiant_team_id`, `dire_team_id` | команды матча |
| `radiant_name`, `dire_name` | названия команд |
| `league`, `leagueid` | турнир |
| `series_id`, `series_type` | серия |
| `game_mode`, `lobby_type` | служебные поля матча |
| `players` | составы и статистика игроков |
| `picks_bans` | действия драфта |
| `draft_timings` | тайминги действий, если доступны |

## `players[]`

Одна запись соответствует одному игроку.

Используемые поля:

- `account_id`
- `name`
- `player_slot`
- `hero_id`
- `kills`, `deaths`, `assists`
- `last_hits`, `denies`
- `net_worth`
- `gold_per_min`, `xp_per_min`
- `hero_damage`, `tower_damage`, `hero_healing`
- `obs_placed`, `sen_placed`
- `camps_stacked`, `rune_pickups`, `stuns`
- `lane`, `lane_role`, `is_roaming`, `lane_efficiency`
- `item_0` ... `item_5`
- `backpack_0` ... `backpack_2`
- `item_neutral`
- `purchase_log`

`purchase_log` сохраняется как compact JSON string (`purchase_log_json`), чтобы parquet schema оставалась простой.

## `picks_bans[]`

Одна запись соответствует одному выбору или бану героя.

Используемые поля:

- `order`: порядок действия в драфте;
- `is_pick`: `true` для pick, `false` для ban;
- `hero_id`: выбранный или забаненный герой;
- `team`: сторона/команда, совершившая действие.

## `draft_timings[]`

Если OpenDota возвращает тайминги, используется:

- `order`
- `total_time_taken`

В текущем patch data это поле может отсутствовать или быть пустым, поэтому оно не является основным признаком модели.
