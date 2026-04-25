# Raw match JSON structure

Сырые JSON-файлы OpenDota матчей хранятся локально в:

```text
data/patch_60/matches/
```
Ниже описаны только поля, которые реально используются текущим проектом.

## 1. Top-level fields

| field | meaning | used in |
|---|---|---|
| `match_id` | Идентификатор матча | `matches.parquet`, `players.parquet`, `picks_bans.parquet` |
| `start_time` | Время начала матча | `matches.parquet`, split по времени |
| `duration` | Длительность матча | `matches.parquet`, расчет `kills_per_min`, не draft-feature |
| `radiant_win` | Победа Radiant | `matches.parquet`, `heroes_stats.parquet`, `draft_events.parquet`; не draft-feature |
| `patch` | Номер патча OpenDota | `matches.parquet`, ML features |
| `radiant_score` | Счет Radiant | `matches.parquet`, не draft-feature |
| `dire_score` | Счет Dire | `matches.parquet`, не draft-feature |
| `radiant_team_id` | ID команды Radiant | `matches.parquet`, `draft_events.parquet` |
| `radiant_name` / `radiant_team_name` | Название команды Radiant | `matches.parquet`, `draft_events.parquet` |
| `dire_team_id` | ID команды Dire | `matches.parquet`, `draft_events.parquet` |
| `dire_name` / `dire_team_name` | Название команды Dire | `matches.parquet`, `draft_events.parquet` |
| `league.name` | Название лиги | `matches.parquet`, ML features |
| `players` | Список игроков матча | `players.parquet`, составы команд в `matches.parquet`, `heroes_stats.parquet` |
| `picks_bans` | Список действий драфта | `picks_bans.parquet`, `draft_events.parquet` |
| `draft_timings` | Тайминги действий драфта | `picks_bans.parquet`, поле `total_time_taken` |

## 2. players[]

`players[]` - список игроков матча. Одна запись соответствует одному игроку.

| field | meaning | used in |
|---|---|---|
| `account_id` | ID аккаунта игрока | `players.parquet` |
| `name` | Никнейм игрока | `players.parquet` как `nickname` |
| `player_slot` | Слот игрока и сторона | `players.parquet`, определение Radiant/Dire |
| `hero_id` | ID героя игрока | `players.parquet`, составы команд, `heroes_stats.parquet` |
| `kills` | Убийства | `players.parquet`, не draft-feature |
| `deaths` | Смерти | `players.parquet`, не draft-feature |
| `assists` | Ассисты | `players.parquet`, не draft-feature |
| `last_hits` | Добивания крипов | `players.parquet`, не draft-feature |
| `denies` | Денаи | `players.parquet`, не draft-feature |
| `teamfight_participation` | Участие в драках | `players.parquet`, не draft-feature |
| `level` | Уровень героя | `players.parquet`, не draft-feature |
| `kills_per_min` | Убийства в минуту | `players.parquet`, не draft-feature |
| `net_worth` | Итоговая ценность героя | `players.parquet`, не draft-feature |
| `gold_per_min` / `gpm` | Золото в минуту | `players.parquet`, не draft-feature |
| `xp_per_min` / `xpm` | Опыт в минуту | `players.parquet`, не draft-feature |
| `hero_damage` | Урон по героям | `players.parquet`, не draft-feature |
| `tower_damage` | Урон по строениям | `players.parquet`, не draft-feature |
| `hero_healing` | Лечение героев | `players.parquet`, не draft-feature |

## 3. picks_bans[]

`picks_bans[]` - список действий драфта. Одна запись соответствует одному выбору или бану героя.

| field | meaning | used in |
|---|---|---|
| `order` | Порядок действия в драфте | `picks_bans.parquet`, `draft_events.parquet`, `draft_states.parquet` |
| `is_pick` | `true` для выбора, `false` для бана | `picks_bans.parquet`, `action_type` |
| `hero_id` | ID выбранного или забаненного героя | `picks_bans.parquet`, target в candidate tables |
| `team` | Сторона/команда, совершившая действие | `picks_bans.parquet`, расчет acting/opponent side |

## 4. draft_timings[]

| field | meaning | used in |
|---|---|---|
| `order` | Порядок действия драфта | связывание с `picks_bans[]` |
| `total_time_taken` | Время, затраченное на действие | `picks_bans.parquet` |

В текущем `patch_60` поле `total_time_taken` может отсутствовать или быть полностью пустым, поэтому оно не используется как основной признак модели.
