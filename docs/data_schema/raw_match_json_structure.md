# Raw match JSON structure

Сырые JSON-файлы OpenDota матчей хранятся локально в:

```text
data/patch_60/matches/
```

Они не коммитятся в GitHub. Ниже описаны только поля, которые реально используются текущим проектом.

Патч задается централизованно в `src/config.py`. OpenDota Explorer SQL использует patch label, например `7.41`, а raw JSON содержит числовой patch id, например `60`. Соответствие хранится в `PATCH_MAP`.

Основной список команд - ручной `TEAM_IDS` из `src/config.py`. Парсер фильтрует матчи по `matches.radiant_team_id` / `matches.dire_team_id`, а не по `notable_players.team_id`, чтобы матч попадал в выборку по реальной команде матча.

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
| `leagueid` | ID лиги | `matches.parquet` |
| `series_id` | ID серии | `matches.parquet` |
| `series_type` | Тип серии | `matches.parquet` |
| `game_mode` | Игровой режим | `matches.parquet` |
| `lobby_type` | Тип лобби | `matches.parquet` |
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

При конвертации `players.parquet` дополнительно получает поля `is_radiant`, `side`, `team_id`, `team_name`, `win`. Они нужны для будущих team-specific и player x hero признаков, но post-match поля вроде `win`, `kills`, `gpm`, `xpm`, `net_worth` не должны напрямую использоваться как draft-features.
