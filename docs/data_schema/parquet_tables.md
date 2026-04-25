# Parquet tables

Сами parquet-файлы не хранятся в GitHub из-за размера и потому что они являются локальными артефактами пайплайна. Этот документ описывает структуру основных parquet-таблиц проекта.

Патчи централизованы в `src/config.py`: OpenDota SQL использует patch label, например `7.41`, а JSON матча использует patch id, например `60`. Соответствие хранится в `PATCH_MAP`.

## Raw Tables

### matches.parquet

**Назначение:** базовая таблица матчей.

**Одна строка:** один матч.

**Текущий размер:** `(345, 22)` в `project_status.md` до добавления новых служебных полей. После пересборки таблица также содержит `league_id`, `series_id`, `series_type`, `game_mode`, `lobby_type`.

**Ключевые поля:** `match_id`, `start_time`, `patch`, `radiant_team_id`, `dire_team_id`, `league_name`.

| column | meaning | type/expected type |
|---|---|---|
| `match_id` | Идентификатор матча OpenDota | integer |
| `start_time` | Время начала матча | integer timestamp |
| `duration` | Длительность матча в секундах | integer |
| `radiant_win` | Победила ли Radiant | boolean |
| `patch` | Номер патча OpenDota | integer |
| `radiant_score` | Счет Radiant по убийствам | integer |
| `dire_score` | Счет Dire по убийствам | integer |
| `radiant_team_id` | ID команды Radiant | integer/null |
| `radiant_team_name` | Название команды Radiant | string/null |
| `dire_team_id` | ID команды Dire | integer/null |
| `dire_team_name` | Название команды Dire | string/null |
| `league_id` | ID лиги OpenDota | integer/null |
| `league_name` | Название лиги | string |
| `series_id` | ID серии | integer/null |
| `series_type` | Тип серии | integer/null |
| `game_mode` | Игровой режим OpenDota | integer/null |
| `lobby_type` | Тип лобби OpenDota | integer/null |
| `radiant_hero_1` | Герой Radiant 1 | integer hero_id |
| `dire_hero_1` | Герой Dire 1 | integer hero_id |
| `radiant_hero_2` | Герой Radiant 2 | integer hero_id |
| `dire_hero_2` | Герой Dire 2 | integer hero_id |
| `radiant_hero_3` | Герой Radiant 3 | integer hero_id |
| `dire_hero_3` | Герой Dire 3 | integer hero_id |
| `radiant_hero_4` | Герой Radiant 4 | integer hero_id |
| `dire_hero_4` | Герой Dire 4 | integer hero_id |
| `radiant_hero_5` | Герой Radiant 5 | integer hero_id |
| `dire_hero_5` | Герой Dire 5 | integer hero_id |

### players.parquet

**Назначение:** статистика игроков в матчах.

**Одна строка:** один игрок в конкретном матче.

**Текущий размер:** `(3450, 20)` в `project_status.md` до добавления новых team/player context полей. После пересборки таблица также содержит `is_radiant`, `side`, `team_id`, `team_name`, `win`.

**Ключевые поля:** `match_id`, `account_id`, `player_slot`, `hero_id`.

| column | meaning | type/expected type |
|---|---|---|
| `match_id` | Идентификатор матча | integer |
| `account_id` | ID аккаунта игрока | integer |
| `nickname` | Никнейм игрока | string/null |
| `player_slot` | Слот игрока, определяет сторону | integer |
| `is_radiant` | Игрок на стороне Radiant | boolean |
| `side` | Сторона игрока: `radiant` или `dire` | string |
| `team_id` | ID команды игрока | integer/null |
| `team_name` | Название команды игрока | string/null |
| `win` | Победила ли команда игрока | boolean |
| `hero_id` | ID героя | integer |
| `hero_name` | Название героя | string |
| `kills` | Убийства | integer |
| `deaths` | Смерти | integer |
| `assists` | Ассисты | integer |
| `last_hits` | Добивания крипов | integer |
| `denies` | Денаи | integer |
| `teamfight_participation` | Участие в командных драках | float |
| `level` | Уровень героя к концу матча | integer |
| `kills_per_min` | Убийства в минуту | float |
| `net_worth` | Итоговая ценность героя | integer |
| `gold_per_min` | Золото в минуту | integer/float |
| `xp_per_min` | Опыт в минуту | integer/float |
| `hero_damage` | Урон по героям | integer |
| `tower_damage` | Урон по строениям | integer |
| `hero_healing` | Лечение героев | integer |

### picks_bans.parquet

**Назначение:** последовательность действий драфта.

**Одна строка:** одно действие драфта.

**Текущий размер:** `(8255, 7)`.

**Ключевые поля:** `match_id`, `order`, `is_pick`, `hero_id`, `team`.

| column | meaning | type/expected type |
|---|---|---|
| `match_id` | Идентификатор матча | integer |
| `order` | Порядок действия в драфте | integer |
| `is_pick` | `true` для выбора героя, `false` для запрета | boolean |
| `hero_id` | ID выбранного или забаненного героя | integer |
| `hero_name` | Название героя | string |
| `team` | Сторона/команда, совершившая действие | integer |
| `total_time_taken` | Время, затраченное на действие | float/null |

### heroes_stats.parquet

**Назначение:** агрегированная статистика героев по текущему набору матчей.

**Одна строка:** один герой.

**Текущий размер:** `(118, 8)`.

**Ключевые поля:** `hero_id`, `hero_name`.

| column | meaning | type/expected type |
|---|---|---|
| `hero_id` | ID героя | integer |
| `hero_name` | Название героя | string |
| `matches_played` | Количество матчей героя | integer |
| `wins` | Количество побед героя | integer |
| `winrate` | Доля побед | float |
| `pick_rate` | Частота выбора героя | float |
| `ban_rate` | Частота бана героя | float |
| `pick_or_ban_rate` | Суммарная частота выбора или бана | float |

## ML Tables

### draft_events.parquet

**Назначение:** обогащенные события драфта, построенные из `picks_bans.parquet` и `matches.parquet`.

**Одна строка:** одно событие драфта.

**Текущий размер:** `(8232, 24)`.

**Ключевые поля:** `state_id`, `match_id`, `order`, `action_type`, `hero_id`, `acting_team_id`, `opponent_team_id`.

`acting_team_win` описывает результат матча для команды, которая совершила действие. Это post-match поле и не должно использоваться как признак модели.

| column | meaning | type/expected type |
|---|---|---|
| `match_id` | Идентификатор матча | integer |
| `order` | Порядок действия в драфте | integer |
| `is_pick` | Было ли действие выбором героя | boolean |
| `hero_id` | ID героя в действии | integer |
| `hero_name` | Название героя | string |
| `team` | Код стороны/команды из raw draft action | integer |
| `total_time_taken` | Время действия, если доступно | float/null |
| `start_time` | Время начала матча | integer timestamp |
| `patch` | Номер патча OpenDota | integer |
| `league_name` | Название лиги | string |
| `radiant_win` | Победила ли Radiant | boolean |
| `radiant_team_id` | ID команды Radiant | integer/null |
| `radiant_team_name` | Название команды Radiant | string/null |
| `dire_team_id` | ID команды Dire | integer/null |
| `dire_team_name` | Название команды Dire | string/null |
| `acting_side` | Сторона, совершившая действие | string |
| `opponent_side` | Противоположная сторона | string |
| `acting_team_id` | ID команды, совершившей действие | integer/null |
| `acting_team_name` | Название команды, совершившей действие | string/null |
| `opponent_team_id` | ID команды-соперника | integer/null |
| `opponent_team_name` | Название команды-соперника | string/null |
| `acting_team_win` | Победила ли команда, совершившая действие | boolean |
| `action_type` | Тип действия: `pick` или `ban` | string |
| `state_id` | ID состояния драфта | string |

### draft_states.parquet

**Назначение:** состояния драфта перед каждым действием.

**Одна строка:** состояние драфта перед конкретным действием.

**Текущий размер:** `(8232, 27)`.

**Ключевые поля:** `state_id`, `match_id`, `order`, `action_type`, `chosen_hero_id`, `available_heroes`.

Списки `*_before` описывают состояние до текущего действия, то есть не включают текущего выбранного или забаненного героя.

| column | meaning | type/expected type |
|---|---|---|
| `state_id` | ID состояния драфта | string |
| `match_id` | Идентификатор матча | integer |
| `order` | Порядок текущего действия | integer |
| `draft_phase` | Фаза драфта | integer |
| `action_type` | Тип текущего действия: `pick` или `ban` | string |
| `is_pick` | Было ли текущее действие выбором героя | boolean |
| `chosen_hero_id` | Герой, реально выбранный или забаненный в этом состоянии | integer |
| `chosen_hero_name` | Название выбранного или забаненного героя | string |
| `acting_side` | Сторона, совершающая действие | string |
| `acting_team_id` | ID команды, совершающей действие | integer/null |
| `acting_team_name` | Название команды, совершающей действие | string/null |
| `opponent_team_id` | ID команды-соперника | integer/null |
| `opponent_team_name` | Название команды-соперника | string/null |
| `patch` | Номер патча OpenDota | integer |
| `league_name` | Название лиги | string |
| `start_time` | Время начала матча | integer timestamp |
| `acting_team_win` | Победила ли команда, совершающая действие | boolean |
| `ally_picks_before` | Герои союзников, выбранные до текущего действия | list[int] |
| `enemy_picks_before` | Герои соперника, выбранные до текущего действия | list[int] |
| `ally_bans_before` | Баны союзной команды до текущего действия | list[int] |
| `enemy_bans_before` | Баны соперника до текущего действия | list[int] |
| `n_ally_picks_before` | Число союзных пиков до действия | integer |
| `n_enemy_picks_before` | Число вражеских пиков до действия | integer |
| `n_ally_bans_before` | Число союзных банов до действия | integer |
| `n_enemy_bans_before` | Число вражеских банов до действия | integer |
| `available_heroes` | Список героев, доступных до действия | list[int] |
| `available_hero_count` | Количество доступных героев | integer |

### draft_candidates_pick.parquet

**Назначение:** candidate table для задачи ранжирования pick-кандидатов.

**Одна строка:** один доступный герой-кандидат в конкретном состоянии драфта для действия `pick`.

**Текущий размер:** `(385189, 24)`.

**Ключевые поля:** `state_id`, `candidate_hero_id`, `target`.

`target = 1`, если этот герой был реально выбран в этом состоянии, иначе `0`.

| column | meaning | type/expected type |
|---|---|---|
| `state_id` | ID состояния драфта | string |
| `match_id` | Идентификатор матча | integer |
| `order` | Порядок действия | integer |
| `draft_phase` | Фаза драфта | integer |
| `action_type` | Тип действия, здесь `pick` | string |
| `acting_side` | Сторона, совершающая действие | string |
| `acting_team_id` | ID команды, совершающей действие | integer/null |
| `opponent_team_id` | ID команды-соперника | integer/null |
| `patch` | Номер патча OpenDota | integer |
| `league_name` | Название лиги | string |
| `start_time` | Время начала матча | integer timestamp |
| `n_ally_picks_before` | Число союзных пиков до действия | integer |
| `n_enemy_picks_before` | Число вражеских пиков до действия | integer |
| `n_ally_bans_before` | Число союзных банов до действия | integer |
| `n_enemy_bans_before` | Число вражеских банов до действия | integer |
| `available_hero_count` | Количество доступных героев | integer |
| `candidate_hero_id` | ID героя-кандидата | integer |
| `target` | Целевая метка для кандидата | integer, 0/1 |
| `candidate_hero_name` | Название героя-кандидата | string |
| `candidate_matches_played` | Сколько матчей сыграно героем в датасете | integer/float |
| `candidate_winrate` | Winrate героя в датасете | float |
| `candidate_pick_rate` | Pick rate героя в датасете | float |
| `candidate_ban_rate` | Ban rate героя в датасете | float |
| `candidate_pick_or_ban_rate` | Pick-or-ban rate героя в датасете | float |

### draft_candidates_ban.parquet

**Назначение:** candidate table для задачи ранжирования ban-кандидатов.

**Одна строка:** один доступный герой-кандидат в конкретном состоянии драфта для действия `ban`.

**Текущий размер:** `(565607, 24)`.

**Ключевые поля:** `state_id`, `candidate_hero_id`, `target`.

`target = 1`, если этот герой был реально забанен в этом состоянии, иначе `0`.

| column | meaning | type/expected type |
|---|---|---|
| `state_id` | ID состояния драфта | string |
| `match_id` | Идентификатор матча | integer |
| `order` | Порядок действия | integer |
| `draft_phase` | Фаза драфта | integer |
| `action_type` | Тип действия, здесь `ban` | string |
| `acting_side` | Сторона, совершающая действие | string |
| `acting_team_id` | ID команды, совершающей действие | integer/null |
| `opponent_team_id` | ID команды-соперника | integer/null |
| `patch` | Номер патча OpenDota | integer |
| `league_name` | Название лиги | string |
| `start_time` | Время начала матча | integer timestamp |
| `n_ally_picks_before` | Число союзных пиков до действия | integer |
| `n_enemy_picks_before` | Число вражеских пиков до действия | integer |
| `n_ally_bans_before` | Число союзных банов до действия | integer |
| `n_enemy_bans_before` | Число вражеских банов до действия | integer |
| `available_hero_count` | Количество доступных героев | integer |
| `candidate_hero_id` | ID героя-кандидата | integer |
| `target` | Целевая метка для кандидата | integer, 0/1 |
| `candidate_hero_name` | Название героя-кандидата | string |
| `candidate_matches_played` | Сколько матчей сыграно героем в датасете | integer/float |
| `candidate_winrate` | Winrate героя в датасете | float |
| `candidate_pick_rate` | Pick rate героя в датасете | float |
| `candidate_ban_rate` | Ban rate героя в датасете | float |
| `candidate_pick_or_ban_rate` | Pick-or-ban rate героя в датасете | float |

### hero_synergy.parquet

**Назначение:** таблица парных синергий героев, построенная отдельно от Streamlit GUI.

**Одна строка:** unordered пара героев в одной команде.

**Ключевые поля:** `hero1_id`, `hero2_id`.

Для строгой финальной ML-методологии такие агрегаты нужно считать только на train-части или в rolling/as-of режиме, чтобы избежать leakage.

| column | meaning | type/expected type |
|---|---|---|
| `hero1_id` | ID первого героя пары | integer |
| `hero2_id` | ID второго героя пары | integer |
| `games` | Количество совместных игр | integer |
| `wins` | Количество побед пары | integer |
| `pair_winrate` | Winrate пары | float |
| `hero1_wr` | Общий winrate первого героя | float |
| `hero2_wr` | Общий winrate второго героя | float |
| `baseline_winrate` | Средний baseline winrate двух героев | float |
| `synergy_delta` | `pair_winrate - baseline_winrate` | float |
| `delta_hero1` | `pair_winrate - hero1_wr`, EDA поле | float |
| `delta_hero2` | `pair_winrate - hero2_wr`, EDA поле | float |
| `pair_pick_freq` | Частота совместного появления пары | float |
| `p_value` | p-value биномиального теста | float/null |
| `significant` | `p_value < alpha` | boolean |
| `hero1_name` | Название первого героя | string |
| `hero2_name` | Название второго героя | string |

### hero_matchups.parquet

**Назначение:** таблица направленных matchup/counter статистик.

**Одна строка:** направленная пара `hero_id` против `vs_hero_id`.

**Ключевые поля:** `hero_id`, `vs_hero_id`.

| column | meaning | type/expected type |
|---|---|---|
| `hero_id` | ID героя | integer |
| `vs_hero_id` | ID героя-соперника | integer |
| `games` | Количество игр matchup | integer |
| `wins` | Победы `hero_id` против `vs_hero_id` | integer |
| `matchup_winrate` | Winrate в matchup | float |
| `hero_winrate` | Общий winrate героя | float |
| `counter_delta` | `matchup_winrate - hero_winrate` | float |
| `p_value_win` | p-value биномиального теста | float/null |
| `significant_win` | `p_value_win < alpha` | boolean |
| `hero_name` | Название героя | string |
| `vs_hero_name` | Название героя-соперника | string |

### hero_conditional_bans.parquet

**Назначение:** направленная аналитическая таблица условных банов.

**Одна строка:** пара `picked_hero_id` и `banned_hero_id`.

**Ключевые поля:** `picked_hero_id`, `banned_hero_id`.

| column | meaning | type/expected type |
|---|---|---|
| `picked_hero_id` | ID пикнутого героя | integer |
| `banned_hero_id` | ID забаненного героя | integer |
| `picked_games` | Число матчей, где герой был пикнут | integer |
| `bans_by_picked_team` | Сколько раз бан сделала команда пикнутого героя | integer |
| `bans_by_opponent_team` | Сколько раз бан сделал соперник | integer |
| `ban_rate_by_picked_team` | Условная частота бана своей командой | float |
| `ban_rate_by_opponent_team` | Условная частота бана соперником | float |
| `picked_hero_name` | Название пикнутого героя | string |
| `banned_hero_name` | Название забаненного героя | string |

### draft_candidates_pick_interactions.parquet / draft_candidates_ban_interactions.parquet

**Назначение:** расширенные candidate tables с interaction features.

**Одна строка:** тот же кандидат, что и в base candidate table, плюс агрегаты синергий и контрпиков относительно `draft_states`.

Эти таблицы строятся скриптом `src/ml/add_interaction_features.py` и пока не заменяют baseline автоматически; обучение выбирает вариант через `--dataset base` или `--dataset interactions`.
