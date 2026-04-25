# Project status
## Patch
`PATCH = 60`
## Tables
### matches
- path: `data\patch_60\matches.parquet`
- shape: `(345, 22)`

Columns:
```text
match_id
start_time
duration
radiant_win
patch
radiant_score
dire_score
radiant_team_id
radiant_team_name
dire_team_id
dire_team_name
league_name
radiant_hero_1
dire_hero_1
radiant_hero_2
dire_hero_2
radiant_hero_3
dire_hero_3
radiant_hero_4
dire_hero_4
radiant_hero_5
dire_hero_5
```

Top missing values:
```text
dire_team_name: 5
dire_team_id: 5
radiant_team_id: 4
radiant_team_name: 4
match_id: 0
start_time: 0
radiant_score: 0
patch: 0
radiant_win: 0
duration: 0
dire_score: 0
league_name: 0
radiant_hero_1: 0
dire_hero_1: 0
radiant_hero_2: 0
```
### players
- path: `data\patch_60\players.parquet`
- shape: `(3450, 20)`

Columns:
```text
match_id
account_id
nickname
player_slot
hero_id
hero_name
kills
deaths
assists
last_hits
denies
teamfight_participation
level
kills_per_min
net_worth
gold_per_min
xp_per_min
hero_damage
tower_damage
hero_healing
```

Top missing values:
```text
nickname: 864
match_id: 0
account_id: 0
player_slot: 0
hero_id: 0
hero_name: 0
kills: 0
deaths: 0
assists: 0
last_hits: 0
denies: 0
teamfight_participation: 0
level: 0
kills_per_min: 0
net_worth: 0
```
### picks_bans
- path: `data\patch_60\picks_bans.parquet`
- shape: `(8255, 7)`

Columns:
```text
match_id
order
is_pick
hero_id
hero_name
team
total_time_taken
```

Top missing values:
```text
total_time_taken: 8255
order: 0
match_id: 0
is_pick: 0
hero_id: 0
hero_name: 0
team: 0
```
### heroes_stats
- path: `data\patch_60\heroes_stats.parquet`
- shape: `(118, 8)`

Columns:
```text
hero_id
hero_name
matches_played
wins
winrate
pick_rate
ban_rate
pick_or_ban_rate
```

Top missing values:
```text
hero_id: 0
hero_name: 0
matches_played: 0
wins: 0
winrate: 0
pick_rate: 0
ban_rate: 0
pick_or_ban_rate: 0
```
### draft_events
- path: `data\patch_60\ml\draft_events.parquet`
- shape: `(8232, 24)`

Columns:
```text
match_id
order
is_pick
hero_id
hero_name
team
total_time_taken
start_time
patch
league_name
radiant_win
radiant_team_id
radiant_team_name
dire_team_id
dire_team_name
acting_side
opponent_side
acting_team_id
acting_team_name
opponent_team_id
opponent_team_name
acting_team_win
action_type
state_id
```

Top missing values:
```text
total_time_taken: 8232
dire_team_name: 120
dire_team_id: 120
acting_team_name: 108
acting_team_id: 108
opponent_team_name: 108
opponent_team_id: 108
radiant_team_id: 96
radiant_team_name: 96
start_time: 0
hero_name: 0
team: 0
order: 0
is_pick: 0
match_id: 0
```
### draft_states
- path: `data\patch_60\ml\draft_states.parquet`
- shape: `(8232, 27)`

Columns:
```text
state_id
match_id
order
draft_phase
action_type
is_pick
chosen_hero_id
chosen_hero_name
acting_side
acting_team_id
acting_team_name
opponent_team_id
opponent_team_name
patch
league_name
start_time
acting_team_win
ally_picks_before
enemy_picks_before
ally_bans_before
enemy_bans_before
n_ally_picks_before
n_enemy_picks_before
n_ally_bans_before
n_enemy_bans_before
available_heroes
available_hero_count
```

Top missing values:
```text
acting_team_id: 108
acting_team_name: 108
opponent_team_id: 108
opponent_team_name: 108
match_id: 0
order: 0
action_type: 0
chosen_hero_id: 0
is_pick: 0
draft_phase: 0
state_id: 0
chosen_hero_name: 0
acting_side: 0
patch: 0
league_name: 0
```
### draft_candidates_pick
- path: `data\patch_60\ml\draft_candidates_pick.parquet`
- shape: `(385189, 24)`

Columns:
```text
state_id
match_id
order
draft_phase
action_type
acting_side
acting_team_id
opponent_team_id
patch
league_name
start_time
n_ally_picks_before
n_enemy_picks_before
n_ally_bans_before
n_enemy_bans_before
available_hero_count
candidate_hero_id
target
candidate_hero_name
candidate_matches_played
candidate_winrate
candidate_pick_rate
candidate_ban_rate
candidate_pick_or_ban_rate
```

Top missing values:
```text
candidate_hero_name: 30864
acting_team_id: 5055
opponent_team_id: 5052
state_id: 0
draft_phase: 0
order: 0
acting_side: 0
match_id: 0
action_type: 0
patch: 0
start_time: 0
league_name: 0
n_enemy_picks_before: 0
n_ally_bans_before: 0
n_enemy_bans_before: 0
```
### draft_candidates_ban
- path: `data\patch_60\ml\draft_candidates_ban.parquet`
- shape: `(565607, 24)`

Columns:
```text
state_id
match_id
order
draft_phase
action_type
acting_side
acting_team_id
opponent_team_id
patch
league_name
start_time
n_ally_picks_before
n_enemy_picks_before
n_ally_bans_before
n_enemy_bans_before
available_hero_count
candidate_hero_id
target
candidate_hero_name
candidate_matches_played
candidate_winrate
candidate_pick_rate
candidate_ban_rate
candidate_pick_or_ban_rate
```

Top missing values:
```text
candidate_hero_name: 43215
acting_team_id: 7428
opponent_team_id: 7413
state_id: 0
draft_phase: 0
order: 0
acting_side: 0
match_id: 0
action_type: 0
patch: 0
start_time: 0
league_name: 0
n_enemy_picks_before: 0
n_ally_bans_before: 0
n_enemy_bans_before: 0
```
## Metrics
### Pick metrics
```json
{
  "states": 520,
  "top1": 0.11923076923076924,
  "top3": 0.26346153846153847,
  "top5": 0.34615384615384615,
  "top10": 0.49038461538461536,
  "mean_rank": 18.26153846153846,
  "mrr": 0.24150220043390594
}
```
### Ban metrics
```json
{
  "states": 728,
  "top1": 0.1043956043956044,
  "top3": 0.2554945054945055,
  "top5": 0.3228021978021978,
  "top10": 0.5412087912087912,
  "mean_rank": 15.081043956043956,
  "mrr": 0.23433798625190072
}
```
## Features
### Pick features
```json
{
  "features": [
    "order",
    "draft_phase",
    "acting_side",
    "acting_team_id",
    "opponent_team_id",
    "patch",
    "league_name",
    "n_ally_picks_before",
    "n_enemy_picks_before",
    "n_ally_bans_before",
    "n_enemy_bans_before",
    "available_hero_count",
    "candidate_hero_id",
    "candidate_matches_played",
    "candidate_winrate",
    "candidate_pick_rate",
    "candidate_ban_rate",
    "candidate_pick_or_ban_rate"
  ],
  "cat_features": [
    "acting_side",
    "acting_team_id",
    "opponent_team_id",
    "patch",
    "league_name",
    "candidate_hero_id"
  ]
}
```
### Ban features
```json
{
  "features": [
    "order",
    "draft_phase",
    "acting_side",
    "acting_team_id",
    "opponent_team_id",
    "patch",
    "league_name",
    "n_ally_picks_before",
    "n_enemy_picks_before",
    "n_ally_bans_before",
    "n_enemy_bans_before",
    "available_hero_count",
    "candidate_hero_id",
    "candidate_matches_played",
    "candidate_winrate",
    "candidate_pick_rate",
    "candidate_ban_rate",
    "candidate_pick_or_ban_rate"
  ],
  "cat_features": [
    "acting_side",
    "acting_team_id",
    "opponent_team_id",
    "patch",
    "league_name",
    "candidate_hero_id"
  ]
}
```
## Model files
```text
models\patch_60\ban_features.json | 0.62 KB
models\patch_60\ban_metrics.json | 0.21 KB
models\patch_60\ban_model.cbm | 1219.74 KB
models\patch_60\notebook_report.md | 6.43 KB
models\patch_60\pick_features.json | 0.62 KB
models\patch_60\pick_metrics.json | 0.21 KB
models\patch_60\pick_model.cbm | 1269.32 KB
```
