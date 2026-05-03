DATASET_CHOICES = ["base", "interactions", "players"]

BASE_FEATURES = [
    "order",
    "draft_phase",
    "action_type",
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
    "candidate_pick_or_ban_rate",
]

INTERACTION_FEATURES = [
    "candidate_ally_synergy_mean",
    "candidate_ally_synergy_max",
    "candidate_ally_synergy_min",
    "candidate_ally_synergy_games_mean",
    "candidate_vs_enemy_counter_mean",
    "candidate_vs_enemy_counter_max",
    "candidate_vs_enemy_counter_min",
    "candidate_vs_enemy_matchup_games_mean",
    "enemy_vs_candidate_counter_mean",
    "candidate_enemy_synergy_mean",
    "candidate_enemy_synergy_max",
    "candidate_enemy_synergy_min",
    "candidate_enemy_synergy_games_mean",
    "candidate_vs_ally_counter_mean",
    "candidate_vs_ally_counter_max",
    "candidate_vs_ally_counter_min",
    "candidate_vs_ally_matchup_games_mean",
    "ally_vs_candidate_counter_mean",
]

PLAYER_FEATURES = [
    "own_player_hero_games_alltime_max",
    "own_player_hero_games_alltime_mean",
    "own_player_hero_winrate_alltime_max",
    "own_player_hero_winrate_alltime_mean",
    "own_player_hero_avg_kda_alltime_max",
    "own_player_hero_avg_gold_per_min_alltime_max",
    "own_player_hero_avg_xp_per_min_alltime_max",
    "own_player_hero_avg_hero_damage_alltime_max",
    "own_player_hero_avg_tower_damage_alltime_max",
    "opponent_player_hero_games_alltime_max",
    "opponent_player_hero_games_alltime_mean",
    "opponent_player_hero_winrate_alltime_max",
    "opponent_player_hero_winrate_alltime_mean",
    "opponent_player_hero_avg_kda_alltime_max",
    "opponent_player_hero_avg_gold_per_min_alltime_max",
    "opponent_player_hero_avg_xp_per_min_alltime_max",
    "opponent_player_hero_avg_hero_damage_alltime_max",
    "opponent_player_hero_avg_tower_damage_alltime_max",
    "own_player_hero_games_patch_max",
    "own_player_hero_winrate_patch_max",
    "own_player_hero_avg_kda_patch_max",
    "own_player_hero_avg_gold_per_min_patch_max",
    "own_player_hero_avg_xp_per_min_patch_max",
    "opponent_player_hero_games_patch_max",
    "opponent_player_hero_winrate_patch_max",
    "opponent_player_hero_avg_kda_patch_max",
    "opponent_player_hero_avg_gold_per_min_patch_max",
    "opponent_player_hero_avg_xp_per_min_patch_max",
    "own_roster_player_matches_alltime_mean",
    "own_roster_player_winrate_alltime_mean",
    "own_roster_player_avg_kda_alltime_mean",
    "own_roster_player_avg_gold_per_min_alltime_mean",
    "own_roster_player_avg_xp_per_min_alltime_mean",
    "opponent_roster_player_matches_alltime_mean",
    "opponent_roster_player_winrate_alltime_mean",
    "opponent_roster_player_avg_kda_alltime_mean",
    "opponent_roster_player_avg_gold_per_min_alltime_mean",
    "opponent_roster_player_avg_xp_per_min_alltime_mean",
    "own_roster_player_matches_patch_mean",
    "own_roster_player_winrate_patch_mean",
    "own_roster_player_avg_kda_patch_mean",
    "opponent_roster_player_matches_patch_mean",
    "opponent_roster_player_winrate_patch_mean",
    "opponent_roster_player_avg_kda_patch_mean",
]

CAT_FEATURES = [
    "action_type",
    "acting_side",
    "acting_team_id",
    "opponent_team_id",
    "patch",
    "league_name",
    "candidate_hero_id",
]


def output_stem(action, dataset):
    return f"{action}_{dataset}"


def dataset_path(ml_dir, action, dataset):
    suffix = "" if dataset == "base" else f"_{dataset}"
    return ml_dir / f"draft_candidates_{action}{suffix}.parquet"


def select_features(df, dataset):
    features = BASE_FEATURES.copy()
    if dataset in {"interactions", "players"}:
        features.extend(INTERACTION_FEATURES)
    if dataset == "players":
        features.extend(PLAYER_FEATURES)
    return [col for col in features if col in df.columns]
