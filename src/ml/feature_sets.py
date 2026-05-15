DATASET_CHOICES = [
    "base",
    "interactions",
    "interactions_role",
    "players_team",
    "players_team_role",
]

BASE_FEATURES = [
    "order",
    "draft_phase",
    "action_type",
    "acting_side",
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

TEAM_ID_FEATURES = [
    "acting_team_id",
    "opponent_team_id",
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

TEAM_PRIORITY_FEATURES = [
    "own_team_candidate_matches_alltime",
    "own_team_candidate_pick_count_alltime",
    "own_team_candidate_pick_rate_alltime",
    "own_team_candidate_winrate_alltime",
    "own_team_candidate_early_pick_rate_alltime",
    "own_team_candidate_avg_pick_order_alltime",
    "own_team_candidate_ban_against_count_alltime",
    "own_team_candidate_ban_against_rate_alltime",
    "own_team_candidate_contested_count_alltime",
    "own_team_candidate_contested_rate_alltime",
    "own_team_candidate_matches_patch",
    "own_team_candidate_pick_count_patch",
    "own_team_candidate_pick_rate_patch",
    "own_team_candidate_winrate_patch",
    "own_team_candidate_early_pick_rate_patch",
    "own_team_candidate_avg_pick_order_patch",
    "own_team_candidate_ban_against_count_patch",
    "own_team_candidate_ban_against_rate_patch",
    "own_team_candidate_contested_count_patch",
    "own_team_candidate_contested_rate_patch",
    "opponent_team_candidate_matches_alltime",
    "opponent_team_candidate_pick_count_alltime",
    "opponent_team_candidate_pick_rate_alltime",
    "opponent_team_candidate_winrate_alltime",
    "opponent_team_candidate_early_pick_rate_alltime",
    "opponent_team_candidate_avg_pick_order_alltime",
    "opponent_team_candidate_ban_against_count_alltime",
    "opponent_team_candidate_ban_against_rate_alltime",
    "opponent_team_candidate_contested_count_alltime",
    "opponent_team_candidate_contested_rate_alltime",
    "opponent_team_candidate_matches_patch",
    "opponent_team_candidate_pick_count_patch",
    "opponent_team_candidate_pick_rate_patch",
    "opponent_team_candidate_winrate_patch",
    "opponent_team_candidate_early_pick_rate_patch",
    "opponent_team_candidate_avg_pick_order_patch",
    "opponent_team_candidate_ban_against_count_patch",
    "opponent_team_candidate_ban_against_rate_patch",
    "opponent_team_candidate_contested_count_patch",
    "opponent_team_candidate_contested_rate_patch",
]

ROLE_FEATURES = [
    "candidate_pos1_prob",
    "candidate_pos2_prob",
    "candidate_pos3_prob",
    "candidate_pos4_prob",
    "candidate_pos5_prob",
    "candidate_core_prob",
    "candidate_support_prob",
    "candidate_flex_score",
    "ally_pos1_filled",
    "ally_pos2_filled",
    "ally_pos3_filled",
    "ally_pos4_filled",
    "ally_pos5_filled",
    "ally_core_count_soft",
    "ally_support_count_soft",
    "enemy_pos1_filled",
    "enemy_pos2_filled",
    "enemy_pos3_filled",
    "enemy_pos4_filled",
    "enemy_pos5_filled",
    "enemy_core_count_soft",
    "enemy_support_count_soft",
    "candidate_own_role_fit_score",
    "candidate_own_role_conflict_score",
    "candidate_own_core_fit",
    "candidate_own_support_fit",
    "candidate_enemy_role_fit_score",
    "candidate_enemy_role_conflict_score",
    "candidate_enemy_core_fit",
    "candidate_enemy_support_fit",
    "own_best_player_candidate_role_fit",
    "own_mean_player_candidate_role_fit",
    "opponent_best_player_candidate_role_fit",
    "opponent_mean_player_candidate_role_fit",
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


def canonical_feature_dataset(dataset):
    return dataset


def dataset_path(ml_dir, action, dataset):
    dataset = canonical_feature_dataset(dataset)
    suffix = "" if dataset == "base" else f"_{dataset}"
    return ml_dir / f"draft_candidates_{action}{suffix}.parquet"


def select_features(df, dataset, action=None):
    dataset = canonical_feature_dataset(dataset)
    features = BASE_FEATURES.copy()
    role_datasets = {"interactions_role", "players_team_role"}
    if dataset in {"interactions", "players_team", *role_datasets}:
        features.extend(INTERACTION_FEATURES)
    if dataset in {"players_team", "players_team_role"}:
        features.extend(PLAYER_FEATURES)
    if dataset in {"players_team", "players_team_role"}:
        features.extend(TEAM_ID_FEATURES)
        features.extend(TEAM_PRIORITY_FEATURES)
    if dataset in role_datasets:
        features.extend(ROLE_FEATURES)
    return [col for col in features if col in df.columns]
