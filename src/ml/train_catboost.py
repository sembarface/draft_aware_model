import argparse
import json

import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_paths
from src.ml.split_data import split_by_time


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CatBoostRanker for draft candidates.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--action", choices=["pick", "ban"], default="pick", help="Draft action to train.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="base", help="Candidate dataset variant.")
    parser.add_argument("--loss-function", choices=["YetiRank", "PairLogit"], default="YetiRank", help="Ranking loss.")
    return parser.parse_args(argv)


def dataset_path(ml_dir, action, dataset):
    suffix = "" if dataset == "base" else f"_{dataset}"
    return ml_dir / f"draft_candidates_{action}{suffix}.parquet"


def output_stem(action, dataset):
    return f"{action}_{dataset}"


def feature_config_path(patch_label, action, dataset):
    report_dirs = get_ml_report_dirs(patch_label)
    return report_dirs["features"] / f"{output_stem(action, dataset)}_features.json"


def select_features(df, dataset):
    features = BASE_FEATURES.copy()
    if dataset in {"interactions", "players"}:
        features.extend(INTERACTION_FEATURES)
    if dataset == "players":
        features.extend(PLAYER_FEATURES)
    return [col for col in features if col in df.columns]


def _prepare(train, valid, test, features, cat_features):
    for col in cat_features:
        train[col] = train[col].fillna("unknown").astype(str)
        valid[col] = valid[col].fillna("unknown").astype(str)
        test[col] = test[col].fillna("unknown").astype(str)
    numeric_features = [col for col in features if col not in cat_features]
    for col in numeric_features:
        train[col] = train[col].fillna(0)
        valid[col] = valid[col].fillna(0)
        test[col] = test[col].fillna(0)
    return train, valid, test


def _sort_for_grouping(df):
    return df.sort_values(["state_id", "candidate_hero_id"]).reset_index(drop=True)


def train_model(action="pick", dataset="base", patch_label=PATCH_LABEL, loss_function="YetiRank"):
    _, _, _, ml_dir, model_dir = get_patch_paths(patch_label)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = dataset_path(ml_dir, action, dataset)
    df = pd.read_parquet(path)

    train, valid, test = split_by_time(df)
    train = _sort_for_grouping(train)
    valid = _sort_for_grouping(valid)
    test = _sort_for_grouping(test)

    print("train:", train.shape)
    print("valid:", valid.shape)
    print("test:", test.shape)

    features = select_features(df, dataset)
    cat_features = [col for col in CAT_FEATURES if col in features]
    train, valid, test = _prepare(train, valid, test, features, cat_features)

    train_pool = Pool(
        train[features],
        train["target"],
        cat_features=cat_features,
        group_id=train["state_id"].astype(str),
    )
    valid_pool = Pool(
        valid[features],
        valid["target"],
        cat_features=cat_features,
        group_id=valid["state_id"].astype(str),
    )

    model = CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function=loss_function,
        eval_metric="NDCG",
        random_seed=42,
        verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    stem = output_stem(action, dataset)
    model.save_model(model_dir / f"{stem}_model.cbm")
    features_path = feature_config_path(patch_label, action, dataset)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "patch_label": patch_label,
                "action": action,
                "dataset": dataset,
                "model_type": "ranker",
                "cutoff_rule": "source_start_time < current_match_start_time",
                "features": features,
                "cat_features": cat_features,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    test.to_parquet(ml_dir / f"{stem}_test.parquet", index=False)
    print(f"saved: {model_dir / f'{stem}_model.cbm'}")
    print(f"saved: {features_path}")
    print(f"saved: {ml_dir / f'{stem}_test.parquet'}")
    return model


def main(argv=None):
    args = parse_args(argv)
    return train_model(args.action, args.dataset, args.patch_label, args.loss_function)


if __name__ == "__main__":
    main()
