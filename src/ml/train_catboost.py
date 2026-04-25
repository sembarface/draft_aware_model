import argparse
import json

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import PATCH_LABEL, get_patch_paths
from src.ml.split_data import split_by_time


BASE_FEATURES = [
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

CAT_FEATURES = [
    "acting_side",
    "acting_team_id",
    "opponent_team_id",
    "patch",
    "league_name",
    "candidate_hero_id",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CatBoost model for draft candidates.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--action", choices=["pick", "ban"], default="pick", help="Draft action to train.")
    parser.add_argument(
        "--dataset",
        choices=["base", "interactions"],
        default="base",
        help="Candidate dataset variant.",
    )
    return parser.parse_args(argv)


def dataset_path(ml_dir, action, dataset):
    suffix = "" if dataset == "base" else "_interactions"
    return ml_dir / f"draft_candidates_{action}{suffix}.parquet"


def output_stem(action, dataset):
    return f"{action}_{dataset}"


def select_features(df, dataset):
    features = BASE_FEATURES.copy()
    if dataset == "interactions":
        features.extend([col for col in INTERACTION_FEATURES if col in df.columns])
    return features


def train_model(action="pick", dataset="base", patch_label=PATCH_LABEL):
    _, _, _, ml_dir, model_dir = get_patch_paths(patch_label)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = dataset_path(ml_dir, action, dataset)
    df = pd.read_parquet(path)

    train, valid, test = split_by_time(df)
    print("train:", train.shape)
    print("valid:", valid.shape)
    print("test:", test.shape)

    print("train states:", train["state_id"].nunique())
    print("valid states:", valid["state_id"].nunique())
    print("test states:", test["state_id"].nunique())

    print("train matches:", train["match_id"].nunique())
    print("valid matches:", valid["match_id"].nunique())
    print("test matches:", test["match_id"].nunique())

    features = select_features(df, dataset)
    cat_features = [col for col in CAT_FEATURES if col in features]

    for col in cat_features:
        train[col] = train[col].fillna("unknown").astype(str)
        valid[col] = valid[col].fillna("unknown").astype(str)
        test[col] = test[col].fillna("unknown").astype(str)

    numeric_features = [col for col in features if col not in cat_features]
    for col in numeric_features:
        train[col] = train[col].fillna(0)
        valid[col] = valid[col].fillna(0)
        test[col] = test[col].fillna(0)

    X_train = train[features]
    y_train = train["target"]
    X_valid = valid[features]
    y_valid = valid["target"]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    stem = output_stem(action, dataset)
    model.save_model(model_dir / f"{stem}_model.cbm")

    with open(model_dir / f"{stem}_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "patch_label": patch_label,
                "action": action,
                "dataset": dataset,
                "features": features,
                "cat_features": cat_features,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # To switch to interaction features, build
    # draft_candidates_{action}_interactions.parquet and run --dataset interactions.
    test.to_parquet(ml_dir / f"{stem}_test.parquet", index=False)


def main(argv=None):
    args = parse_args(argv)
    return train_model(args.action, args.dataset, args.patch_label)


if __name__ == "__main__":
    main()
