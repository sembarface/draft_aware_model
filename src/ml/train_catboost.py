from pathlib import Path
import json
import pandas as pd

from catboost import CatBoostClassifier, Pool
from split_data import split_by_time


PATCH = 60
ACTION = "pick"

BASE = Path(f"data/patch_{PATCH}")
ML_DIR = BASE / "ml"
MODEL_DIR = Path(f"models/patch_{PATCH}")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model(action):
    path = ML_DIR / f"draft_candidates_{action}.parquet"
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

    features = [
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

    cat_features = [
        "acting_side",
        "acting_team_id",
        "opponent_team_id",
        "patch",
        "league_name",
        "candidate_hero_id",
    ]

    for col in cat_features:
        train[col] = train[col].astype(str)
        valid[col] = valid[col].astype(str)
        test[col] = test[col].astype(str)
    
    for col in ["acting_team_id", "opponent_team_id", "league_name"]:
        train[col] = train[col].fillna("unknown").astype(str)
        valid[col] = valid[col].fillna("unknown").astype(str)
        test[col] = test[col].fillna("unknown").astype(str)

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
        verbose=100
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    model.save_model(MODEL_DIR / f"{action}_model.cbm")

    with open(MODEL_DIR / f"{action}_features.json", "w", encoding="utf-8") as f:
        json.dump({
            "features": features,
            "cat_features": cat_features
        }, f, ensure_ascii=False, indent=2)

    test.to_parquet(ML_DIR / f"{action}_test.parquet", index=False)


if __name__ == "__main__":
    train_model(ACTION)