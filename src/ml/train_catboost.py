import argparse
import json

import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_paths
from src.ml.feature_sets import CAT_FEATURES, DATASET_CHOICES, dataset_path, output_stem, select_features
from src.ml.split_data import split_by_time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CatBoostRanker for draft candidates.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--action", choices=["pick", "ban"], default="pick", help="Draft action to train.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="base", help="Candidate dataset variant.")
    parser.add_argument("--loss-function", choices=["YetiRank", "PairLogit"], default="YetiRank", help="Ranking loss.")
    return parser.parse_args(argv)


def feature_config_path(patch_label, action, dataset):
    report_dirs = get_ml_report_dirs(patch_label)
    return report_dirs["features"] / f"{output_stem(action, dataset)}_features.json"


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


def _validate_candidate_table(df, table_name):
    required = {"state_id", "match_id", "start_time", "target", "candidate_hero_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")
    target_sum = df.groupby("state_id")["target"].sum()
    bad = target_sum[target_sum != 1]
    if not bad.empty:
        raise ValueError(f"{table_name}: every state_id must have exactly one target=1. Examples: {bad.head().to_dict()}")


def _validate_split(train, valid, test):
    split_ids = {
        "train": set(train["match_id"].unique()),
        "valid": set(valid["match_id"].unique()),
        "test": set(test["match_id"].unique()),
    }
    overlaps = {
        "train_valid": split_ids["train"] & split_ids["valid"],
        "train_test": split_ids["train"] & split_ids["test"],
        "valid_test": split_ids["valid"] & split_ids["test"],
    }
    bad = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
    if bad:
        raise ValueError(f"match_id leakage across splits: {bad}")


def train_model(action="pick", dataset="base", patch_label=PATCH_LABEL, loss_function="YetiRank"):
    _, _, _, ml_dir, model_dir = get_patch_paths(patch_label)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = dataset_path(ml_dir, action, dataset)
    df = pd.read_parquet(path)
    _validate_candidate_table(df, path.name)

    train, valid, test = split_by_time(df)
    _validate_split(train, valid, test)
    train = _sort_for_grouping(train)
    valid = _sort_for_grouping(valid)
    test = _sort_for_grouping(test)

    print("train:", train.shape)
    print("valid:", valid.shape)
    print("test:", test.shape)

    features = select_features(df, dataset)
    if not features:
        raise ValueError(f"No training features selected for dataset={dataset}")
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
