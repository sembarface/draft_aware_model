import argparse
import json

import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_paths
from src.ml.train_catboost import DATASET_CHOICES, output_stem


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate CatBoostRanker draft ranking model.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--action", choices=["pick", "ban"], default="pick", help="Draft action to evaluate.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="base", help="Candidate dataset variant.")
    return parser.parse_args(argv)


def ranking_metrics(df):
    ranks = []
    for _, group in df.groupby("state_id"):
        group = group.sort_values("pred", ascending=False).reset_index(drop=True)
        pos = group.index[group["target"] == 1]
        if len(pos) == 0:
            continue
        ranks.append(int(pos[0]) + 1)

    ranks = pd.Series(ranks, dtype="float64")
    if ranks.empty:
        return {"states": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "mean_rank": 0.0, "mrr": 0.0}
    return {
        "states": int(len(ranks)),
        "top1": float((ranks <= 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1 / ranks).mean()),
    }


def evaluate(action="pick", dataset="base", patch_label=PATCH_LABEL):
    _, _, _, ml_dir, model_dir = get_patch_paths(patch_label)
    report_dirs = get_ml_report_dirs(patch_label)
    stem = output_stem(action, dataset)
    features_path = report_dirs["features"] / f"{stem}_features.json"
    if not features_path.exists():
        old_features_path = model_dir / f"{stem}_features.json"
        if old_features_path.exists():
            features_path = old_features_path
    model_path = model_dir / f"{stem}_model.cbm"
    test_path = ml_dir / f"{stem}_test.parquet"

    with open(features_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "ranker":
        raise ValueError(f"{features_path} is not a ranker feature config")

    features = cfg["features"]
    cat_features = cfg["cat_features"]
    test = pd.read_parquet(test_path)

    for col in cat_features:
        test[col] = test[col].fillna("unknown").astype(str)
    numeric_features = [col for col in features if col not in cat_features]
    for col in numeric_features:
        test[col] = test[col].fillna(0)

    model = CatBoostRanker()
    model.load_model(model_path)
    pool = Pool(test[features], cat_features=cat_features)
    test["pred"] = model.predict(pool)
    try:
        importance = model.get_feature_importance(pool, type="PredictionValuesChange")
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        importance_path = report_dirs["importance"] / f"{stem}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"saved: {importance_path}")
    except Exception as exc:
        print(f"warning: could not save feature importance for {stem}: {exc}")

    metrics = ranking_metrics(test)
    predictions_path = ml_dir / f"predictions_{stem}_test.parquet"
    metrics_path = report_dirs["metrics"] / f"{stem}_metrics.json"
    test.to_parquet(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(metrics)
    print(f"saved: {predictions_path}")
    print(f"saved: {metrics_path}")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    return evaluate(args.action, args.dataset, args.patch_label)


if __name__ == "__main__":
    main()
