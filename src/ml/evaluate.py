from pathlib import Path
import json
import pandas as pd

from catboost import CatBoostClassifier, Pool


PATCH = 60
ACTION = "pick"

BASE = Path(f"data/patch_{PATCH}")
ML_DIR = BASE / "ml"
MODEL_DIR = Path(f"models/patch_{PATCH}")


def ranking_metrics(df):
    ranks = []

    for _, g in df.groupby("state_id"):
        g = g.sort_values("pred", ascending=False).reset_index(drop=True)
        pos = g.index[g["target"] == 1]

        if len(pos) == 0:
            continue

        rank = int(pos[0]) + 1
        ranks.append(rank)

    ranks = pd.Series(ranks)

    return {
        "states": int(len(ranks)),
        "top1": float((ranks <= 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1 / ranks).mean()),
    }


def evaluate(action):
    with open(MODEL_DIR / f"{action}_features.json", encoding="utf-8") as f:
        cfg = json.load(f)

    features = cfg["features"]
    cat_features = cfg["cat_features"]

    test = pd.read_parquet(ML_DIR / f"{action}_test.parquet")

    for col in cat_features:
        test[col] = test[col].astype(str)

    model = CatBoostClassifier()
    model.load_model(MODEL_DIR / f"{action}_model.cbm")

    pool = Pool(test[features], cat_features=cat_features)
    test["pred"] = model.predict_proba(pool)[:, 1]

    metrics = ranking_metrics(test)

    test.to_parquet(ML_DIR / f"predictions_{action}_test.parquet", index=False)

    with open(MODEL_DIR / f"{action}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(metrics)


if __name__ == "__main__":
    evaluate(ACTION)