import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_paths
from src.ml.role_utils import POSITION_COLS, load_hero_roles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze ranking metrics by draft context.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="Patch label, e.g. 7.41.")
    parser.add_argument("--dataset", default="players_team_role", help="Dataset to analyze.")
    return parser.parse_args(argv)


def _to_markdown(df):
    if df.empty:
        return "No data.\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def _target_rows(predictions):
    ranks = []
    for _, group in predictions.groupby("state_id"):
        group = group.sort_values("pred", ascending=False).reset_index(drop=True)
        pos = group.index[group["target"] == 1]
        if len(pos) == 0:
            continue
        row = group.loc[pos[0]].copy()
        row["rank"] = int(pos[0]) + 1
        ranks.append(row)
    return pd.DataFrame(ranks)


def _action_bucket(action_type, order):
    try:
        order = int(order)
    except (TypeError, ValueError):
        return "unknown"
    if action_type == "pick":
        if order in {8, 9}:
            return "pick_opening"
        if order in {13, 14, 15, 16, 17, 18}:
            return "pick_mid"
        if order in {23, 24}:
            return "pick_late"
    if action_type == "ban":
        if 1 <= order <= 7:
            return "ban_phase1"
        if 10 <= order <= 12:
            return "ban_phase2"
        if 19 <= order <= 22:
            return "ban_phase3"
    return "unknown"


def _rank_metrics_from_targets(targets):
    ranks = targets["rank"].astype(float)
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


def _group_metrics(targets, cols, action):
    rows = []
    for key, group in targets.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {"action": action}
        row.update(dict(zip(cols, key)))
        row.update(_rank_metrics_from_targets(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _role_lookup():
    roles = load_hero_roles()
    roles["dominant_role"] = roles[POSITION_COLS].idxmax(axis=1).str.replace("_prob", "", regex=False)
    return roles.set_index("hero_id")["dominant_role"].to_dict()


def analyze_metrics_by_context(patch_label=PATCH_LABEL, dataset="players_team_role"):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)
    report_dirs = get_ml_report_dirs(patch_label)
    role_by_hero = _role_lookup()
    suffix = "" if dataset == "players_team_role" else f"_{dataset}"
    all_rows = []
    split_outputs = {
        "order": [],
        "phase": [],
        "order_bucket": [],
        "role": [],
        "team": [],
    }

    for action in ["pick", "ban"]:
        path = ml_dir / f"predictions_{action}_{dataset}_test.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing predictions: {path}")
        predictions = pd.read_parquet(path)
        predictions["order_bucket"] = predictions.apply(
            lambda row: _action_bucket(row.get("action_type"), row.get("order")),
            axis=1,
        )
        targets = _target_rows(predictions)
        hero_ids = pd.to_numeric(targets["candidate_hero_id"], errors="coerce").astype("Int64")
        targets["dominant_role"] = hero_ids.map(role_by_hero).fillna("unknown")

        groups = {
            "order": ["order"],
            "phase": ["draft_phase"],
            "order_bucket": ["order_bucket"],
            "role": ["dominant_role"],
            "team": ["acting_team_id"],
            "opponent_team": ["opponent_team_id"],
        }
        for name, cols in groups.items():
            df = _group_metrics(targets, cols, action)
            df.insert(1, "context", name)
            all_rows.append(df)
            if name in split_outputs:
                split_outputs[name].append(df)
            elif name == "opponent_team":
                split_outputs["team"].append(df)

    context_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    csv_path = report_dirs["errors"] / f"metrics_by_context{suffix}.csv"
    context_df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")

    output_map = {
        "order": f"metrics_by_order{suffix}.md",
        "phase": f"metrics_by_phase{suffix}.md",
        "order_bucket": f"metrics_by_order_bucket{suffix}.md",
        "role": f"metrics_by_role{suffix}.md",
        "team": f"metrics_by_team{suffix}.md",
    }
    for name, frames in split_outputs.items():
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = report_dirs["errors"] / output_map[name]
        path.write_text(f"# Metrics by {name}\n\n" + _to_markdown(df), encoding="utf-8")
        print(f"saved: {path}")
    return context_df


def main(argv=None):
    args = parse_args(argv)
    return analyze_metrics_by_context(args.patch_label, args.dataset)


if __name__ == "__main__":
    main()
