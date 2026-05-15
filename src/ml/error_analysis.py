import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_paths


OLD_DATASET = "players_team"
NEW_DATASET = "players_team_role"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare old/new model ranks on saved prediction parquet files.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--old-dataset", default=OLD_DATASET, help="Baseline dataset name.")
    parser.add_argument("--new-dataset", default=NEW_DATASET, help="New dataset name.")
    parser.add_argument("--left-dataset", default=None, help="Alias for --old-dataset.")
    parser.add_argument("--right-dataset", default=None, help="Alias for --new-dataset.")
    return parser.parse_args(argv)


def _prediction_path(ml_dir, action, dataset):
    return ml_dir / f"predictions_{action}_{dataset}_test.parquet"


def _target_ranks(path, rank_col):
    df = pd.read_parquet(path)
    required = {"state_id", "target", "pred", "candidate_hero_id", "candidate_hero_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.sort_values(["state_id", "pred"], ascending=[True, False]).copy()
    df[rank_col] = df.groupby("state_id").cumcount() + 1
    target = df[df["target"] == 1].copy()
    target["candidate_hero_id"] = pd.to_numeric(target["candidate_hero_id"], errors="coerce").astype("Int64")

    keep = [
        "state_id",
        "candidate_hero_id",
        "candidate_hero_name",
        rank_col,
    ]
    for col in ["match_id", "order", "draft_phase", "action_type", "acting_side", "acting_team_id", "opponent_team_id"]:
        if col in target.columns:
            keep.insert(-3, col)
    target = target[keep].rename(
        columns={
            "candidate_hero_id": "true_hero_id",
            "candidate_hero_name": "true_hero_name",
        }
    )
    return target


def compare_action(ml_dir, report_dirs, action, old_dataset=OLD_DATASET, new_dataset=NEW_DATASET):
    old_path = _prediction_path(ml_dir, action, old_dataset)
    new_path = _prediction_path(ml_dir, action, new_dataset)
    if not old_path.exists():
        raise FileNotFoundError(f"missing old predictions: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"missing new predictions: {new_path}")

    old_ranks = _target_ranks(old_path, "old_rank")
    new_ranks = _target_ranks(new_path, "new_rank")

    keys = ["state_id", "true_hero_id", "true_hero_name"]
    meta_cols = [col for col in old_ranks.columns if col not in keys + ["old_rank"]]
    merged = old_ranks.merge(new_ranks[keys + ["new_rank"]], on=keys, how="inner")
    merged = merged[[*meta_cols, *keys, "old_rank", "new_rank"]]
    merged["rank_delta"] = merged["old_rank"] - merged["new_rank"]
    merged["old_top10"] = merged["old_rank"] <= 10
    merged["new_top10"] = merged["new_rank"] <= 10
    merged["old_top5"] = merged["old_rank"] <= 5
    merged["new_top5"] = merged["new_rank"] <= 5
    merged["old_top3"] = merged["old_rank"] <= 3
    merged["new_top3"] = merged["new_rank"] <= 3
    merged["improved_into_top10"] = (merged["old_rank"] > 10) & (merged["new_rank"] <= 10)
    merged["worsened_out_of_top10"] = (merged["old_rank"] <= 10) & (merged["new_rank"] > 10)
    merged["improved_into_top5"] = (merged["old_rank"] > 5) & (merged["new_rank"] <= 5)
    merged["worsened_out_of_top5"] = (merged["old_rank"] <= 5) & (merged["new_rank"] > 5)
    merged["improved_into_top3"] = (merged["old_rank"] > 3) & (merged["new_rank"] <= 3)
    merged["worsened_out_of_top3"] = (merged["old_rank"] <= 3) & (merged["new_rank"] > 3)

    output_path = report_dirs["errors"] / f"{action}_{old_dataset}_vs_{new_dataset}.csv"
    merged.to_csv(output_path, index=False)
    print(f"saved: {output_path}")
    return merged, output_path


def _summary_row(action, df):
    return {
        "action": action,
        "states": int(len(df)),
        "mean_old_rank": float(df["old_rank"].mean()) if not df.empty else 0.0,
        "mean_new_rank": float(df["new_rank"].mean()) if not df.empty else 0.0,
        "mean_rank_delta": float(df["rank_delta"].mean()) if not df.empty else 0.0,
        "improved_into_top10": int(df["improved_into_top10"].sum()) if not df.empty else 0,
        "worsened_out_of_top10": int(df["worsened_out_of_top10"].sum()) if not df.empty else 0,
        "improved_into_top5": int(df["improved_into_top5"].sum()) if not df.empty else 0,
        "worsened_out_of_top5": int(df["worsened_out_of_top5"].sum()) if not df.empty else 0,
        "improved_into_top3": int(df["improved_into_top3"].sum()) if not df.empty else 0,
        "worsened_out_of_top3": int(df["worsened_out_of_top3"].sum()) if not df.empty else 0,
    }


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(df):
    if df.empty:
        return "No data.\n"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def _compact_examples(df, ascending):
    cols = [
        col
        for col in [
            "state_id",
            "match_id",
            "order",
            "draft_phase",
            "acting_side",
            "true_hero_id",
            "true_hero_name",
            "old_rank",
            "new_rank",
            "rank_delta",
        ]
        if col in df.columns
    ]
    return df.sort_values("rank_delta", ascending=ascending)[cols].head(20)


def write_summary(report_dirs, results, old_dataset=OLD_DATASET, new_dataset=NEW_DATASET):
    rows = [_summary_row(action, df) for action, df in results.items()]
    summary = pd.DataFrame(rows)
    lines = [f"# Model error analysis: {old_dataset} vs {new_dataset}\n\n"]
    lines.append("## Summary\n\n")
    lines.append(_to_markdown(summary))

    for action, df in results.items():
        lines.append(f"\n## {action}: top-20 biggest improvements\n\n")
        lines.append(_to_markdown(_compact_examples(df, ascending=False)))
        lines.append(f"\n## {action}: top-20 biggest regressions\n\n")
        lines.append(_to_markdown(_compact_examples(df, ascending=True)))

    named_path = report_dirs["errors"] / f"model_error_analysis_{old_dataset}_vs_{new_dataset}.md"
    named_path.write_text("".join(lines), encoding="utf-8")
    print(f"saved: {named_path}")
    default_path = report_dirs["errors"] / "model_error_analysis.md"
    default_path.write_text("".join(lines), encoding="utf-8")
    print(f"saved: {default_path}")
    return named_path


def run_error_analysis(patch_label=PATCH_LABEL, old_dataset=OLD_DATASET, new_dataset=NEW_DATASET):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)
    report_dirs = get_ml_report_dirs(patch_label)
    results = {}
    for action in ["pick", "ban"]:
        results[action], _ = compare_action(ml_dir, report_dirs, action, old_dataset, new_dataset)
    write_summary(report_dirs, results, old_dataset, new_dataset)
    return results


def main(argv=None):
    args = parse_args(argv)
    old_dataset = args.left_dataset or args.old_dataset
    new_dataset = args.right_dataset or args.new_dataset
    return run_error_analysis(args.patch_label, old_dataset, new_dataset)


if __name__ == "__main__":
    main()
