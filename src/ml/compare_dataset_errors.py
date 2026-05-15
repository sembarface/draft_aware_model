import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare target ranks between two evaluated datasets.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--left", default="players_team", help="Baseline dataset.")
    parser.add_argument("--right", default="players_team_role", help="Candidate dataset.")
    parser.add_argument("--threshold", type=int, default=10, help="Rank cutoff for improved/regressed groups.")
    return parser.parse_args(argv)


def _prediction_path(ml_dir, action, dataset):
    return ml_dir / f"predictions_{action}_{dataset}_test.parquet"


def _rank_frame(path, label):
    df = pd.read_parquet(path)
    df = df.sort_values(["state_id", "pred"], ascending=[True, False]).copy()
    df[f"{label}_rank"] = df.groupby("state_id").cumcount() + 1
    top = (
        df.groupby("state_id")
        .head(1)[["state_id", "candidate_hero_id", "candidate_hero_name", "pred"]]
        .rename(
            columns={
                "candidate_hero_id": f"{label}_top_hero_id",
                "candidate_hero_name": f"{label}_top_hero_name",
                "pred": f"{label}_top_score",
            }
        )
    )
    target = df[df["target"] == 1].copy()
    keep = [
        "state_id",
        "match_id",
        "order",
        "draft_phase",
        "action_type",
        "acting_side",
        "acting_team_id",
        "opponent_team_id",
        "candidate_hero_id",
        "candidate_hero_name",
        "pred",
        f"{label}_rank",
    ]
    extra = [
        "candidate_own_role_fit_score",
        "candidate_own_role_conflict_score",
        "candidate_enemy_role_fit_score",
        "candidate_enemy_role_conflict_score",
        "candidate_core_prob",
        "candidate_support_prob",
        "candidate_flex_score",
        "own_team_candidate_contested_rate_patch",
        "opponent_team_candidate_contested_rate_patch",
    ]
    keep.extend([col for col in extra if col in target.columns])
    target = target[keep].rename(
        columns={
            "candidate_hero_id": "real_hero_id",
            "candidate_hero_name": "real_hero_name",
            "pred": f"{label}_real_score",
        }
    )
    return target.merge(top, on="state_id", how="left")


def _compare_action(ml_dir, report_dirs, action, left, right, threshold):
    left_path = _prediction_path(ml_dir, action, left)
    right_path = _prediction_path(ml_dir, action, right)
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"missing predictions for {action}: {left_path} / {right_path}")

    left_df = _rank_frame(left_path, "left")
    right_df = _rank_frame(right_path, "right")
    common_cols = [
        "state_id",
        "match_id",
        "order",
        "draft_phase",
        "action_type",
        "acting_side",
        "acting_team_id",
        "opponent_team_id",
        "real_hero_id",
        "real_hero_name",
    ]
    merged = left_df.merge(right_df, on=common_cols, how="inner", suffixes=("", "_right"))
    merged["rank_delta"] = merged["right_rank"] - merged["left_rank"]
    merged["improved_to_top10"] = (merged["left_rank"] > threshold) & (merged["right_rank"] <= threshold)
    merged["regressed_from_top10"] = (merged["left_rank"] <= threshold) & (merged["right_rank"] > threshold)

    improved = merged[merged["improved_to_top10"]].sort_values(["rank_delta", "right_rank"])
    regressed = merged[merged["regressed_from_top10"]].sort_values(["rank_delta", "left_rank"], ascending=[False, True])

    all_path = report_dirs["errors"] / f"{action}_{left}_vs_{right}_rank_comparison.csv"
    improved_path = report_dirs["errors"] / f"{action}_{left}_vs_{right}_improved_to_top{threshold}.csv"
    regressed_path = report_dirs["errors"] / f"{action}_{left}_vs_{right}_regressed_from_top{threshold}.csv"
    merged.to_csv(all_path, index=False)
    improved.to_csv(improved_path, index=False)
    regressed.to_csv(regressed_path, index=False)

    return {
        "action": action,
        "states": len(merged),
        "mean_left_rank": float(merged["left_rank"].mean()),
        "mean_right_rank": float(merged["right_rank"].mean()),
        "mean_rank_delta": float(merged["rank_delta"].mean()),
        f"improved_to_top{threshold}": int(merged["improved_to_top10"].sum()),
        f"regressed_from_top{threshold}": int(merged["regressed_from_top10"].sum()),
        "all_path": str(all_path),
        "improved_path": str(improved_path),
        "regressed_path": str(regressed_path),
    }


def _to_markdown(df):
    if df.empty:
        return "No data.\n"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def compare_dataset_errors(patch_label=PATCH_LABEL, left="players_team", right="players_team_role", threshold=10):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)
    report_dirs = get_ml_report_dirs(patch_label)
    rows = [_compare_action(ml_dir, report_dirs, action, left, right, threshold) for action in ["pick", "ban"]]
    summary = pd.DataFrame(rows)
    summary_path = report_dirs["errors"] / f"{left}_vs_{right}_summary.csv"
    md_path = report_dirs["errors"] / f"{left}_vs_{right}_summary.md"
    summary.to_csv(summary_path, index=False)
    md_path.write_text(f"# {left} vs {right} rank comparison\n\n" + _to_markdown(summary), encoding="utf-8")
    print(f"saved: {summary_path}")
    print(f"saved: {md_path}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    return compare_dataset_errors(args.patch_label, args.left, args.right, args.threshold)


if __name__ == "__main__":
    main()
