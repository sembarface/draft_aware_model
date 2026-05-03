import argparse
import json

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs
from src.ml.feature_sets import DATASET_CHOICES, output_stem


METRICS = ["states", "top1", "top3", "top5", "top10", "mean_rank", "mrr"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export compact ML metric reports.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
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


def _metrics_rows(report_dirs):
    rows = []
    for action in ["pick", "ban"]:
        for dataset in DATASET_CHOICES:
            path = report_dirs["metrics"] / f"{output_stem(action, dataset)}_metrics.json"
            if not path.exists():
                continue
            metrics = _read_json(path)
            row = {"action": action, "dataset": dataset}
            row.update({metric: metrics.get(metric) for metric in METRICS})
            rows.append(row)
    return rows


def _write_metrics(report_dirs, rows):
    if not rows:
        print("warning: no metrics found to export")
        return None
    df = pd.DataFrame(rows)
    csv_path = report_dirs["metrics"] / "model_metrics.csv"
    md_path = report_dirs["metrics"] / "model_metrics.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text("# Model metrics\n\n" + _to_markdown(df), encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    return df


def _write_comparison(report_dirs, metrics_df):
    if metrics_df is None or metrics_df.empty:
        return
    rows = []
    for action in ["pick", "ban"]:
        pairs = [("base", "players"), ("players", "players_smooth")]
        for left_name, right_name in pairs:
            left = metrics_df[(metrics_df["action"] == action) & (metrics_df["dataset"] == left_name)]
            right = metrics_df[(metrics_df["action"] == action) & (metrics_df["dataset"] == right_name)]
            if left.empty or right.empty:
                continue
            left = left.iloc[0]
            right = right.iloc[0]
            for metric in METRICS:
                if metric == "states":
                    continue
                rows.append({
                    "action": action,
                    "comparison": f"{left_name}_vs_{right_name}",
                    "metric": metric,
                    left_name: left.get(metric),
                    right_name: right.get(metric),
                    "diff": right.get(metric) - left.get(metric),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path = report_dirs["metrics"] / "model_comparison.csv"
    md_path = report_dirs["metrics"] / "model_comparison.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text("# Model comparison\n\n" + _to_markdown(df), encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")


def _write_importance_markdown(report_dirs):
    frames = []
    for path in sorted(report_dirs["importance"].glob("*_feature_importance.csv")):
        df = pd.read_csv(path).head(30)
        df.insert(0, "model", path.name.replace("_feature_importance.csv", ""))
        frames.append(df)
    if not frames:
        return
    lines = ["# Feature importance\n\n"]
    for df in frames:
        model = df["model"].iloc[0]
        lines.append(f"## {model}\n\n")
        lines.append(_to_markdown(df[["feature", "importance"]]))
        lines.append("\n\n")
    path = report_dirs["importance"] / "feature_importance.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"saved: {path}")


def export_reports(patch_label=PATCH_LABEL):
    report_dirs = get_ml_report_dirs(patch_label)
    metrics_df = _write_metrics(report_dirs, _metrics_rows(report_dirs))
    _write_comparison(report_dirs, metrics_df)
    _write_importance_markdown(report_dirs)


def main(argv=None):
    args = parse_args(argv)
    return export_reports(args.patch_label)


if __name__ == "__main__":
    main()
