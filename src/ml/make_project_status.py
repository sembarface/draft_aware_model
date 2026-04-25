import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths


OUT = Path("project_status.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Write project status markdown report.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def table_info(path):
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    return {
        "path": str(path),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "na_top": df.isna().sum().sort_values(ascending=False).head(15).to_dict(),
    }


def read_json(path):
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_files(folder):
    if not folder.exists():
        return []

    rows = []
    for path in folder.rglob("*"):
        if path.is_file():
            rows.append({
                "path": str(path),
                "size_kb": round(path.stat().st_size / 1024, 2),
            })
    return rows


def add_table_section(lines, tables):
    lines.append("## Tables\n")
    for name, info in tables.items():
        lines.append(f"### {name}\n")

        if info is None:
            lines.append("Файл не найден.\n")
            continue

        lines.append(f"- path: `{info['path']}`\n")
        lines.append(f"- shape: `{info['shape']}`\n")

        lines.append("\nColumns:\n")
        lines.append("```text\n")
        for col in info["columns"]:
            lines.append(f"{col}\n")
        lines.append("```\n")

        lines.append("\nTop missing values:\n")
        lines.append("```text\n")
        for col, value in info["na_top"].items():
            lines.append(f"{col}: {value}\n")
        lines.append("```\n")


def add_json_section(lines, title, payload, missing_text):
    lines.append(f"### {title}\n")
    if payload:
        lines.append("```json\n")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        lines.append("\n```\n")
    else:
        lines.append(f"{missing_text}\n")


def make_project_status(patch_label=PATCH_LABEL):
    patch_num, base_dir, _, ml_dir, model_dir = get_patch_paths(patch_label)

    tables = {
        "matches": table_info(base_dir / "matches.parquet"),
        "players": table_info(base_dir / "players.parquet"),
        "picks_bans": table_info(base_dir / "picks_bans.parquet"),
        "heroes_stats": table_info(base_dir / "heroes_stats.parquet"),
        "draft_events": table_info(ml_dir / "draft_events.parquet"),
        "draft_states": table_info(ml_dir / "draft_states.parquet"),
        "draft_candidates_pick": table_info(ml_dir / "draft_candidates_pick.parquet"),
        "draft_candidates_ban": table_info(ml_dir / "draft_candidates_ban.parquet"),
        "hero_synergy": table_info(ml_dir / "hero_synergy.parquet"),
        "hero_matchups": table_info(ml_dir / "hero_matchups.parquet"),
        "hero_conditional_bans": table_info(ml_dir / "hero_conditional_bans.parquet"),
        "draft_candidates_pick_interactions": table_info(ml_dir / "draft_candidates_pick_interactions.parquet"),
        "draft_candidates_ban_interactions": table_info(ml_dir / "draft_candidates_ban_interactions.parquet"),
    }

    metrics = {
        "Pick base metrics": read_json(model_dir / "pick_base_metrics.json"),
        "Ban base metrics": read_json(model_dir / "ban_base_metrics.json"),
        "Pick interactions metrics": read_json(model_dir / "pick_interactions_metrics.json"),
        "Ban interactions metrics": read_json(model_dir / "ban_interactions_metrics.json"),
    }

    features = {
        "Pick base features": read_json(model_dir / "pick_base_features.json"),
        "Ban base features": read_json(model_dir / "ban_base_features.json"),
        "Pick interactions features": read_json(model_dir / "pick_interactions_features.json"),
        "Ban interactions features": read_json(model_dir / "ban_interactions_features.json"),
    }

    lines = []
    lines.append("# Project status\n")
    lines.append("## Patch\n")
    lines.append(f"`PATCH_LABEL = {patch_label}`\n")
    lines.append(f"`PATCH_NUM = {patch_num}`\n")

    add_table_section(lines, tables)

    lines.append("## Metrics\n")
    for title, payload in metrics.items():
        add_json_section(lines, title, payload, f"Файл {title} не найден.")

    lines.append("## Features\n")
    for title, payload in features.items():
        add_json_section(lines, title, payload, f"Файл {title} не найден.")

    lines.append("## Model files\n")
    lines.append("```text\n")
    for item in list_files(model_dir):
        lines.append(f"{item['path']} | {item['size_kb']} KB\n")
    lines.append("```\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print("Saved:", OUT)
    return OUT


def main(argv=None):
    args = parse_args(argv)
    return make_project_status(args.patch_label)


if __name__ == "__main__":
    main()
