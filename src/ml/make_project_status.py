from pathlib import Path
import json
import pandas as pd
import os


PATCH = 60

BASE = Path(f"data/patch_{PATCH}")
ML_DIR = BASE / "ml"
MODEL_DIR = Path(f"models/patch_{PATCH}")

OUT = Path("project_status.md")


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


tables = {
    "matches": table_info(BASE / "matches.parquet"),
    "players": table_info(BASE / "players.parquet"),
    "picks_bans": table_info(BASE / "picks_bans.parquet"),
    "heroes_stats": table_info(BASE / "heroes_stats.parquet"),
    "draft_events": table_info(ML_DIR / "draft_events.parquet"),
    "draft_states": table_info(ML_DIR / "draft_states.parquet"),
    "draft_candidates_pick": table_info(ML_DIR / "draft_candidates_pick.parquet"),
    "draft_candidates_ban": table_info(ML_DIR / "draft_candidates_ban.parquet"),
}

pick_metrics = read_json(MODEL_DIR / "pick_metrics.json")
ban_metrics = read_json(MODEL_DIR / "ban_metrics.json")

pick_features = read_json(MODEL_DIR / "pick_features.json")
ban_features = read_json(MODEL_DIR / "ban_features.json")

model_files = list_files(MODEL_DIR)

lines = []

lines.append(f"# Project status\n")
lines.append(f"## Patch\n")
lines.append(f"`PATCH = {PATCH}`\n")

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

lines.append("## Metrics\n")

lines.append("### Pick metrics\n")
if pick_metrics:
    lines.append("```json\n")
    lines.append(json.dumps(pick_metrics, ensure_ascii=False, indent=2))
    lines.append("\n```\n")
else:
    lines.append("Файл pick_metrics.json не найден.\n")

lines.append("### Ban metrics\n")
if ban_metrics:
    lines.append("```json\n")
    lines.append(json.dumps(ban_metrics, ensure_ascii=False, indent=2))
    lines.append("\n```\n")
else:
    lines.append("Файл ban_metrics.json не найден.\n")

lines.append("## Features\n")

lines.append("### Pick features\n")
if pick_features:
    lines.append("```json\n")
    lines.append(json.dumps(pick_features, ensure_ascii=False, indent=2))
    lines.append("\n```\n")
else:
    lines.append("Файл pick_features.json не найден.\n")

lines.append("### Ban features\n")
if ban_features:
    lines.append("```json\n")
    lines.append(json.dumps(ban_features, ensure_ascii=False, indent=2))
    lines.append("\n```\n")
else:
    lines.append("Файл ban_features.json не найден.\n")

lines.append("## Model files\n")
lines.append("```text\n")
for item in model_files:
    lines.append(f"{item['path']} | {item['size_kb']} KB\n")
lines.append("```\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("".join(lines))

print("Saved:", OUT)