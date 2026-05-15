import argparse
from pathlib import Path

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs
from src.ml.role_utils import HERO_ROLES_PATH, POSITION_COLS, load_hero_roles


WATCHLIST_HEROES = [
    "Dawnbreaker",
    "Lone Druid",
    "Windranger",
    "Night Stalker",
    "Zeus",
    "Grimstroke",
    "Terrorblade",
    "Marci",
    "Tusk",
    "Tiny",
    "Rubick",
    "Hoodwink",
    "Ember Spirit",
    "Underlord",
    "Enigma",
    "Sand King",
    "Ancient Apparition",
    "Pugna",
    "Clockwerk",
    "Dark Seer",
    "Muerta",
    "Brewmaster",
    "Bristleback",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inspect hero role priors.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="Patch label for report directory.")
    parser.add_argument("--path", default=str(HERO_ROLES_PATH), help="Hero roles CSV path.")
    parser.add_argument("--heroes", nargs="*", default=WATCHLIST_HEROES, help="Hero names to inspect.")
    parser.add_argument("--min-hero-games", type=int, default=5, help="Low-sample warning threshold.")
    return parser.parse_args(argv)


def _load_raw(path):
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = load_hero_roles(path)
    for col in POSITION_COLS + ["core_prob", "support_prob", "flex_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "total_role_games" in df.columns:
        df["total_role_games"] = pd.to_numeric(df["total_role_games"], errors="coerce").fillna(0).astype(int)
    if "role_source" not in df.columns:
        df["role_source"] = "unknown"
    return df


def _warning(row, min_hero_games):
    warnings = []
    games = row.get("total_role_games")
    if pd.notna(games) and int(games) < min_hero_games:
        warnings.append(f"low sample: {int(games)} games")
    max_pos = max(row[col] for col in POSITION_COLS)
    if max_pos >= 0.95 and (pd.isna(games) or int(games) < min_hero_games):
        warnings.append("very rigid on low sample")
    if max_pos >= 0.95:
        warnings.append("very rigid")
    return "; ".join(dict.fromkeys(warnings)) or "ok"


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


def inspect_hero_roles(patch_label=PATCH_LABEL, path=HERO_ROLES_PATH, heroes=None, min_hero_games=5):
    heroes = heroes or WATCHLIST_HEROES
    roles = _load_raw(path)
    roles["dominant_position"] = roles[POSITION_COLS].idxmax(axis=1).str.replace("_prob", "", regex=False)
    roles["max_pos_prob"] = roles[POSITION_COLS].max(axis=1)
    roles["warning"] = roles.apply(lambda row: _warning(row, min_hero_games), axis=1)
    selected = roles[roles["hero_name"].isin(heroes)].copy().sort_values("hero_name")

    cols = [
        "hero_id",
        "hero_name",
        "dominant_position",
        "max_pos_prob",
        "total_role_games",
        "pos1_count",
        "pos2_count",
        "pos3_count",
        "pos4_count",
        "pos5_count",
        *POSITION_COLS,
        "core_prob",
        "support_prob",
        "flex_score",
        "role_source",
        "warning",
    ]
    cols = [col for col in cols if col in selected.columns]
    report_dirs = get_ml_report_dirs(patch_label)
    md_path = report_dirs["features"] / "hero_roles_diagnostics.md"
    csv_path = report_dirs["features"] / "hero_roles_diagnostics.csv"
    selected[cols].to_csv(csv_path, index=False)
    md_path.write_text("# Hero roles diagnostics\n\n" + _to_markdown(selected[cols]), encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    return selected[cols]


def main(argv=None):
    args = parse_args(argv)
    return inspect_hero_roles(args.patch_label, args.path, args.heroes, args.min_hero_games)


if __name__ == "__main__":
    main()
