import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs
from src.ml.role_utils import POSITION_COLS, load_hero_roles


DEFAULT_HEROES = [
    "Tiny",
    "Pangolier",
    "Windranger",
    "Mirana",
    "Pudge",
    "Marci",
    "Nature's Prophet",
    "Dawnbreaker",
    "Dragon Knight",
    "Huskar",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit selected hero role priors.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--heroes", nargs="*", default=DEFAULT_HEROES, help="Hero names to audit.")
    return parser.parse_args(argv)


def _flags(row):
    flags = []
    max_pos = max(row[col] for col in POSITION_COLS)
    active_positions = sum(row[col] >= 0.20 for col in POSITION_COLS)
    if max_pos >= 0.85 and row["flex_score"] < 0.25:
        flags.append("very rigid")
    if active_positions >= 3:
        flags.append("broad flex")
    if row["core_prob"] >= 0.85 and row["support_prob"] <= 0.15:
        flags.append("core-heavy")
    if row["support_prob"] >= 0.85 and row["core_prob"] <= 0.15:
        flags.append("support-heavy")
    return ", ".join(flags) if flags else "ok"


def _to_markdown(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def audit_hero_roles(patch_label=PATCH_LABEL, heroes=None):
    roles = load_hero_roles()
    heroes = heroes or DEFAULT_HEROES
    audit = roles[roles["hero_name"].isin(heroes)].copy()
    missing = sorted(set(heroes) - set(audit["hero_name"]))
    audit["dominant_position"] = audit[POSITION_COLS].idxmax(axis=1).str.replace("_prob", "", regex=False)
    audit["active_positions_ge_020"] = (audit[POSITION_COLS] >= 0.20).sum(axis=1)
    audit["audit_flag"] = audit.apply(_flags, axis=1)
    cols = [
        "hero_id",
        "hero_name",
        "dominant_position",
        "active_positions_ge_020",
        *POSITION_COLS,
        "core_prob",
        "support_prob",
        "flex_score",
        "audit_flag",
    ]
    audit = audit[cols].sort_values("hero_name")
    report_dirs = get_ml_report_dirs(patch_label)
    csv_path = report_dirs["features"] / "hero_role_audit.csv"
    md_path = report_dirs["features"] / "hero_role_audit.md"
    audit.to_csv(csv_path, index=False)
    text = "# Hero role audit\n\n" + _to_markdown(audit)
    if missing:
        text += "\nMissing heroes: " + ", ".join(missing) + "\n"
    md_path.write_text(text, encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    return audit


def main(argv=None):
    args = parse_args(argv)
    return audit_hero_roles(args.patch_label, args.heroes)


if __name__ == "__main__":
    main()
