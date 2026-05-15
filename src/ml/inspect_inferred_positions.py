import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs
from src.ml.build_hero_roles import WATCHLIST_HEROES, infer_positions, load_player_rows


STAT_COLS = [
    "last_hits",
    "net_worth",
    "gold_per_min",
    "xp_per_min",
    "obs_placed",
    "sen_placed",
    "camps_stacked",
    "rune_pickups",
    "hero_damage",
    "tower_damage",
    "stuns",
    "carry_item_score",
    "offlane_item_score",
    "support_item_score",
    "lane_safe",
    "lane_mid",
    "lane_off",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inspect inferred player positions for one patch.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="Patch label, e.g. 7.41.")
    parser.add_argument("--match-id", type=int, default=None, help="Optional match id for sample output.")
    return parser.parse_args(argv)


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


def _position_distribution(df):
    counts = df["inferred_position"].value_counts().rename_axis("inferred_position").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum() if counts["count"].sum() else 0.0
    return counts.sort_values("inferred_position")


def _average_stats(df):
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.groupby("inferred_position")[STAT_COLS].mean(numeric_only=True).reset_index()


def _core_sanity(df):
    cols = [
        "last_hits",
        "net_worth",
        "gold_per_min",
        "xp_per_min",
        "tower_damage",
        "hero_damage",
        "carry_item_score",
        "offlane_item_score",
        "lane_safe",
        "lane_mid",
        "lane_off",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df[df["inferred_position"].isin(["pos1", "pos2", "pos3"])].groupby("inferred_position")[cols].mean().reset_index()


def _support_sanity(df):
    cols = [
        "obs_placed",
        "sen_placed",
        "stuns",
        "rune_pickups",
        "hero_damage",
        "support_item_score",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df[df["inferred_position"].isin(["pos4", "pos5"])].groupby("inferred_position")[cols].mean().reset_index()


def _watchlist_positions(df):
    part = df[df["hero_name"].isin(WATCHLIST_HEROES)].copy()
    if part.empty:
        return pd.DataFrame(columns=["hero_name", "inferred_position", "count", "share"])
    counts = (
        part.groupby(["hero_name", "inferred_position"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("hero_name")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts.sort_values(["hero_name", "share"], ascending=[True, False])


def _watchlist_wide(df):
    part = df[df["hero_name"].isin(WATCHLIST_HEROES)].copy()
    if part.empty:
        return pd.DataFrame()
    counts = part.groupby(["hero_name", "inferred_position"]).size().reset_index(name="count")
    wide = counts.pivot_table(index="hero_name", columns="inferred_position", values="count", fill_value=0).reset_index()
    for pos in ["pos1", "pos2", "pos3", "pos4", "pos5"]:
        if pos not in wide.columns:
            wide[pos] = 0
        wide[f"{pos}_count"] = wide[pos].astype(int)
    count_cols = [f"{pos}_count" for pos in ["pos1", "pos2", "pos3", "pos4", "pos5"]]
    wide["total"] = wide[count_cols].sum(axis=1)
    for pos in ["pos1", "pos2", "pos3", "pos4", "pos5"]:
        wide[f"{pos}_prob"] = wide[f"{pos}_count"] / wide["total"].where(wide["total"] > 0, 1)
    prob_cols = [f"{pos}_prob" for pos in ["pos1", "pos2", "pos3", "pos4", "pos5"]]
    return wide[["hero_name", *count_cols, "total", *prob_cols]].sort_values("hero_name")


def inspect_inferred_positions(patch_label=PATCH_LABEL, match_id=None):
    players = load_player_rows([patch_label])
    if match_id is not None:
        players = players[players["match_id"] == match_id].copy()
    positioned = infer_positions(players)

    report_dirs = get_ml_report_dirs(patch_label)
    sample_path = report_dirs["features"] / "inferred_positions_sample.csv"
    md_path = report_dirs["features"] / "inferred_positions_summary.md"

    sample_cols = [
        "match_id",
        "side",
        "team_id",
        "account_id",
        "hero_id",
        "hero_name",
        "inferred_position",
        *[col for col in STAT_COLS if col in positioned.columns],
        "lane",
        "lane_role",
        "lane_bucket",
        "lane_safe",
        "lane_mid",
        "lane_off",
        "is_roaming",
        "carry_item_score",
        "offlane_item_score",
        "support_item_score",
    ]
    sample_cols = [col for col in sample_cols if col in positioned.columns]
    sample = positioned.sort_values(["match_id", "side", "inferred_position"]).head(500)
    sample[sample_cols].to_csv(sample_path, index=False)

    lines = ["# Inferred player positions\n\n"]
    lines.append("## Position distribution\n\n")
    lines.append(_to_markdown(_position_distribution(positioned)))
    lines.append("\n## Average stats by inferred position\n\n")
    lines.append(_to_markdown(_average_stats(positioned)))
    lines.append("\n## Core sanity checks\n\n")
    lines.append(_to_markdown(_core_sanity(positioned)))
    lines.append("\n## Support sanity checks\n\n")
    lines.append(_to_markdown(_support_sanity(positioned)))
    lines.append("\n## Watchlist heroes by inferred position\n\n")
    lines.append(_to_markdown(_watchlist_positions(positioned)))
    lines.append("\n## Watchlist hero position priors\n\n")
    lines.append(_to_markdown(_watchlist_wide(positioned)))
    if match_id is not None:
        lines.append(f"\n## Match {match_id} sample\n\n")
        lines.append(_to_markdown(positioned[sample_cols].sort_values(["side", "inferred_position"])))
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"saved: {sample_path}")
    print(f"saved: {md_path}")
    return positioned


def main(argv=None):
    args = parse_args(argv)
    return inspect_inferred_positions(args.patch_label, args.match_id)


if __name__ == "__main__":
    main()
