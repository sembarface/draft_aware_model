import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PATCH_LABEL, get_ml_report_dirs, get_patch_num
from src.ml.role_utils import HERO_ROLES_PATH, POSITION_COLS, default_role_vector


SMOOTH_K = 5
PLAYER_PRIOR_K = 5
POSITIONS = ["pos1", "pos2", "pos3", "pos4", "pos5"]
DEFAULT_POS_PRIOR = {pos: 0.2 for pos in POSITIONS}
HEROES_CSV = Path("data/heroes.csv")

WATCHLIST_HEROES = [
    "Sand King",
    "Underlord",
    "Brewmaster",
    "Tiny",
    "Dawnbreaker",
    "Terrorblade",
    "Lone Druid",
    "Bristleback",
    "Ember Spirit",
    "Enigma",
    "Clockwerk",
    "Tusk",
    "Rubick",
    "Hoodwink",
    "Marci",
    "Zeus",
    "Windranger",
    "Muerta",
    "Dark Seer",
    "Ancient Apparition",
    "Grimstroke",
    "Pugna",
]

ITEM_SLOT_COLS = [
    "item_0",
    "item_1",
    "item_2",
    "item_3",
    "item_4",
    "item_5",
    "backpack_0",
    "backpack_1",
    "backpack_2",
    "item_neutral",
]

PLAYER_COLS = [
    "match_id",
    "start_time",
    "account_id",
    "team_id",
    "side",
    "is_radiant",
    "hero_id",
    "hero_name",
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
    "hero_healing",
    "lane",
    "lane_role",
    "is_roaming",
    "lane_efficiency",
    *ITEM_SLOT_COLS,
    "purchase_log_json",
]

# Compact item signals. IDs are stable OpenDota/Dota item ids where known; purchase_log
# keys cover the same groups when only item names are available in raw JSON.
CARRY_ITEM_IDS = {137, 139, 141, 145, 147, 156, 158, 160, 166, 174, 208, 236, 249, 263}
OFFLANE_ITEM_IDS = {1, 90, 112, 116, 119, 125, 127, 164, 180, 185, 210, 226, 231, 242}
SUPPORT_ITEM_IDS = {37, 79, 90, 102, 226, 229, 231, 232, 254, 256, 269, 301}

CARRY_ITEM_NAMES = {
    "battle_fury",
    "manta",
    "butterfly",
    "satanic",
    "daedalus",
    "mjollnir",
    "maelstrom",
    "hurricane_pike",
    "dragon_lance",
    "diffusal_blade",
    "radiance",
    "skadi",
    "eye_of_skadi",
    "silver_edge",
    "abyssal_blade",
}
OFFLANE_ITEM_NAMES = {
    "blink",
    "pipe",
    "crimson_guard",
    "vanguard",
    "lotus_orb",
    "guardian_greaves",
    "shivas_guard",
    "black_king_bar",
    "heavens_halberd",
    "heaven_halberd",
    "blade_mail",
    "blademail",
    "assault",
    "assault_cuirass",
    "drums",
    "boots_of_bearing",
    "helm_of_the_dominator",
    "vlads",
    "vladmir",
}
SUPPORT_ITEM_NAMES = {
    "glimmer_cape",
    "force_staff",
    "ghost",
    "ghost_scepter",
    "aeon_disk",
    "aether_lens",
    "mekansm",
    "guardian_greaves",
    "holy_locket",
    "solar_crest",
    "pavise",
    "pipe",
    "lotus_orb",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build empirical hero role priors from local OpenDota parquet tables.")
    parser.add_argument("--patch-label", default=None, help="Single patch label, e.g. 7.41.")
    parser.add_argument("--patch-labels", nargs="*", default=None, help="Patch labels to aggregate.")
    parser.add_argument("--min-hero-games", type=int, default=5, help="Low-sample threshold for diagnostics.")
    parser.add_argument("--output-path", default=str(HERO_ROLES_PATH), help="Output CSV path.")
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostics reports.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostics reports.")
    return parser.parse_args(argv)


def _patch_labels(args):
    if args.patch_labels:
        return args.patch_labels
    if args.patch_label:
        return [args.patch_label]
    return [PATCH_LABEL]


def _read_players(patch_label):
    patch_num = get_patch_num(patch_label)
    path = Path(f"data/patch_{patch_num}/players.parquet")
    if not path.exists():
        print(f"warning: missing {path}")
        return pd.DataFrame(columns=PLAYER_COLS)
    df = pd.read_parquet(path)

    if "start_time" not in df.columns:
        matches_path = Path(f"data/patch_{patch_num}/matches.parquet")
        if matches_path.exists():
            matches = pd.read_parquet(matches_path, columns=["match_id", "start_time"])
            df = df.merge(matches, on="match_id", how="left")

    for col in PLAYER_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[PLAYER_COLS].copy()
    df["patch_label"] = patch_label
    return df


def load_player_rows(patch_labels):
    frames = [_read_players(label) for label in patch_labels]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=[*PLAYER_COLS, "patch_label"])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["match_id", "hero_id", "side"]).copy()
    df["hero_id"] = pd.to_numeric(df["hero_id"], errors="coerce")
    df = df.dropna(subset=["hero_id"]).copy()
    df["hero_id"] = df["hero_id"].astype(int)
    return df


def _rank_percentile(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(values) <= 1:
        return pd.Series([1.0] * len(values), index=values.index)
    if values.nunique() <= 1:
        return pd.Series([0.5] * len(values), index=values.index)
    return values.rank(method="average", pct=True)


def _boolish(series):
    return series.astype("string").fillna("").str.lower().isin(["true", "1", "1.0", "yes"])


def _numeric(row, col):
    value = row.get(col)
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def lane_bucket(row):
    """Best-effort OpenDota lane bucket.

    lane_role is treated as the stronger hint when available. For raw lane values,
    lane=2 is mid. lane=1/3 are mapped with side awareness because some OpenDota
    exports encode bottom/top rather than safe/off directly.
    """
    lane_role = _numeric(row, "lane_role")
    lane = _numeric(row, "lane")
    is_radiant = bool(row.get("is_radiant")) if pd.notna(row.get("is_radiant")) else row.get("side") == "radiant"

    if lane_role == 2:
        return "mid"
    if lane_role == 1:
        return "safe"
    if lane_role == 3:
        return "off"

    if lane == 2:
        return "mid"
    if lane == 1:
        return "safe" if is_radiant else "off"
    if lane == 3:
        return "off" if is_radiant else "safe"
    return "unknown"


def is_mid_lane(row):
    return lane_bucket(row) == "mid"


def is_safe_lane(row):
    return lane_bucket(row) == "safe"


def is_off_lane(row):
    return lane_bucket(row) == "off"


def _purchase_names(row):
    raw = row.get("purchase_log_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return set()
    if not isinstance(parsed, list):
        return set()
    names = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = item.get("key") or item.get("item") or item.get("name")
        if key:
            names.add(str(key))
    return names


def _item_ids(row):
    ids = set()
    for col in ITEM_SLOT_COLS:
        value = row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            item_id = int(float(value))
        except Exception:
            continue
        if item_id > 0:
            ids.add(item_id)
    return ids


def _item_group_score(row, id_set, name_set):
    ids = _item_ids(row)
    names = _purchase_names(row)
    count = len(ids & id_set) + len(names & name_set)
    return min(float(count), 3.0) / 3.0


def _add_scores(team_df):
    df = team_df.copy()
    for col in [
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
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[f"_{col}_pct"] = _rank_percentile(df[col])

    df["_farm_score"] = (
        df["_last_hits_pct"]
        + df["_net_worth_pct"]
        + df["_gold_per_min_pct"]
        + df["_xp_per_min_pct"]
    )
    df["_support_score"] = (
        df["_obs_placed_pct"]
        + df["_sen_placed_pct"]
        + df["_camps_stacked_pct"]
        + df["_rune_pickups_pct"]
        - df["_last_hits_pct"]
    )
    df["_damage_score"] = df["_hero_damage_pct"] + df["_tower_damage_pct"]
    df["_is_roaming_hint"] = _boolish(df["is_roaming"]) if "is_roaming" in df.columns else False
    df["_vision_score"] = df["_obs_placed_pct"] + df["_sen_placed_pct"]
    df["_roam_score"] = (
        df["_rune_pickups_pct"]
        + df["_damage_score"]
        + df["_stuns_pct"]
        + df["_is_roaming_hint"].astype(float)
    )
    df["_hard_support_score"] = df["_vision_score"] + df["_support_score"] - df["_farm_score"]
    df["_soft_support_score"] = df["_roam_score"] + 0.5 * df["_support_score"] + 0.5 * df["_damage_score"]
    df["_lane"] = pd.to_numeric(df.get("lane"), errors="coerce")
    df["_lane_role"] = pd.to_numeric(df.get("lane_role"), errors="coerce")
    df["lane_bucket"] = df.apply(lane_bucket, axis=1)
    df["lane_safe"] = df["lane_bucket"].eq("safe").astype(float)
    df["lane_mid"] = df["lane_bucket"].eq("mid").astype(float)
    df["lane_off"] = df["lane_bucket"].eq("off").astype(float)
    df["carry_item_score"] = df.apply(lambda row: _item_group_score(row, CARRY_ITEM_IDS, CARRY_ITEM_NAMES), axis=1)
    df["offlane_item_score"] = df.apply(lambda row: _item_group_score(row, OFFLANE_ITEM_IDS, OFFLANE_ITEM_NAMES), axis=1)
    df["support_item_score"] = df.apply(lambda row: _item_group_score(row, SUPPORT_ITEM_IDS, SUPPORT_ITEM_NAMES), axis=1)
    return df


def get_player_position_prior(account_id, acc, k=PLAYER_PRIOR_K):
    if pd.isna(account_id):
        return DEFAULT_POS_PRIOR.copy()
    counts = acc.get(account_id, {})
    total = sum(counts.get(pos, 0) for pos in POSITIONS)
    return {
        pos: (counts.get(pos, 0) + k * DEFAULT_POS_PRIOR[pos]) / (total + k)
        for pos in POSITIONS
    }


def _position_scores(row, prior):
    lower_farm = max(0.0, 4.0 - float(row["_farm_score"])) / 4.0
    safe = float(row["lane_safe"])
    mid = float(row["lane_mid"])
    off = float(row["lane_off"])
    roaming = float(row["_is_roaming_hint"])
    return {
        "pos1": (
            2.1 * safe
            + 1.6 * row["carry_item_score"]
            + 1.1 * row["_farm_score"]
            + 1.0 * row["_tower_damage_pct"]
            + 2.0 * prior["pos1"]
            - 0.8 * row["_vision_score"]
            - 0.4 * row["_support_score"]
            - 0.8 * roaming
        ),
        "pos2": (
            2.4 * mid
            + 1.2 * row["_xp_per_min_pct"]
            + 0.9 * row["_rune_pickups_pct"]
            + 0.8 * row["_damage_score"]
            + 0.5 * row["_farm_score"]
            + 2.0 * prior["pos2"]
        ),
        "pos3": (
            2.1 * off
            + 1.4 * row["offlane_item_score"]
            + 0.8 * row["_farm_score"]
            + 0.6 * row["_damage_score"]
            + 0.5 * row["_stuns_pct"]
            + 2.0 * prior["pos3"]
            - 0.5 * row["carry_item_score"]
        ),
        "pos4": (
            1.5 * roaming
            + 1.0 * row["_stuns_pct"]
            + 1.1 * row["_soft_support_score"]
            + 0.9 * row["support_item_score"]
            + 0.5 * row["_damage_score"]
            + 1.0 * lower_farm
            + 2.0 * prior["pos4"]
        ),
        "pos5": (
            1.2 * row["_hard_support_score"]
            + 1.4 * row["_vision_score"]
            + 1.0 * row["support_item_score"]
            + 1.2 * lower_farm
            + 2.0 * prior["pos5"]
        ),
    }


def assign_positions_optimal(team_df, player_position_counts=None):
    player_position_counts = player_position_counts or {}
    df = _add_scores(team_df)
    if df.empty:
        return df.assign(inferred_position=pd.Series(dtype="object"))

    indices = list(df.index)
    score_rows = []
    for idx, row in df.iterrows():
        prior = get_player_position_prior(row.get("account_id"), player_position_counts)
        for pos in POSITIONS:
            df.at[idx, f"player_{pos}_prior"] = prior[pos]
        score_rows.append(_position_scores(row, prior))

    n = len(indices)
    if n <= len(POSITIONS):
        best_perm = None
        best_score = -np.inf
        for perm in permutations(POSITIONS, n):
            score = sum(score_rows[i][perm[i]] for i in range(n))
            if score > best_score:
                best_score = score
                best_perm = perm
        assignments = {indices[i]: best_perm[i] for i in range(n)} if best_perm else {}
    else:
        row_best = [max(scores.values()) for scores in score_rows]
        selected = sorted(range(n), key=lambda i: row_best[i], reverse=True)[: len(POSITIONS)]
        best_perm = None
        best_score = -np.inf
        for perm in permutations(POSITIONS):
            score = sum(score_rows[row_i][perm[i]] for i, row_i in enumerate(selected))
            if score > best_score:
                best_score = score
                best_perm = perm
        assignments = {indices[row_i]: best_perm[i] for i, row_i in enumerate(selected)} if best_perm else {}
        for i, idx in enumerate(indices):
            assignments.setdefault(idx, "pos5")

    out = df.copy()
    out["inferred_position"] = out.index.map(assignments).fillna("pos5")
    return out


def infer_positions_for_team(team_df, player_position_counts=None):
    return assign_positions_optimal(team_df, player_position_counts)


def infer_positions(players):
    if players.empty:
        return players.assign(inferred_position=pd.Series(dtype="object"))

    work = players.copy()
    work["_sort_start_time"] = pd.to_numeric(work.get("start_time"), errors="coerce").fillna(0)
    work["_sort_match_id"] = pd.to_numeric(work["match_id"], errors="coerce").fillna(0)
    work = work.sort_values(["_sort_start_time", "_sort_match_id", "side"]).copy()

    player_position_counts = {}
    rows = []
    for _, match_group in work.groupby(["_sort_start_time", "_sort_match_id"], sort=False):
        positioned_teams = []
        for _, team in match_group.groupby("side", sort=False):
            positioned_teams.append(infer_positions_for_team(team, player_position_counts))
        match_positioned = pd.concat(positioned_teams, ignore_index=False) if positioned_teams else match_group
        rows.append(match_positioned)
        for _, row in match_positioned.iterrows():
            account_id = row.get("account_id")
            pos = row.get("inferred_position")
            if pd.isna(account_id) or pos not in POSITIONS:
                continue
            counts = player_position_counts.setdefault(account_id, {position: 0 for position in POSITIONS})
            counts[pos] += 1

    out = pd.concat(rows, ignore_index=True) if rows else work
    return out.drop(columns=[col for col in ["_sort_start_time", "_sort_match_id"] if col in out.columns])


def _heroes_reference():
    heroes = pd.read_csv(HEROES_CSV).rename(columns={"id": "hero_id", "name": "hero_name"})
    return heroes[["hero_id", "hero_name"]]


def build_role_table(positioned, min_hero_games=5):
    heroes = _heroes_reference()
    if positioned.empty:
        counts = pd.DataFrame(columns=["hero_id", "hero_name", "inferred_position", "count"])
    else:
        counts = (
            positioned.groupby(["hero_id", "hero_name", "inferred_position"])
            .size()
            .reset_index(name="count")
        )
    wide = counts.pivot_table(
        index=["hero_id", "hero_name"],
        columns="inferred_position",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for pos in POSITIONS:
        if pos not in wide.columns:
            wide[pos] = 0
        wide[f"{pos}_count"] = wide[pos].astype(int)
    count_cols = [f"{pos}_count" for pos in POSITIONS]
    wide["total_role_games"] = wide[count_cols].sum(axis=1)
    result = heroes.merge(wide[["hero_id", *count_cols, "total_role_games"]], on="hero_id", how="left")
    for col in count_cols:
        result[col] = result[col].fillna(0).astype(int)
    result["total_role_games"] = result["total_role_games"].fillna(0).astype(int)
    for pos in POSITIONS:
        result[f"{pos}_prob"] = (
            result[f"{pos}_count"] + SMOOTH_K * DEFAULT_POS_PRIOR[pos]
        ) / (result["total_role_games"] + SMOOTH_K)
    result["core_prob"] = result["pos1_prob"] + result["pos2_prob"] + result["pos3_prob"]
    result["support_prob"] = result["pos4_prob"] + result["pos5_prob"]
    result["flex_score"] = (1.0 - result[POSITION_COLS].max(axis=1)).clip(0.0, 1.0)

    default_mask = result["total_role_games"].eq(0)
    for col, value in default_role_vector().items():
        if col in result.columns:
            result.loc[default_mask, col] = value
    result["role_source"] = "data_inferred"
    result["low_sample"] = result["total_role_games"] < min_hero_games
    return result


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


def write_diagnostics(role_table, players, patch_labels, min_hero_games, patch_label_for_reports):
    report_dirs = get_ml_report_dirs(patch_label_for_reports)
    diagnostics_path = report_dirs["features"] / "hero_roles_inferred_diagnostics.csv"
    md_path = report_dirs["features"] / "hero_roles_inferred.md"

    diagnostic_cols = [
        "hero_id",
        "hero_name",
        "pos1_count",
        "pos2_count",
        "pos3_count",
        "pos4_count",
        "pos5_count",
        "total_role_games",
        *POSITION_COLS,
        "core_prob",
        "support_prob",
        "flex_score",
        "role_source",
    ]
    role_table[diagnostic_cols].to_csv(diagnostics_path, index=False)

    max_pos = role_table[POSITION_COLS].max(axis=1)
    flexible = role_table.sort_values(["flex_score", "total_role_games"], ascending=[False, False]).head(30)
    rigid = role_table.assign(max_pos_prob=max_pos).sort_values(["max_pos_prob", "total_role_games"], ascending=[False, False]).head(30)
    low_sample = role_table[role_table["total_role_games"] < min_hero_games].sort_values("total_role_games").head(30)
    watchlist = role_table[role_table["hero_name"].isin(WATCHLIST_HEROES)].sort_values("hero_name")

    lines = ["# Empirical hero role priors\n\n"]
    lines.append("Hero role priors are inferred strictly from local OpenDota match/player statistics. Manual hero-role overrides are not used.\n\n")
    lines.append("## Summary\n\n")
    summary = pd.DataFrame([
        {"metric": "patch_labels_used", "value": ", ".join(patch_labels)},
        {"metric": "rows_used", "value": len(players)},
        {"metric": "unique_matches", "value": players["match_id"].nunique() if not players.empty else 0},
        {"metric": "unique_heroes", "value": players["hero_id"].nunique() if not players.empty else 0},
        {"metric": "heroes_with_inferred_roles", "value": int((role_table["total_role_games"] > 0).sum())},
        {"metric": "heroes_with_default_vectors", "value": int((role_table["total_role_games"] == 0).sum())},
        {"metric": "manual_overrides_applied", "value": 0},
    ])
    lines.append(_to_markdown(summary))
    lines.append("\n## Top-30 most flexible heroes\n\n")
    lines.append(_to_markdown(flexible[diagnostic_cols].head(30)))
    lines.append("\n## Top-30 most rigid heroes\n\n")
    lines.append(_to_markdown(rigid[["hero_id", "hero_name", "max_pos_prob", *diagnostic_cols[7:]]].head(30)))
    lines.append("\n## Low sample heroes\n\n")
    lines.append(_to_markdown(low_sample[diagnostic_cols].head(30)))
    lines.append("\n## Watchlist heroes\n\n")
    lines.append(_to_markdown(watchlist[diagnostic_cols]))
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"saved: {diagnostics_path}")
    print(f"saved: {md_path}")


def build_hero_roles(
    patch_labels,
    min_hero_games=5,
    output_path=HERO_ROLES_PATH,
    diagnostics=False,
):
    players = load_player_rows(patch_labels)
    positioned = infer_positions(players)
    role_table = build_role_table(positioned, min_hero_games=min_hero_games)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "hero_id",
        "hero_name",
        *POSITION_COLS,
        "core_prob",
        "support_prob",
        "flex_score",
        "total_role_games",
        "role_source",
    ]
    role_table[output_cols].to_csv(output_path, index=False)
    print(f"saved: {output_path}")
    if diagnostics:
        write_diagnostics(role_table, positioned, patch_labels, min_hero_games, patch_labels[-1] if patch_labels else PATCH_LABEL)
    return role_table


def main(argv=None):
    args = parse_args(argv)
    labels = _patch_labels(args)
    write_reports = args.diagnostics or not args.no_diagnostics
    return build_hero_roles(
        labels,
        args.min_hero_games,
        args.output_path,
        diagnostics=write_reports,
    )


if __name__ == "__main__":
    main()
