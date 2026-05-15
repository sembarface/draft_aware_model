import argparse
from collections import defaultdict

import pandas as pd

from src.config import PATCH_LABEL, PATCH_MAP, get_patch_paths, get_previous_patch_labels
from src.ml.feature_sets import TEAM_PRIORITY_FEATURES


EARLY_PICK_MAX_ORDER = 9
HEROES_PATH = "data/heroes.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add team hero priority features to players candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _empty_acc():
    return {
        "team_matches": defaultdict(int),
        "team_hero_picks": defaultdict(int),
        "team_hero_pick_wins": defaultdict(int),
        "team_hero_early_picks": defaultdict(int),
        "team_hero_pick_order_sum": defaultdict(float),
        "team_hero_bans_against": defaultdict(int),
    }


def _team_id(value):
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _hero_id(value):
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _team_side(value):
    if value is None or pd.isna(value):
        return None
    try:
        side = int(value)
    except Exception:
        return None
    return side if side in {0, 1} else None


def _load_patch_tables(patch_label):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    matches_path = base_dir / "matches.parquet"
    picks_path = base_dir / "picks_bans.parquet"
    if not matches_path.exists() or not picks_path.exists():
        print(f"warning: missing matches/picks_bans for patch {patch_label}; skipped")
        return None, None
    return pd.read_parquet(matches_path), pd.read_parquet(picks_path)


def _match_info(matches):
    cols = ["match_id", "radiant_team_id", "dire_team_id", "radiant_win"]
    return matches[[col for col in cols if col in matches.columns]].drop_duplicates("match_id").set_index("match_id")


def _update_acc(acc, matches, picks_bans):
    if matches is None or matches.empty:
        return
    info = _match_info(matches)

    for _, row in info.reset_index().iterrows():
        radiant_id = _team_id(row.get("radiant_team_id"))
        dire_id = _team_id(row.get("dire_team_id"))
        if radiant_id is not None:
            acc["team_matches"][radiant_id] += 1
        if dire_id is not None:
            acc["team_matches"][dire_id] += 1

    if picks_bans is None or picks_bans.empty:
        return

    part = picks_bans[picks_bans["match_id"].isin(info.index)].copy()
    for _, row in part.iterrows():
        match_id = row.get("match_id")
        if match_id not in info.index:
            continue
        match = info.loc[match_id]
        radiant_id = _team_id(match.get("radiant_team_id"))
        dire_id = _team_id(match.get("dire_team_id"))
        if radiant_id is None or dire_id is None:
            continue

        hero_id = _hero_id(row.get("hero_id"))
        if hero_id is None:
            continue

        side = _team_side(row.get("team"))
        if side is None:
            continue
        acting_team_id = radiant_id if side == 0 else dire_id
        opponent_team_id = dire_id if side == 0 else radiant_id
        radiant_win_value = match.get("radiant_win")
        has_winner = not pd.isna(radiant_win_value)
        radiant_win = bool(radiant_win_value) if has_winner else None
        acting_team_win = (radiant_win if side == 0 else not radiant_win) if has_winner else None

        if bool(row.get("is_pick")):
            key = (acting_team_id, hero_id)
            order = int(row.get("order", 0) or 0)
            acc["team_hero_picks"][key] += 1
            acc["team_hero_pick_order_sum"][key] += order
            if order <= EARLY_PICK_MAX_ORDER:
                acc["team_hero_early_picks"][key] += 1
            if acting_team_win is True:
                acc["team_hero_pick_wins"][key] += 1
        else:
            acc["team_hero_bans_against"][(opponent_team_id, hero_id)] += 1


def _initial_alltime_acc(patch_label):
    acc = _empty_acc()
    for old_label in get_previous_patch_labels(patch_label):
        matches, picks_bans = _load_patch_tables(old_label)
        _update_acc(acc, matches, picks_bans)
    return acc


def build_full_team_priority_acc(patch_label=PATCH_LABEL):
    """Build accumulators for local UI inference using all available local data up to the selected patch."""
    alltime_acc = _empty_acc()
    patch_acc = _empty_acc()
    target_num = PATCH_MAP[patch_label]
    for label, patch_num in PATCH_MAP.items():
        if patch_num > target_num:
            continue
        matches, picks_bans = _load_patch_tables(label)
        _update_acc(alltime_acc, matches, picks_bans)
        if label == patch_label:
            _update_acc(patch_acc, matches, picks_bans)
    return alltime_acc, patch_acc


def _team_features(team_id, hero_id, acc, prefix, scope):
    team_id = _team_id(team_id)
    hero_id = _hero_id(hero_id)
    if team_id is None or hero_id is None:
        return _default_features(prefix, scope)

    matches = acc["team_matches"].get(team_id, 0)
    pick_count = acc["team_hero_picks"].get((team_id, hero_id), 0)
    pick_wins = acc["team_hero_pick_wins"].get((team_id, hero_id), 0)
    early_pick_count = acc["team_hero_early_picks"].get((team_id, hero_id), 0)
    pick_order_sum = acc["team_hero_pick_order_sum"].get((team_id, hero_id), 0.0)
    ban_against_count = acc["team_hero_bans_against"].get((team_id, hero_id), 0)
    contested_count = pick_count + ban_against_count

    return {
        f"{prefix}team_candidate_matches_{scope}": matches,
        f"{prefix}team_candidate_pick_count_{scope}": pick_count,
        f"{prefix}team_candidate_pick_rate_{scope}": pick_count / matches if matches else 0.0,
        f"{prefix}team_candidate_winrate_{scope}": pick_wins / pick_count if pick_count else 0.5,
        f"{prefix}team_candidate_early_pick_rate_{scope}": early_pick_count / pick_count if pick_count else 0.0,
        f"{prefix}team_candidate_avg_pick_order_{scope}": pick_order_sum / pick_count if pick_count else 0.0,
        f"{prefix}team_candidate_ban_against_count_{scope}": ban_against_count,
        f"{prefix}team_candidate_ban_against_rate_{scope}": ban_against_count / matches if matches else 0.0,
        f"{prefix}team_candidate_contested_count_{scope}": contested_count,
        f"{prefix}team_candidate_contested_rate_{scope}": contested_count / matches if matches else 0.0,
    }


def _default_features(prefix, scope):
    return {
        f"{prefix}team_candidate_matches_{scope}": 0,
        f"{prefix}team_candidate_pick_count_{scope}": 0,
        f"{prefix}team_candidate_pick_rate_{scope}": 0.0,
        f"{prefix}team_candidate_winrate_{scope}": 0.5,
        f"{prefix}team_candidate_early_pick_rate_{scope}": 0.0,
        f"{prefix}team_candidate_avg_pick_order_{scope}": 0.0,
        f"{prefix}team_candidate_ban_against_count_{scope}": 0,
        f"{prefix}team_candidate_ban_against_rate_{scope}": 0.0,
        f"{prefix}team_candidate_contested_count_{scope}": 0,
        f"{prefix}team_candidate_contested_rate_{scope}": 0.0,
    }


def team_priority_features_for_teams(hero_id, acting_team_id, opponent_team_id, alltime_acc, patch_acc):
    features = {}
    for scope, acc in [("alltime", alltime_acc), ("patch", patch_acc)]:
        features.update(_team_features(acting_team_id, hero_id, acc, "own_", scope))
        features.update(_team_features(opponent_team_id, hero_id, acc, "opponent_", scope))
    return features


def team_priority_features_for_team(hero_id, team_id, alltime_acc, patch_acc):
    features = {}
    for scope, acc in [("alltime", alltime_acc), ("patch", patch_acc)]:
        features.update(_team_features(team_id, hero_id, acc, "", scope))
    return features


def _sort_matches(matches):
    matches = matches.copy()
    matches["_sort_start_time"] = pd.to_numeric(matches["start_time"], errors="coerce").fillna(float("inf"))
    return matches.sort_values(["_sort_start_time", "match_id"])


def _match_slice(df, match_ids):
    return df[df["match_id"].isin(match_ids)].copy() if df is not None and not df.empty else df


def _features_for_row(row, alltime_acc, patch_acc):
    return team_priority_features_for_teams(
        row.get("candidate_hero_id"),
        row.get("acting_team_id"),
        row.get("opponent_team_id"),
        alltime_acc,
        patch_acc,
    )


def _add_for_table(candidates, matches, picks_bans, alltime_acc, patch_acc):
    rows = []
    candidates_by_match = {match_id: group for match_id, group in candidates.groupby("match_id", sort=False)}
    for _, match_group in matches.groupby("_sort_start_time", sort=True):
        match_ids = set(match_group["match_id"].tolist())
        for match_id in match_ids:
            part = candidates_by_match.get(match_id)
            if part is None:
                continue
            feature_part = part.apply(lambda row: pd.Series(_features_for_row(row, alltime_acc, patch_acc)), axis=1)
            rows.append(pd.concat([part.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1))
        current_matches = match_group.drop(columns=["_sort_start_time"], errors="ignore")
        current_picks = _match_slice(picks_bans, match_ids)
        _update_acc(alltime_acc, current_matches, current_picks)
        _update_acc(patch_acc, current_matches, current_picks)
    return pd.concat(rows, ignore_index=True) if rows else candidates


def _validate_output(input_df, output_df):
    if len(input_df) != len(output_df):
        raise ValueError(f"row count changed: input={len(input_df)}, output={len(output_df)}")
    if input_df["state_id"].nunique() != output_df["state_id"].nunique():
        raise ValueError("state_id count changed")
    target_sum = output_df.groupby("state_id")["target"].sum()
    bad = target_sum[target_sum != 1]
    if not bad.empty:
        raise ValueError(f"target sum must equal 1 per state_id: {bad.head().to_dict()}")
    missing = [col for col in TEAM_PRIORITY_FEATURES if col not in output_df.columns]
    if missing:
        raise ValueError(f"missing team-priority columns: {missing}")
    if output_df[TEAM_PRIORITY_FEATURES].fillna(0).abs().sum().sum() == 0:
        print("warning: all team-priority features are zero")


def add_team_priority_features(patch_label=PATCH_LABEL):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)
    matches = _sort_matches(pd.read_parquet(base_dir / "matches.parquet"))
    picks_bans = pd.read_parquet(base_dir / "picks_bans.parquet")

    for action in ["pick", "ban"]:
        path = ml_dir / f"draft_candidates_{action}_players.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing players candidate table: {path}")
        candidates = pd.read_parquet(path)
        out = _add_for_table(
            candidates,
            matches,
            picks_bans,
            _initial_alltime_acc(patch_label),
            _empty_acc(),
        )
        _validate_output(candidates, out)
        out_path = ml_dir / f"draft_candidates_{action}_players_team.parquet"
        out.to_parquet(out_path, index=False)
        print(f"draft_candidates_{action}_players_team: {out.shape} -> {out_path}")


def main(argv=None):
    args = parse_args(argv)
    return add_team_priority_features(args.patch_label)


if __name__ == "__main__":
    main()
