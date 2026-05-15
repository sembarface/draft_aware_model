import argparse
from collections import defaultdict

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths, get_previous_patch_labels


HEROES_PATH = "data/heroes.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build draft candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _hero_acc():
    return {
        "matches_played": defaultdict(int),
        "wins": defaultdict(int),
        "pick_count": defaultdict(int),
        "ban_count": defaultdict(int),
        "total_matches": 0,
        "total_picks": 0,
        "total_bans": 0,
    }


def _load_patch_tables(patch_label):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    required = [base_dir / "matches.parquet", base_dir / "players.parquet", base_dir / "picks_bans.parquet"]
    if not all(path.exists() for path in required):
        return None
    return {
        "matches": pd.read_parquet(required[0]),
        "players": pd.read_parquet(required[1]),
        "picks_bans": pd.read_parquet(required[2]),
    }


def _update_acc(acc, matches, players, picks_bans):
    if matches is None or matches.empty:
        return
    match_ids = set(matches["match_id"].dropna().tolist())
    acc["total_matches"] += len(match_ids)

    if players is not None and not players.empty:
        part = players[players["match_id"].isin(match_ids)].copy()
        for _, row in part.iterrows():
            if pd.isna(row.get("hero_id")):
                continue
            hero_id = int(row["hero_id"])
            acc["matches_played"][hero_id] += 1
            if bool(row.get("win")):
                acc["wins"][hero_id] += 1

    if picks_bans is not None and not picks_bans.empty:
        part = picks_bans[picks_bans["match_id"].isin(match_ids)].copy()
        for _, row in part.iterrows():
            if pd.isna(row.get("hero_id")):
                continue
            hero_id = int(row["hero_id"])
            if bool(row.get("is_pick")):
                acc["pick_count"][hero_id] += 1
                acc["total_picks"] += 1
            else:
                acc["ban_count"][hero_id] += 1
                acc["total_bans"] += 1


def _initial_acc(patch_label):
    acc = _hero_acc()
    for old_label in get_previous_patch_labels(patch_label):
        tables = _load_patch_tables(old_label)
        if tables is None:
            print(f"warning: missing raw parquet tables for previous patch {old_label}; skipped")
            continue
        _update_acc(acc, tables["matches"], tables["players"], tables["picks_bans"])
    return acc


def _hero_stats(hero_id, hero_name, acc):
    matches = acc["matches_played"].get(hero_id, 0)
    wins = acc["wins"].get(hero_id, 0)
    pick_count = acc["pick_count"].get(hero_id, 0)
    ban_count = acc["ban_count"].get(hero_id, 0)
    total_picks = acc["total_picks"]
    total_bans = acc["total_bans"]
    total_matches = acc["total_matches"]
    return {
        "candidate_hero_name": hero_name,
        "candidate_matches_played": matches,
        "candidate_winrate": wins / matches if matches else 0.5,
        "candidate_pick_rate": pick_count / total_picks if total_picks else 0.0,
        "candidate_ban_rate": ban_count / total_bans if total_bans else 0.0,
        "candidate_pick_or_ban_rate": (pick_count + ban_count) / total_matches if total_matches else 0.0,
    }


def _validate_targets(df, table_name):
    target_sum = df.groupby("state_id")["target"].sum()
    bad = target_sum[target_sum != 1]
    if not bad.empty:
        raise ValueError(f"{table_name}: target sum must equal 1 for every state_id. Bad examples: {bad.head().to_dict()}")


def _match_slice(df, match_ids):
    return df[df["match_id"].isin(match_ids)].copy() if df is not None and not df.empty else df


def _sort_matches(matches):
    matches = matches.copy()
    matches["_sort_start_time"] = pd.to_numeric(matches["start_time"], errors="coerce").fillna(float("inf"))
    return matches.sort_values(["_sort_start_time", "match_id"])


def build_draft_candidates(patch_label=PATCH_LABEL):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)

    states = pd.read_parquet(ml_dir / "draft_states.parquet").sort_values(["start_time", "match_id", "order"])
    current_matches = _sort_matches(pd.read_parquet(base_dir / "matches.parquet"))
    current_players = pd.read_parquet(base_dir / "players.parquet")
    current_picks = pd.read_parquet(base_dir / "picks_bans.parquet")

    heroes = pd.read_csv(HEROES_PATH)[["id", "name"]].rename(columns={"id": "hero_id", "name": "hero_name"})
    hero_names = dict(zip(heroes["hero_id"].astype(int), heroes["hero_name"]))

    acc = _initial_acc(patch_label)
    rows = []

    states_by_match = {match_id: group for match_id, group in states.groupby("match_id", sort=False)}
    for _, match_group in current_matches.groupby("_sort_start_time", sort=True):
        match_ids = set(match_group["match_id"].tolist())
        for match_id in match_ids:
            match_states = states_by_match.get(match_id)
            if match_states is None:
                continue
            for _, state in match_states.iterrows():
                for hero_id in state["available_heroes"]:
                    hero_id = int(hero_id)
                    row = {
                        "state_id": state["state_id"],
                        "match_id": match_id,
                        "order": state["order"],
                        "draft_phase": state["draft_phase"],
                        "action_type": state["action_type"],
                        "acting_side": state["acting_side"],
                        "acting_team_id": state["acting_team_id"],
                        "opponent_team_id": state["opponent_team_id"],
                        "patch": state["patch"],
                        "league_name": state["league_name"],
                        "start_time": state["start_time"],
                        "n_ally_picks_before": state["n_ally_picks_before"],
                        "n_enemy_picks_before": state["n_enemy_picks_before"],
                        "n_ally_bans_before": state["n_ally_bans_before"],
                        "n_enemy_bans_before": state["n_enemy_bans_before"],
                        "available_hero_count": state["available_hero_count"],
                        "candidate_hero_id": hero_id,
                        "target": int(hero_id == int(state["chosen_hero_id"])),
                    }
                    row.update(_hero_stats(hero_id, hero_names.get(hero_id, f"hero_{hero_id}"), acc))
                    rows.append(row)

        _update_acc(
            acc,
            match_group.drop(columns=["_sort_start_time"], errors="ignore"),
            _match_slice(current_players, match_ids),
            _match_slice(current_picks, match_ids),
        )

    candidates = pd.DataFrame(rows)
    for col in ["acting_team_id", "opponent_team_id", "league_name"]:
        candidates[col] = candidates[col].fillna("unknown").astype(str)

    pick = candidates[candidates["action_type"] == "pick"].copy()
    ban = candidates[candidates["action_type"] == "ban"].copy()

    _validate_targets(pick, "draft_candidates_pick")
    _validate_targets(ban, "draft_candidates_ban")

    pick_path = ml_dir / "draft_candidates_pick.parquet"
    ban_path = ml_dir / "draft_candidates_ban.parquet"
    pick.to_parquet(pick_path, index=False)
    ban.to_parquet(ban_path, index=False)

    print(f"draft_candidates_pick: {pick.shape}")
    print(f"draft_candidates_ban: {ban.shape}")
    print(f"saved: {pick_path}")
    print(f"saved: {ban_path}")
    return pick, ban


def main(argv=None):
    args = parse_args(argv)
    return build_draft_candidates(args.patch_label)


if __name__ == "__main__":
    main()
