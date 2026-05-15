import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths
from src.ml.add_player_features import _empty_acc, _initial_alltime_acc, _rosters, _sort_matches, _update_acc
from src.ml.feature_sets import ROLE_FEATURES
from src.ml.role_utils import (
    build_role_lookup,
    candidate_role_features,
    load_hero_roles,
    roster_candidate_role_fit,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add role-aware features to players_team candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _state_lookup(ml_dir):
    states = pd.read_parquet(ml_dir / "draft_states.parquet", columns=["state_id", "ally_picks_before", "enemy_picks_before"])
    return states.set_index("state_id").to_dict("index")


def _player_stats_lookup(acc, roster):
    lookup = {}
    for player in roster:
        account_id = player.get("account_id")
        if account_id is None:
            continue
        stats = acc["player"].get(account_id, {})
        matches = stats.get("matches", 0)
        deaths = stats.get("deaths", 0)
        lookup[account_id] = {
            "matches": matches,
            "winrate": stats.get("wins", 0) / matches if matches else 0.5,
            "avg_kda": (stats.get("kills", 0) + stats.get("assists", 0)) / max(1, deaths),
            "avg_gold_per_min": stats.get("gold_per_min", 0) / matches if matches else 0.0,
            "avg_xp_per_min": stats.get("xp_per_min", 0) / matches if matches else 0.0,
        }
    return lookup


def _role_features_for_row(row, state_lookup, rosters, role_lookup, player_acc, fit_cache):
    state = state_lookup.get(row["state_id"], {})
    ally_picks = state.get("ally_picks_before", [])
    enemy_picks = state.get("enemy_picks_before", [])
    candidate = int(row["candidate_hero_id"])
    output = candidate_role_features(candidate, ally_picks, enemy_picks, role_lookup)

    own_side = row["acting_side"]
    opponent_side = "dire" if own_side == "radiant" else "radiant"
    match_id = row["match_id"]
    own_key = (match_id, own_side, candidate)
    opponent_key = (match_id, opponent_side, candidate)
    if own_key not in fit_cache:
        own_roster = rosters.get((match_id, own_side), [])
        fit_cache[own_key] = roster_candidate_role_fit(
            own_roster,
            candidate,
            role_lookup,
            _player_stats_lookup(player_acc, own_roster),
        )
    if opponent_key not in fit_cache:
        opponent_roster = rosters.get((match_id, opponent_side), [])
        fit_cache[opponent_key] = roster_candidate_role_fit(
            opponent_roster,
            candidate,
            role_lookup,
            _player_stats_lookup(player_acc, opponent_roster),
        )
    own_fit = fit_cache[own_key]
    opponent_fit = fit_cache[opponent_key]
    output["own_best_player_candidate_role_fit"] = own_fit["best_player_candidate_role_fit"]
    output["own_mean_player_candidate_role_fit"] = own_fit["mean_player_candidate_role_fit"]
    output["opponent_best_player_candidate_role_fit"] = opponent_fit["best_player_candidate_role_fit"]
    output["opponent_mean_player_candidate_role_fit"] = opponent_fit["mean_player_candidate_role_fit"]
    return output


def _add_for_table(candidates, matches, players, rosters, state_lookup, role_lookup, alltime_acc, patch_acc):
    rows = []
    fit_cache = {}
    candidates_by_match = {match_id: group for match_id, group in candidates.groupby("match_id", sort=False)}
    for _, match_group in matches.groupby("_sort_start_time", sort=True):
        match_ids = set(match_group["match_id"].tolist())
        for match_id in match_ids:
            part = candidates_by_match.get(match_id)
            if part is None:
                continue
            feature_part = part.apply(
                lambda row: pd.Series(_role_features_for_row(row, state_lookup, rosters, role_lookup, alltime_acc, fit_cache)),
                axis=1,
            )
            rows.append(pd.concat([part.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1))
        current_players = players[players["match_id"].isin(match_ids)].copy()
        _update_acc(alltime_acc, current_players)
        _update_acc(patch_acc, current_players)
        fit_cache.clear()
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
    missing = [col for col in ROLE_FEATURES if col not in output_df.columns]
    if missing:
        raise ValueError(f"missing role columns: {missing}")
    if output_df[ROLE_FEATURES].fillna(0).abs().sum().sum() == 0:
        raise ValueError("all role features are zero")


def add_role_features(patch_label=PATCH_LABEL):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)
    matches = _sort_matches(pd.read_parquet(base_dir / "matches.parquet"))
    players = pd.read_parquet(base_dir / "players.parquet")
    rosters = _rosters(players)
    state_lookup = _state_lookup(ml_dir)
    role_lookup = build_role_lookup(load_hero_roles())

    for action in ["pick", "ban"]:
        path = ml_dir / f"draft_candidates_{action}_players_team.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing players_team candidate table: {path}")
        candidates = pd.read_parquet(path)
        out = _add_for_table(
            candidates,
            matches,
            players,
            rosters,
            state_lookup,
            role_lookup,
            _initial_alltime_acc(patch_label),
            _empty_acc(),
        )
        _validate_output(candidates, out)
        out_path = ml_dir / f"draft_candidates_{action}_players_team_role.parquet"
        out.to_parquet(out_path, index=False)
        print(f"draft_candidates_{action}_players_team_role: {out.shape} -> {out_path}")


def main(argv=None):
    args = parse_args(argv)
    return add_role_features(args.patch_label)


if __name__ == "__main__":
    main()
