import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths, get_previous_patch_labels


PLAYER_METRICS = [
    "kills",
    "deaths",
    "assists",
    "gold_per_min",
    "xp_per_min",
    "hero_damage",
    "tower_damage",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add player statistics to interaction candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--smooth-k", type=float, default=20.0, help="Smoothing strength for players_smooth.")
    return parser.parse_args(argv)


def _id(value):
    if pd.isna(value):
        return None
    return int(value)


def _empty_acc():
    return {
        "player": defaultdict(lambda: {"matches": 0, "wins": 0, **{col: 0.0 for col in PLAYER_METRICS}}),
        "player_hero": defaultdict(lambda: {"games": 0, "wins": 0, **{col: 0.0 for col in PLAYER_METRICS}}),
    }


def _clean_num(value):
    return 0.0 if pd.isna(value) else float(value)


def _update_acc(acc, players):
    if players is None or players.empty:
        return
    for _, row in players.iterrows():
        account_id = _id(row.get("account_id"))
        hero_id = _id(row.get("hero_id"))
        if account_id is None or hero_id is None:
            continue
        player_stats = acc["player"][account_id]
        player_stats["matches"] += 1
        if bool(row.get("win")):
            player_stats["wins"] += 1
        hero_stats = acc["player_hero"][(account_id, hero_id)]
        hero_stats["games"] += 1
        if bool(row.get("win")):
            hero_stats["wins"] += 1
        for col in PLAYER_METRICS:
            value = _clean_num(row.get(col))
            player_stats[col] += value
            hero_stats[col] += value


def _patch_players(patch_label):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    path = base_dir / "players.parquet"
    if not path.exists():
        print(f"warning: missing {path}")
        return None
    return pd.read_parquet(path)


def _initial_alltime_acc(patch_label):
    acc = _empty_acc()
    for old_label in get_previous_patch_labels(patch_label):
        players = _patch_players(old_label)
        if players is not None:
            _update_acc(acc, players)
    return acc


def _player_summary(stats):
    matches = stats.get("matches", 0)
    deaths = stats.get("deaths", 0)
    return {
        "matches": matches,
        "winrate": stats.get("wins", 0) / matches if matches else 0.5,
        "avg_kda": (stats.get("kills", 0) + stats.get("assists", 0)) / max(1, deaths),
        "avg_gold_per_min": stats.get("gold_per_min", 0) / matches if matches else 0.0,
        "avg_xp_per_min": stats.get("xp_per_min", 0) / matches if matches else 0.0,
    }


def _player_hero_summary(stats):
    games = stats.get("games", 0)
    deaths = stats.get("deaths", 0)
    return {
        "games": games,
        "winrate": stats.get("wins", 0) / games if games else 0.5,
        "avg_kda": (stats.get("kills", 0) + stats.get("assists", 0)) / max(1, deaths),
        "avg_gold_per_min": stats.get("gold_per_min", 0) / games if games else 0.0,
        "avg_xp_per_min": stats.get("xp_per_min", 0) / games if games else 0.0,
        "avg_hero_damage": stats.get("hero_damage", 0) / games if games else 0.0,
        "avg_tower_damage": stats.get("tower_damage", 0) / games if games else 0.0,
    }


def _smooth_value(games, value, prior, smooth_k):
    return float((games * value + smooth_k * prior) / (games + smooth_k)) if games + smooth_k else float(prior)


def _player_hero_smooth_summary(hero_stats, player_stats, smooth_k):
    raw = _player_hero_summary(hero_stats)
    prior = _player_summary(player_stats)
    games = raw["games"]
    return {
        "log_games": float(np.log1p(games)),
        "reliability": float(games / (games + smooth_k)) if games + smooth_k else 0.0,
        "winrate_smooth": _smooth_value(games, raw["winrate"], prior["winrate"], smooth_k),
        "avg_kda_smooth": _smooth_value(games, raw["avg_kda"], prior["avg_kda"], smooth_k),
        "avg_gold_per_min_smooth": _smooth_value(
            games, raw["avg_gold_per_min"], prior["avg_gold_per_min"], smooth_k
        ),
        "avg_xp_per_min_smooth": _smooth_value(games, raw["avg_xp_per_min"], prior["avg_xp_per_min"], smooth_k),
        "avg_hero_damage_smooth": _smooth_value(games, raw["avg_hero_damage"], 0.0, smooth_k),
        "avg_tower_damage_smooth": _smooth_value(games, raw["avg_tower_damage"], 0.0, smooth_k),
    }


def _mean(values, default=0.0):
    return float(np.mean(values)) if values else default


def _max(values, default=0.0):
    return float(np.max(values)) if values else default


def _roster_features(roster, candidate_hero, alltime_acc, patch_acc, prefix):
    output = {}
    best = None
    for scope, acc in [("alltime", alltime_acc), ("patch", patch_acc)]:
        hero_values = {name: [] for name in [
            "games",
            "winrate",
            "avg_kda",
            "avg_gold_per_min",
            "avg_xp_per_min",
            "avg_hero_damage",
            "avg_tower_damage",
        ]}
        player_values = {name: [] for name in ["matches", "winrate", "avg_kda", "avg_gold_per_min", "avg_xp_per_min"]}

        for player in roster:
            account_id = player["account_id"]
            hero_stats = _player_hero_summary(acc["player_hero"].get((account_id, candidate_hero), {}))
            player_stats = _player_summary(acc["player"].get(account_id, {}))
            for name, value in hero_stats.items():
                hero_values[name].append(value)
            for name, value in player_stats.items():
                player_values[name].append(value)

            if scope == "alltime" and hero_stats["games"] > 0:
                candidate_best = {
                    "account_id": account_id,
                    "games": hero_stats["games"],
                    "winrate": hero_stats["winrate"],
                }
                if best is None or (candidate_best["games"], candidate_best["winrate"]) > (best["games"], best["winrate"]):
                    best = candidate_best

        for name, values in hero_values.items():
            if scope == "alltime":
                output[f"{prefix}player_hero_{name}_{scope}_mean"] = _mean(values, 0.5 if name == "winrate" else 0.0)
            output[f"{prefix}player_hero_{name}_{scope}_max"] = _max(values, 0.5 if name == "winrate" else 0.0)
        for name, values in player_values.items():
            output[f"{prefix}roster_player_{name}_{scope}_mean"] = _mean(values, 0.5 if name == "winrate" else 0.0)

    output[f"{prefix}best_player_hero_account_id"] = best["account_id"] if best else pd.NA
    output[f"{prefix}best_player_hero_games_alltime"] = best["games"] if best else 0
    output[f"{prefix}best_player_hero_winrate_alltime"] = best["winrate"] if best else 0.5
    return output


def _roster_smooth_features(roster, candidate_hero, alltime_acc, patch_acc, prefix, smooth_k):
    output = {}
    best = None
    scope_metrics = {
        "alltime": [
            "log_games",
            "reliability",
            "winrate_smooth",
            "avg_kda_smooth",
            "avg_gold_per_min_smooth",
            "avg_xp_per_min_smooth",
            "avg_hero_damage_smooth",
            "avg_tower_damage_smooth",
        ],
        "patch": [
            "log_games",
            "reliability",
            "winrate_smooth",
            "avg_kda_smooth",
            "avg_gold_per_min_smooth",
            "avg_xp_per_min_smooth",
        ],
    }
    for scope, acc in [("alltime", alltime_acc), ("patch", patch_acc)]:
        hero_values = {name: [] for name in scope_metrics[scope]}
        player_values = {name: [] for name in ["matches", "winrate", "avg_kda", "avg_gold_per_min", "avg_xp_per_min"]}

        for player in roster:
            account_id = player["account_id"]
            player_stats_raw = acc["player"].get(account_id, {})
            hero_stats_raw = acc["player_hero"].get((account_id, candidate_hero), {})
            hero_stats = _player_hero_smooth_summary(hero_stats_raw, player_stats_raw, smooth_k)
            player_stats = _player_summary(player_stats_raw)

            for name, value in hero_stats.items():
                if name in hero_values:
                    hero_values[name].append(value)
            for name, value in player_stats.items():
                player_values[name].append(value)

            if scope == "alltime":
                raw = _player_hero_summary(hero_stats_raw)
                if raw["games"] > 0:
                    candidate_best = {
                        "account_id": account_id,
                        "games": raw["games"],
                        "winrate": raw["winrate"],
                    }
                    if best is None or (candidate_best["games"], candidate_best["winrate"]) > (best["games"], best["winrate"]):
                        best = candidate_best

        for name, values in hero_values.items():
            metric_name = f"{name[:-len('_smooth')]}_{scope}_smooth" if name.endswith("_smooth") else f"{name}_{scope}"
            default = 0.5 if name == "winrate_smooth" else 0.0
            output[f"{prefix}player_hero_{metric_name}_max"] = _max(values, default)
            if name in {"log_games", "reliability", "winrate_smooth"}:
                output[f"{prefix}player_hero_{metric_name}_mean"] = _mean(values, default)
        for name, values in player_values.items():
            output[f"{prefix}roster_player_{name}_{scope}_mean"] = _mean(values, 0.5 if name == "winrate" else 0.0)

    output[f"{prefix}best_player_hero_account_id"] = best["account_id"] if best else pd.NA
    output[f"{prefix}best_player_hero_games_alltime"] = best["games"] if best else 0
    output[f"{prefix}best_player_hero_winrate_alltime"] = best["winrate"] if best else 0.5
    return output


def _rosters(players):
    lookup = {}
    if players.empty:
        return lookup
    for (match_id, side), group in players.groupby(["match_id", "side"]):
        lookup[(match_id, side)] = [
            {"account_id": int(row["account_id"])}
            for _, row in group.iterrows()
            if pd.notna(row.get("account_id"))
        ]
    return lookup


def _sort_matches(matches):
    matches = matches.copy()
    matches["_sort_start_time"] = pd.to_numeric(matches["start_time"], errors="coerce").fillna(float("inf"))
    return matches.sort_values(["_sort_start_time", "match_id"])


def _features_for_row(row, rosters, alltime_acc, patch_acc, smooth=False, smooth_k=20.0):
    own_side = row["acting_side"]
    opponent_side = "dire" if own_side == "radiant" else "radiant"
    match_id = row["match_id"]
    candidate = int(row["candidate_hero_id"])
    output = {}
    if smooth:
        output.update(
            _roster_smooth_features(rosters.get((match_id, own_side), []), candidate, alltime_acc, patch_acc, "own_", smooth_k)
        )
        output.update(
            _roster_smooth_features(
                rosters.get((match_id, opponent_side), []), candidate, alltime_acc, patch_acc, "opponent_", smooth_k
            )
        )
    else:
        output.update(_roster_features(rosters.get((match_id, own_side), []), candidate, alltime_acc, patch_acc, "own_"))
        output.update(_roster_features(rosters.get((match_id, opponent_side), []), candidate, alltime_acc, patch_acc, "opponent_"))
    return output


def _add_for_table(candidates, matches, players, rosters, alltime_acc, patch_acc, smooth=False, smooth_k=20.0):
    rows = []
    candidates_by_match = {match_id: group for match_id, group in candidates.groupby("match_id", sort=False)}
    for _, match_group in matches.groupby("_sort_start_time", sort=True):
        match_ids = set(match_group["match_id"].tolist())
        for match_id in match_ids:
            part = candidates_by_match.get(match_id)
            if part is None:
                continue
            feature_part = part.apply(
                lambda row: pd.Series(_features_for_row(row, rosters, alltime_acc, patch_acc, smooth=smooth, smooth_k=smooth_k)),
                axis=1,
            )
            rows.append(pd.concat([part.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1))
        current_players = players[players["match_id"].isin(match_ids)].copy()
        _update_acc(alltime_acc, current_players)
        _update_acc(patch_acc, current_players)
    return pd.concat(rows, ignore_index=True) if rows else candidates


PLAYER_SMOOTH_REQUIRED_COLUMNS = [
    "own_player_hero_log_games_alltime_max",
    "own_player_hero_reliability_alltime_max",
    "own_player_hero_winrate_alltime_smooth_max",
    "opponent_player_hero_log_games_alltime_max",
    "opponent_player_hero_reliability_alltime_max",
    "opponent_player_hero_winrate_alltime_smooth_max",
    "own_player_hero_log_games_patch_max",
    "own_player_hero_reliability_patch_max",
    "own_player_hero_winrate_patch_smooth_max",
    "opponent_player_hero_log_games_patch_max",
    "opponent_player_hero_reliability_patch_max",
    "opponent_player_hero_winrate_patch_smooth_max",
]


def _validate_players_smooth(interactions, out):
    if len(interactions) != len(out):
        raise ValueError(f"players_smooth row count mismatch: interactions={len(interactions)}, output={len(out)}")
    if interactions["state_id"].nunique() != out["state_id"].nunique():
        raise ValueError("players_smooth state_id count differs from interactions table")
    target_sum = out.groupby("state_id")["target"].sum()
    bad = target_sum[target_sum != 1]
    if not bad.empty:
        raise ValueError(f"players_smooth target sum must equal 1 per state_id: {bad.head().to_dict()}")
    missing = [col for col in PLAYER_SMOOTH_REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"players_smooth is missing required columns: {missing}")
    smooth_sum = out[PLAYER_SMOOTH_REQUIRED_COLUMNS].fillna(0).abs().sum().sum()
    if smooth_sum == 0:
        raise ValueError("players_smooth required columns are all zero")


def add_player_features(patch_label=PATCH_LABEL, smooth_k=20.0):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)
    matches = _sort_matches(pd.read_parquet(base_dir / "matches.parquet"))
    players = pd.read_parquet(base_dir / "players.parquet")
    rosters = _rosters(players)

    for action in ["pick", "ban"]:
        path = ml_dir / f"draft_candidates_{action}_interactions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing interactions candidate table: {path}")
        candidates = pd.read_parquet(path)
        out = _add_for_table(
            candidates,
            matches,
            players,
            rosters,
            _initial_alltime_acc(patch_label),
            _empty_acc(),
        )
        out_path = ml_dir / f"draft_candidates_{action}_players.parquet"
        out.to_parquet(out_path, index=False)
        print(f"draft_candidates_{action}_players: {out.shape} -> {out_path}")

        smooth_out = _add_for_table(
            candidates,
            matches,
            players,
            rosters,
            _initial_alltime_acc(patch_label),
            _empty_acc(),
            smooth=True,
            smooth_k=smooth_k,
        )
        _validate_players_smooth(candidates, smooth_out)
        smooth_path = ml_dir / f"draft_candidates_{action}_players_smooth.parquet"
        smooth_out.to_parquet(smooth_path, index=False)
        print(f"draft_candidates_{action}_players_smooth: {smooth_out.shape} -> {smooth_path}")


def main(argv=None):
    args = parse_args(argv)
    return add_player_features(args.patch_label, smooth_k=args.smooth_k)


if __name__ == "__main__":
    main()
