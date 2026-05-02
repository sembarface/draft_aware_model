import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths, get_previous_patch_labels


PICK_INTERACTION_FEATURES = [
    "candidate_ally_synergy_mean",
    "candidate_ally_synergy_max",
    "candidate_ally_synergy_min",
    "candidate_ally_synergy_games_mean",
    "candidate_vs_enemy_counter_mean",
    "candidate_vs_enemy_counter_max",
    "candidate_vs_enemy_counter_min",
    "candidate_vs_enemy_matchup_games_mean",
    "enemy_vs_candidate_counter_mean",
]

BAN_INTERACTION_FEATURES = [
    "candidate_enemy_synergy_mean",
    "candidate_enemy_synergy_max",
    "candidate_enemy_synergy_min",
    "candidate_enemy_synergy_games_mean",
    "candidate_vs_ally_counter_mean",
    "candidate_vs_ally_counter_max",
    "candidate_vs_ally_counter_min",
    "candidate_vs_ally_matchup_games_mean",
    "ally_vs_candidate_counter_mean",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add interaction features to draft candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _as_hero_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.tolist() if pd.notna(item)]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if pd.notna(item)]
    if pd.isna(value):
        return []
    return [int(value)]


def _team_heroes(row, side):
    return [
        int(row[f"{side}_hero_{i}"])
        for i in range(1, 6)
        if f"{side}_hero_{i}" in row.index and pd.notna(row[f"{side}_hero_{i}"])
    ]


def _empty_acc():
    return {
        "hero_games": defaultdict(int),
        "hero_wins": defaultdict(int),
        "pair_games": defaultdict(int),
        "pair_wins": defaultdict(int),
        "matchup_games": defaultdict(int),
        "matchup_wins": defaultdict(int),
    }


def _update_acc(acc, matches):
    if matches is None or matches.empty:
        return
    for _, row in matches.iterrows():
        radiant = _team_heroes(row, "radiant")
        dire = _team_heroes(row, "dire")
        radiant_win = bool(row["radiant_win"])
        for heroes, win in [(radiant, radiant_win), (dire, not radiant_win)]:
            for hero_id in heroes:
                acc["hero_games"][hero_id] += 1
                if win:
                    acc["hero_wins"][hero_id] += 1
            for hero1, hero2 in itertools.combinations(sorted(heroes), 2):
                key = (hero1, hero2)
                acc["pair_games"][key] += 1
                if win:
                    acc["pair_wins"][key] += 1
        for radiant_hero in radiant:
            for dire_hero in dire:
                acc["matchup_games"][(radiant_hero, dire_hero)] += 1
                acc["matchup_games"][(dire_hero, radiant_hero)] += 1
                if radiant_win:
                    acc["matchup_wins"][(radiant_hero, dire_hero)] += 1
                else:
                    acc["matchup_wins"][(dire_hero, radiant_hero)] += 1


def _load_matches(patch_label):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    path = base_dir / "matches.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _sort_matches(matches):
    matches = matches.copy()
    matches["_sort_start_time"] = pd.to_numeric(matches["start_time"], errors="coerce").fillna(float("inf"))
    return matches.sort_values(["_sort_start_time", "match_id"])


def _initial_acc(patch_label):
    acc = _empty_acc()
    for old_label in get_previous_patch_labels(patch_label):
        matches = _load_matches(old_label)
        if matches is None:
            print(f"warning: missing matches.parquet for previous patch {old_label}; skipped")
            continue
        _update_acc(acc, matches)
    return acc


def _hero_wr(hero_id, acc):
    games = acc["hero_games"].get(hero_id, 0)
    return acc["hero_wins"].get(hero_id, 0) / games if games else 0.5


def _summary(values):
    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.max(values)), float(np.min(values))


def _synergy_stats(candidate_hero, context_heroes, acc):
    deltas = []
    games = []
    for hero in context_heroes:
        key = (min(candidate_hero, hero), max(candidate_hero, hero))
        game_count = acc["pair_games"].get(key, 0)
        wins = acc["pair_wins"].get(key, 0)
        pair_wr = wins / game_count if game_count else 0.5
        baseline = (_hero_wr(candidate_hero, acc) + _hero_wr(hero, acc)) / 2
        deltas.append(pair_wr - baseline if game_count else 0.0)
        games.append(game_count)

    mean_delta, max_delta, min_delta = _summary(deltas)
    games_mean = float(np.mean(games)) if games else 0.0
    return mean_delta, max_delta, min_delta, games_mean


def _counter_stats(hero, opponents, acc):
    deltas = []
    games = []
    for opponent in opponents:
        key = (hero, opponent)
        game_count = acc["matchup_games"].get(key, 0)
        wins = acc["matchup_wins"].get(key, 0)
        matchup_wr = wins / game_count if game_count else 0.5
        deltas.append(matchup_wr - _hero_wr(hero, acc) if game_count else 0.0)
        games.append(game_count)

    mean_delta, max_delta, min_delta = _summary(deltas)
    games_mean = float(np.mean(games)) if games else 0.0
    return mean_delta, max_delta, min_delta, games_mean


def _reverse_counter_mean(heroes, candidate_hero, acc):
    deltas = [_counter_stats(hero, [candidate_hero], acc)[0] for hero in heroes]
    return float(np.mean(deltas)) if deltas else 0.0


def _features_for_row(row, acc):
    candidate = int(row["candidate_hero_id"])
    ally_picks = _as_hero_list(row["ally_picks_before"])
    enemy_picks = _as_hero_list(row["enemy_picks_before"])

    if row["action_type"] == "pick":
        ally_syn = _synergy_stats(candidate, ally_picks, acc)
        vs_enemy = _counter_stats(candidate, enemy_picks, acc)
        return {
            "candidate_ally_synergy_mean": ally_syn[0],
            "candidate_ally_synergy_max": ally_syn[1],
            "candidate_ally_synergy_min": ally_syn[2],
            "candidate_ally_synergy_games_mean": ally_syn[3],
            "candidate_vs_enemy_counter_mean": vs_enemy[0],
            "candidate_vs_enemy_counter_max": vs_enemy[1],
            "candidate_vs_enemy_counter_min": vs_enemy[2],
            "candidate_vs_enemy_matchup_games_mean": vs_enemy[3],
            "enemy_vs_candidate_counter_mean": _reverse_counter_mean(enemy_picks, candidate, acc),
        }

    enemy_syn = _synergy_stats(candidate, enemy_picks, acc)
    vs_ally = _counter_stats(candidate, ally_picks, acc)
    return {
        "candidate_enemy_synergy_mean": enemy_syn[0],
        "candidate_enemy_synergy_max": enemy_syn[1],
        "candidate_enemy_synergy_min": enemy_syn[2],
        "candidate_enemy_synergy_games_mean": enemy_syn[3],
        "candidate_vs_ally_counter_mean": vs_ally[0],
        "candidate_vs_ally_counter_max": vs_ally[1],
        "candidate_vs_ally_counter_min": vs_ally[2],
        "candidate_vs_ally_matchup_games_mean": vs_ally[3],
        "ally_vs_candidate_counter_mean": _reverse_counter_mean(ally_picks, candidate, acc),
    }


def _add_for_table(candidates, states, matches, acc):
    state_cols = ["state_id", "ally_picks_before", "enemy_picks_before"]
    df = candidates.merge(states[state_cols], on="state_id", how="left")
    rows = []
    states_by_match = {match_id: group for match_id, group in df.groupby("match_id", sort=False)}
    for _, match_group in matches.groupby("_sort_start_time", sort=True):
        match_ids = set(match_group["match_id"].tolist())
        for match_id in match_ids:
            part = states_by_match.get(match_id)
            if part is None:
                continue
            feature_part = part.apply(lambda row: pd.Series(_features_for_row(row, acc)), axis=1)
            rows.append(pd.concat([part.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1))
        _update_acc(acc, match_group.drop(columns=["_sort_start_time"], errors="ignore"))
    if not rows:
        return df
    return pd.concat(rows, ignore_index=True).drop(columns=["ally_picks_before", "enemy_picks_before"])


def add_interaction_features(patch_label=PATCH_LABEL):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)

    states = pd.read_parquet(ml_dir / "draft_states.parquet")
    matches = _sort_matches(pd.read_parquet(base_dir / "matches.parquet"))

    pick = pd.read_parquet(ml_dir / "draft_candidates_pick.parquet")
    ban = pd.read_parquet(ml_dir / "draft_candidates_ban.parquet")

    pick_features = _add_for_table(pick, states, matches, _initial_acc(patch_label))
    ban_features = _add_for_table(ban, states, matches, _initial_acc(patch_label))

    pick_path = ml_dir / "draft_candidates_pick_interactions.parquet"
    ban_path = ml_dir / "draft_candidates_ban_interactions.parquet"
    pick_features.to_parquet(pick_path, index=False)
    ban_features.to_parquet(ban_path, index=False)

    print(f"draft_candidates_pick_interactions: {pick_features.shape} -> {pick_path}")
    print(f"draft_candidates_ban_interactions: {ban_features.shape} -> {ban_path}")
    return pick_features, ban_features


def main(argv=None):
    args = parse_args(argv)
    return add_interaction_features(args.patch_label)


if __name__ == "__main__":
    main()
