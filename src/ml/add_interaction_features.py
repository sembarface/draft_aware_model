import argparse

import numpy as np
import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths


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


def _build_synergy_lookup(synergy):
    lookup = {}
    for _, row in synergy.iterrows():
        hero1 = int(row["hero1_id"])
        hero2 = int(row["hero2_id"])
        key = (min(hero1, hero2), max(hero1, hero2))
        lookup[key] = (
            float(row["synergy_delta"]) if pd.notna(row["synergy_delta"]) else 0.0,
            float(row["games"]) if pd.notna(row["games"]) else 0.0,
        )
    return lookup


def _build_matchup_lookup(matchups):
    lookup = {}
    for _, row in matchups.iterrows():
        key = (int(row["hero_id"]), int(row["vs_hero_id"]))
        lookup[key] = (
            float(row["counter_delta"]) if pd.notna(row["counter_delta"]) else 0.0,
            float(row["games"]) if pd.notna(row["games"]) else 0.0,
        )
    return lookup


def _summary(values):
    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.max(values)), float(np.min(values))


def _synergy_stats(candidate_hero, context_heroes, synergy_lookup):
    deltas = []
    games = []
    for hero in context_heroes:
        key = (min(candidate_hero, hero), max(candidate_hero, hero))
        delta, game_count = synergy_lookup.get(key, (0.0, 0.0))
        deltas.append(delta)
        games.append(game_count)

    mean_delta, max_delta, min_delta = _summary(deltas)
    games_mean = float(np.mean(games)) if games else 0.0
    return mean_delta, max_delta, min_delta, games_mean


def _counter_stats(hero, opponents, matchup_lookup):
    deltas = []
    games = []
    for opponent in opponents:
        delta, game_count = matchup_lookup.get((hero, opponent), (0.0, 0.0))
        deltas.append(delta)
        games.append(game_count)

    mean_delta, max_delta, min_delta = _summary(deltas)
    games_mean = float(np.mean(games)) if games else 0.0
    return mean_delta, max_delta, min_delta, games_mean


def _reverse_counter_mean(heroes, candidate_hero, matchup_lookup):
    deltas = [
        matchup_lookup.get((hero, candidate_hero), (0.0, 0.0))[0]
        for hero in heroes
    ]
    return float(np.mean(deltas)) if deltas else 0.0


def _pick_features(row, synergy_lookup, matchup_lookup):
    candidate = int(row["candidate_hero_id"])
    ally_picks = _as_hero_list(row["ally_picks_before"])
    enemy_picks = _as_hero_list(row["enemy_picks_before"])

    ally_syn = _synergy_stats(candidate, ally_picks, synergy_lookup)
    vs_enemy = _counter_stats(candidate, enemy_picks, matchup_lookup)
    enemy_vs_candidate = _reverse_counter_mean(enemy_picks, candidate, matchup_lookup)

    return pd.Series({
        "candidate_ally_synergy_mean": ally_syn[0],
        "candidate_ally_synergy_max": ally_syn[1],
        "candidate_ally_synergy_min": ally_syn[2],
        "candidate_ally_synergy_games_mean": ally_syn[3],
        "candidate_vs_enemy_counter_mean": vs_enemy[0],
        "candidate_vs_enemy_counter_max": vs_enemy[1],
        "candidate_vs_enemy_counter_min": vs_enemy[2],
        "candidate_vs_enemy_matchup_games_mean": vs_enemy[3],
        "enemy_vs_candidate_counter_mean": enemy_vs_candidate,
    })


def _ban_features(row, synergy_lookup, matchup_lookup):
    candidate = int(row["candidate_hero_id"])
    ally_picks = _as_hero_list(row["ally_picks_before"])
    enemy_picks = _as_hero_list(row["enemy_picks_before"])

    enemy_syn = _synergy_stats(candidate, enemy_picks, synergy_lookup)
    vs_ally = _counter_stats(candidate, ally_picks, matchup_lookup)
    ally_vs_candidate = _reverse_counter_mean(ally_picks, candidate, matchup_lookup)

    return pd.Series({
        "candidate_enemy_synergy_mean": enemy_syn[0],
        "candidate_enemy_synergy_max": enemy_syn[1],
        "candidate_enemy_synergy_min": enemy_syn[2],
        "candidate_enemy_synergy_games_mean": enemy_syn[3],
        "candidate_vs_ally_counter_mean": vs_ally[0],
        "candidate_vs_ally_counter_max": vs_ally[1],
        "candidate_vs_ally_counter_min": vs_ally[2],
        "candidate_vs_ally_matchup_games_mean": vs_ally[3],
        "ally_vs_candidate_counter_mean": ally_vs_candidate,
    })


def _add_features_for_action(candidates, states, synergy_lookup, matchup_lookup, action):
    state_cols = ["state_id", "ally_picks_before", "enemy_picks_before"]
    df = candidates.merge(states[state_cols], on="state_id", how="left")

    if action == "pick":
        feature_df = df.apply(_pick_features, axis=1, args=(synergy_lookup, matchup_lookup))
    elif action == "ban":
        feature_df = df.apply(_ban_features, axis=1, args=(synergy_lookup, matchup_lookup))
    else:
        raise ValueError(f"Unknown action: {action}")

    df = pd.concat([df, feature_df], axis=1)
    return df.drop(columns=["ally_picks_before", "enemy_picks_before"])


def add_interaction_features(patch_label=PATCH_LABEL):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)

    states = pd.read_parquet(ml_dir / "draft_states.parquet")
    synergy = pd.read_parquet(ml_dir / "hero_synergy.parquet")
    matchups = pd.read_parquet(ml_dir / "hero_matchups.parquet")

    synergy_lookup = _build_synergy_lookup(synergy)
    matchup_lookup = _build_matchup_lookup(matchups)

    pick = pd.read_parquet(ml_dir / "draft_candidates_pick.parquet")
    ban = pd.read_parquet(ml_dir / "draft_candidates_ban.parquet")

    pick_features = _add_features_for_action(pick, states, synergy_lookup, matchup_lookup, "pick")
    ban_features = _add_features_for_action(ban, states, synergy_lookup, matchup_lookup, "ban")

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
