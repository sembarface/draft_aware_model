import argparse
import itertools

import pandas as pd
from scipy.stats import binomtest

from src.config import PATCH_LABEL, get_patch_paths


# For final strict ML methodology these aggregate interaction tables should be
# computed on train-only data or in a rolling/as-of setup to avoid leakage.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build hero interaction parquet tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--min-games", type=int, default=20, help="Minimum games for synergy/matchup EDA filters.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    return parser.parse_args(argv)


def binom_pvalue(successes, trials, p0):
    if (
        trials is None
        or successes is None
        or p0 is None
        or pd.isna(p0)
        or trials <= 0
        or p0 < 0
        or p0 > 1
    ):
        return None
    successes = int(successes)
    trials = int(trials)
    if successes > trials:
        return None
    return binomtest(successes, trials, p0, alternative="two-sided").pvalue


def _team_heroes(row, side):
    return [
        int(row[f"{side}_hero_{i}"])
        for i in range(1, 6)
        if f"{side}_hero_{i}" in row.index and pd.notna(row[f"{side}_hero_{i}"])
    ]


def _hero_maps(heroes_stats):
    hero_name = dict(zip(heroes_stats["hero_id"], heroes_stats["hero_name"]))
    hero_wr = dict(zip(heroes_stats["hero_id"], heroes_stats["winrate"]))
    return hero_name, hero_wr


def _empty_synergy():
    return pd.DataFrame(columns=[
        "hero1_id",
        "hero2_id",
        "games",
        "wins",
        "pair_winrate",
        "hero1_wr",
        "hero2_wr",
        "baseline_winrate",
        "synergy_delta",
        "delta_hero1",
        "delta_hero2",
        "pair_pick_freq",
        "p_value",
        "significant",
        "hero1_name",
        "hero2_name",
    ])


def build_hero_synergy(
    matches,
    heroes_stats,
    min_games=20,
    alpha=0.05,
    apply_min_games=True,
):
    total_matches = matches["match_id"].nunique() if not matches.empty else 0
    hero_name, hero_wr = _hero_maps(heroes_stats)
    records = []

    for _, row in matches.iterrows():
        radiant = _team_heroes(row, "radiant")
        dire = _team_heroes(row, "dire")
        radiant_win = bool(row["radiant_win"])

        for heroes, win in [(radiant, radiant_win), (dire, not radiant_win)]:
            for hero1, hero2 in itertools.combinations(sorted(heroes), 2):
                records.append((hero1, hero2, win))

    if not records:
        return _empty_synergy()

    df = pd.DataFrame(records, columns=["hero1_id", "hero2_id", "win"])
    synergy = (
        df.groupby(["hero1_id", "hero2_id"])
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )

    if apply_min_games:
        synergy = synergy[synergy["games"] >= min_games].copy()

    if synergy.empty:
        return _empty_synergy()

    synergy["pair_winrate"] = synergy["wins"] / synergy["games"]
    synergy["hero1_wr"] = synergy["hero1_id"].map(hero_wr)
    synergy["hero2_wr"] = synergy["hero2_id"].map(hero_wr)
    synergy["baseline_winrate"] = (synergy["hero1_wr"] + synergy["hero2_wr"]) / 2
    synergy["synergy_delta"] = synergy["pair_winrate"] - synergy["baseline_winrate"]
    synergy["delta_hero1"] = synergy["pair_winrate"] - synergy["hero1_wr"]
    synergy["delta_hero2"] = synergy["pair_winrate"] - synergy["hero2_wr"]
    synergy["pair_pick_freq"] = synergy["games"] / total_matches if total_matches else 0
    synergy["p_value"] = synergy.apply(
        lambda row: binom_pvalue(row["wins"], row["games"], row["baseline_winrate"]),
        axis=1,
    )
    synergy["significant"] = synergy["p_value"] < alpha
    synergy["hero1_name"] = synergy["hero1_id"].map(hero_name)
    synergy["hero2_name"] = synergy["hero2_id"].map(hero_name)

    return synergy.sort_values("synergy_delta", ascending=False).reset_index(drop=True)


def _empty_matchups():
    return pd.DataFrame(columns=[
        "hero_id",
        "vs_hero_id",
        "games",
        "wins",
        "matchup_winrate",
        "hero_winrate",
        "counter_delta",
        "p_value_win",
        "significant_win",
        "hero_name",
        "vs_hero_name",
    ])


def build_hero_matchups(
    matches,
    heroes_stats,
    min_games=20,
    alpha=0.05,
    apply_min_games=True,
):
    hero_name, hero_wr = _hero_maps(heroes_stats)
    records = []

    for _, row in matches.iterrows():
        radiant = _team_heroes(row, "radiant")
        dire = _team_heroes(row, "dire")
        radiant_win = bool(row["radiant_win"])

        for radiant_hero in radiant:
            for dire_hero in dire:
                records.append((radiant_hero, dire_hero, radiant_win))
                records.append((dire_hero, radiant_hero, not radiant_win))

    if not records:
        return _empty_matchups()

    df = pd.DataFrame(records, columns=["hero_id", "vs_hero_id", "win"])
    matchups = (
        df.groupby(["hero_id", "vs_hero_id"])
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )

    if apply_min_games:
        matchups = matchups[matchups["games"] >= min_games].copy()

    if matchups.empty:
        return _empty_matchups()

    matchups["matchup_winrate"] = matchups["wins"] / matchups["games"]
    matchups["hero_winrate"] = matchups["hero_id"].map(hero_wr)
    matchups["counter_delta"] = matchups["matchup_winrate"] - matchups["hero_winrate"]
    matchups["p_value_win"] = matchups.apply(
        lambda row: binom_pvalue(row["wins"], row["games"], row["hero_winrate"]),
        axis=1,
    )
    matchups["significant_win"] = matchups["p_value_win"] < alpha
    matchups["hero_name"] = matchups["hero_id"].map(hero_name)
    matchups["vs_hero_name"] = matchups["vs_hero_id"].map(hero_name)

    return matchups.sort_values("counter_delta", ascending=False).reset_index(drop=True)


def build_conditional_bans(picks_bans, min_games=5):
    columns = [
        "picked_hero_id",
        "banned_hero_id",
        "picked_games",
        "bans_by_picked_team",
        "bans_by_opponent_team",
        "ban_rate_by_picked_team",
        "ban_rate_by_opponent_team",
        "picked_hero_name",
        "banned_hero_name",
    ]

    if picks_bans.empty:
        return pd.DataFrame(columns=columns)

    picks = picks_bans[picks_bans["is_pick"]].copy()
    bans = picks_bans[~picks_bans["is_pick"]].copy()
    picked_games = picks.groupby("hero_id")["match_id"].nunique().to_dict()
    hero_name = (
        picks_bans[["hero_id", "hero_name"]]
        .dropna()
        .drop_duplicates("hero_id")
        .set_index("hero_id")["hero_name"]
        .to_dict()
    )

    counts = {}

    for match_id, match_picks in picks.groupby("match_id"):
        match_bans = bans[bans["match_id"] == match_id]
        if match_bans.empty:
            continue

        for _, pick in match_picks.iterrows():
            picked_hero = int(pick["hero_id"])
            pick_team = pick["team"]

            for _, ban in match_bans.iterrows():
                banned_hero = int(ban["hero_id"])
                key = (picked_hero, banned_hero)
                if key not in counts:
                    counts[key] = {
                        "bans_by_picked_team": 0,
                        "bans_by_opponent_team": 0,
                    }

                if ban["team"] == pick_team:
                    counts[key]["bans_by_picked_team"] += 1
                else:
                    counts[key]["bans_by_opponent_team"] += 1

    rows = []
    for (picked_hero, banned_hero), values in counts.items():
        games = int(picked_games.get(picked_hero, 0))
        if games < min_games:
            continue

        rows.append({
            "picked_hero_id": picked_hero,
            "banned_hero_id": banned_hero,
            "picked_games": games,
            "bans_by_picked_team": values["bans_by_picked_team"],
            "bans_by_opponent_team": values["bans_by_opponent_team"],
            "ban_rate_by_picked_team": values["bans_by_picked_team"] / games if games else 0,
            "ban_rate_by_opponent_team": values["bans_by_opponent_team"] / games if games else 0,
            "picked_hero_name": hero_name.get(picked_hero),
            "banned_hero_name": hero_name.get(banned_hero),
        })

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values("ban_rate_by_opponent_team", ascending=False)
        .reset_index(drop=True)
    )


def save_interaction_tables(patch_label=PATCH_LABEL, min_games=20, alpha=0.05):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)
    ml_dir.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(base_dir / "matches.parquet")
    heroes_stats = pd.read_parquet(base_dir / "heroes_stats.parquet")
    picks_bans = pd.read_parquet(base_dir / "picks_bans.parquet")

    synergy = build_hero_synergy(
        matches,
        heroes_stats,
        min_games=min_games,
        alpha=alpha,
        apply_min_games=False,
    )
    matchups = build_hero_matchups(
        matches,
        heroes_stats,
        min_games=min_games,
        alpha=alpha,
        apply_min_games=False,
    )
    conditional_bans = build_conditional_bans(picks_bans, min_games=5)

    synergy_path = ml_dir / "hero_synergy.parquet"
    matchups_path = ml_dir / "hero_matchups.parquet"
    bans_path = ml_dir / "hero_conditional_bans.parquet"

    synergy.to_parquet(synergy_path, index=False)
    matchups.to_parquet(matchups_path, index=False)
    conditional_bans.to_parquet(bans_path, index=False)

    print(f"hero_synergy: {synergy.shape} -> {synergy_path}")
    print(f"hero_matchups: {matchups.shape} -> {matchups_path}")
    print(f"hero_conditional_bans: {conditional_bans.shape} -> {bans_path}")
    return synergy, matchups, conditional_bans


def main(argv=None):
    args = parse_args(argv)
    return save_interaction_tables(
        patch_label=args.patch_label,
        min_games=args.min_games,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
