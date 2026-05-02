import argparse
from pathlib import Path

import pandas as pd

from src.config import PATCH_LABEL, PATCH_MAP, get_patch_num, get_patch_paths


HEROES_PATH = Path("data/heroes.csv")

PLAYER_METRICS = [
    "kills",
    "deaths",
    "assists",
    "gold_per_min",
    "xp_per_min",
    "last_hits",
    "denies",
    "net_worth",
    "level",
    "hero_damage",
    "tower_damage",
    "hero_healing",
    "teamfight_participation",
    "obs_placed",
    "sen_placed",
    "stuns",
    "camps_stacked",
    "rune_pickups",
    "firstblood_claimed",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build player and player-hero statistics.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--all-patches", action="store_true", help="Build stats for every available configured patch.")
    parser.add_argument("--alltime", action="store_true", help="Build data/alltime player stats.")
    return parser.parse_args(argv)


def _available_metrics(players):
    return [col for col in PLAYER_METRICS if col in players.columns]


def _clean_players(players):
    if players.empty:
        return players.copy()
    df = players.copy()
    df = df[pd.notna(df["account_id"])].copy()
    df["account_id"] = df["account_id"].astype("int64")
    if "hero_id" in df.columns:
        df["hero_id"] = pd.to_numeric(df["hero_id"], errors="coerce")
    for col in _available_metrics(df) + ["win"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _kda(group):
    deaths = group["deaths"].sum() if "deaths" in group else 0
    kills = group["kills"].sum() if "kills" in group else 0
    assists = group["assists"].sum() if "assists" in group else 0
    return float((kills + assists) / max(1, deaths))


def _aggregate(players, group_cols, count_name):
    players = _clean_players(players)
    if players.empty:
        return pd.DataFrame(columns=group_cols + [count_name, "wins", "winrate", "avg_kda"])

    metrics = _available_metrics(players)
    agg_spec = {
        count_name: ("match_id", "nunique"),
        "wins": ("win", "sum"),
    }
    for col in metrics:
        agg_spec[f"avg_{col}"] = (col, "mean")

    stats = players.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    stats["winrate"] = stats["wins"] / stats[count_name].where(stats[count_name] > 0, 1)
    kda = players.groupby(group_cols, dropna=False).apply(_kda).rename("avg_kda").reset_index()
    stats = stats.merge(kda, on=group_cols, how="left")
    return stats


def build_player_stats_for_frame(players):
    return _aggregate(players, ["account_id"], "matches")


def build_player_hero_stats_for_frame(players):
    stats = _aggregate(players, ["account_id", "hero_id"], "games")
    if stats.empty or "hero_id" not in stats.columns:
        return stats
    heroes = pd.read_csv(HEROES_PATH)[["id", "name"]].rename(columns={"id": "hero_id", "name": "hero_name"})
    stats["hero_id"] = pd.to_numeric(stats["hero_id"], errors="coerce")
    return stats.merge(heroes, on="hero_id", how="left")


def _patch_players(patch_label):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    path = base_dir / "players.parquet"
    if not path.exists():
        print(f"warning: missing {path}")
        return None
    return pd.read_parquet(path)


def build_patch_player_stats(patch_label=PATCH_LABEL):
    players = _patch_players(patch_label)
    if players is None:
        return None, None
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    player_stats = build_player_stats_for_frame(players)
    player_hero_stats = build_player_hero_stats_for_frame(players)
    player_stats.to_parquet(base_dir / "player_stats.parquet", index=False)
    player_hero_stats.to_parquet(base_dir / "player_hero_stats.parquet", index=False)
    print(f"saved: {base_dir / 'player_stats.parquet'} {player_stats.shape}")
    print(f"saved: {base_dir / 'player_hero_stats.parquet'} {player_hero_stats.shape}")
    return player_stats, player_hero_stats


def build_alltime_player_stats():
    frames = []
    for label in PATCH_MAP:
        players = _patch_players(label)
        if players is not None:
            frames.append(players)
    if not frames:
        raise FileNotFoundError("No players.parquet files found for configured patches")
    all_players = pd.concat(frames, ignore_index=True)
    out_dir = Path("data/alltime")
    out_dir.mkdir(parents=True, exist_ok=True)
    player_stats = build_player_stats_for_frame(all_players)
    player_hero_stats = build_player_hero_stats_for_frame(all_players)
    player_stats.to_parquet(out_dir / "player_stats.parquet", index=False)
    player_hero_stats.to_parquet(out_dir / "player_hero_stats.parquet", index=False)
    print(f"saved: {out_dir / 'player_stats.parquet'} {player_stats.shape}")
    print(f"saved: {out_dir / 'player_hero_stats.parquet'} {player_hero_stats.shape}")
    return player_stats, player_hero_stats


def build_player_stats(patch_label=PATCH_LABEL, all_patches=False, alltime=False):
    if all_patches:
        for label in PATCH_MAP:
            build_patch_player_stats(label)
    else:
        _ = get_patch_num(patch_label)
        build_patch_player_stats(patch_label)
    if alltime or all_patches:
        build_alltime_player_stats()


def main(argv=None):
    args = parse_args(argv)
    return build_player_stats(args.patch_label, all_patches=args.all_patches, alltime=args.alltime)


if __name__ == "__main__":
    main()
