import argparse

import pandas as pd

from src.config import PATCH_LABEL
from src.ml.add_team_priority_features import build_full_team_priority_acc, team_priority_features_for_team


HEROES_PATH = "data/heroes.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inspect team hero priority stats for one team/hero pair.")
    parser.add_argument("--patch-label", default=PATCH_LABEL)
    parser.add_argument("--team-id", required=True, type=int)
    parser.add_argument("--hero-id", type=int)
    parser.add_argument("--hero-name")
    return parser.parse_args(argv)


def _resolve_hero(hero_id=None, hero_name=None):
    heroes = pd.read_csv(HEROES_PATH)[["id", "name"]].rename(columns={"id": "hero_id", "name": "hero_name"})
    if hero_id is not None:
        row = heroes[heroes["hero_id"].astype(int) == int(hero_id)]
    elif hero_name:
        row = heroes[heroes["hero_name"].str.lower() == hero_name.lower()]
    else:
        raise ValueError("Provide either --hero-id or --hero-name")
    if row.empty:
        raise ValueError(f"Hero not found: id={hero_id}, name={hero_name}")
    item = row.iloc[0]
    return int(item["hero_id"]), str(item["hero_name"])


def inspect_team_priority(patch_label=PATCH_LABEL, team_id=None, hero_id=None, hero_name=None):
    hero_id, hero_name = _resolve_hero(hero_id, hero_name)
    alltime_acc, patch_acc = build_full_team_priority_acc(patch_label)
    features = team_priority_features_for_team(hero_id, team_id, alltime_acc, patch_acc)
    print(f"patch_label: {patch_label}")
    print(f"team_id: {team_id}")
    print(f"hero_id: {hero_id}")
    print(f"hero_name: {hero_name}")
    for scope in ["patch", "alltime"]:
        print(f"\n{scope}:")
        for name in [
            "matches",
            "pick_count",
            "pick_rate",
            "winrate",
            "early_pick_rate",
            "avg_pick_order",
            "ban_against_count",
            "ban_against_rate",
            "contested_count",
            "contested_rate",
        ]:
            key = f"team_candidate_{name}_{scope}"
            print(f"{key}: {features.get(key)}")
    return features


def main(argv=None):
    args = parse_args(argv)
    return inspect_team_priority(args.patch_label, args.team_id, args.hero_id, args.hero_name)


if __name__ == "__main__":
    main()
