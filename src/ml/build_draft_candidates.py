import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths


HEROES_PATH = "data/heroes.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build draft candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _fill_hero_name(row):
    name = row.get("candidate_hero_name")
    if pd.notna(name):
        return name
    fallback = row.get("candidate_hero_name_full")
    if pd.notna(fallback):
        return fallback
    return f"hero_{row['candidate_hero_id']}"


def _validate_targets(df, table_name):
    target_sum = df.groupby("state_id")["target"].sum()
    bad = target_sum[target_sum != 1]
    if not bad.empty:
        raise ValueError(f"{table_name}: target sum must equal 1 for every state_id. Bad examples: {bad.head().to_dict()}")


def build_draft_candidates(patch_label=PATCH_LABEL):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)

    states = pd.read_parquet(ml_dir / "draft_states.parquet")
    heroes_stats = pd.read_parquet(base_dir / "heroes_stats.parquet")

    hero_cols = [
        "hero_id",
        "hero_name",
        "matches_played",
        "winrate",
        "pick_rate",
        "ban_rate",
        "pick_or_ban_rate",
    ]

    heroes_stats = heroes_stats[hero_cols].rename(columns={
        "hero_id": "candidate_hero_id",
        "hero_name": "candidate_hero_name",
        "matches_played": "candidate_matches_played",
        "winrate": "candidate_winrate",
        "pick_rate": "candidate_pick_rate",
        "ban_rate": "candidate_ban_rate",
        "pick_or_ban_rate": "candidate_pick_or_ban_rate",
    })

    rows = []

    for _, state in states.iterrows():
        for hero_id in state["available_heroes"]:
            rows.append({
                "state_id": state["state_id"],
                "match_id": state["match_id"],
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
                "target": int(hero_id == state["chosen_hero_id"]),
            })

    candidates = pd.DataFrame(rows)
    candidates = candidates.merge(heroes_stats, on="candidate_hero_id", how="left")

    heroes = pd.read_csv(HEROES_PATH)[["id", "name"]].rename(columns={
        "id": "candidate_hero_id",
        "name": "candidate_hero_name_full",
    })

    candidates = candidates.merge(heroes, on="candidate_hero_id", how="left")
    candidates["candidate_hero_name"] = candidates.apply(_fill_hero_name, axis=1)
    candidates = candidates.drop(columns=["candidate_hero_name_full"])

    candidates["candidate_matches_played"] = candidates["candidate_matches_played"].fillna(0)
    candidates["candidate_pick_rate"] = candidates["candidate_pick_rate"].fillna(0)
    candidates["candidate_ban_rate"] = candidates["candidate_ban_rate"].fillna(0)
    candidates["candidate_pick_or_ban_rate"] = candidates["candidate_pick_or_ban_rate"].fillna(0)

    global_winrate = heroes_stats["candidate_winrate"].dropna().mean()
    if pd.isna(global_winrate):
        global_winrate = 0.5
    candidates["candidate_winrate"] = candidates["candidate_winrate"].fillna(global_winrate)

    for col in ["acting_team_id", "opponent_team_id", "league_name"]:
        candidates[col] = candidates[col].fillna("unknown").astype(str)

    if candidates[
        [
            "candidate_matches_played",
            "candidate_pick_rate",
            "candidate_ban_rate",
            "candidate_pick_or_ban_rate",
            "candidate_winrate",
            "candidate_hero_name",
        ]
    ].isna().any().any():
        raise ValueError("draft_candidates still contain missing candidate fields after fillna")

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
