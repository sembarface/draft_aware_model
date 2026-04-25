from pathlib import Path
import pandas as pd


PATCH = 60
BASE = Path(f"data/patch_{PATCH}")
ML_DIR = BASE / "ml"
HEROES_PATH = Path("data/heroes.csv")


def main():
    states = pd.read_parquet(ML_DIR / "draft_states.parquet")
    heroes_stats = pd.read_parquet(BASE / "heroes_stats.parquet")

    hero_cols = [
        "hero_id", "hero_name",
        "matches_played", "winrate",
        "pick_rate", "ban_rate", "pick_or_ban_rate"
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

    for _, s in states.iterrows():
        for hero_id in s["available_heroes"]:
            rows.append({
                "state_id": s["state_id"],
                "match_id": s["match_id"],
                "order": s["order"],
                "draft_phase": s["draft_phase"],
                "action_type": s["action_type"],

                "acting_side": s["acting_side"],
                "acting_team_id": s["acting_team_id"],
                "opponent_team_id": s["opponent_team_id"],
                "patch": s["patch"],
                "league_name": s["league_name"],
                "start_time": s["start_time"],

                "n_ally_picks_before": s["n_ally_picks_before"],
                "n_enemy_picks_before": s["n_enemy_picks_before"],
                "n_ally_bans_before": s["n_ally_bans_before"],
                "n_enemy_bans_before": s["n_enemy_bans_before"],
                "available_hero_count": s["available_hero_count"],

                "candidate_hero_id": hero_id,
                "target": int(hero_id == s["chosen_hero_id"]),
            })

    candidates = pd.DataFrame(rows)
    candidates = candidates.merge(heroes_stats, on="candidate_hero_id", how="left")

    heroes = pd.read_csv(HEROES_PATH)[["id", "name"]].rename(columns={
        "id": "candidate_hero_id",
        "name": "candidate_hero_name_full",
    })

    candidates = candidates.merge(heroes, on="candidate_hero_id", how="left")
    candidates["candidate_hero_name"] = candidates["candidate_hero_name"].fillna(
        candidates["candidate_hero_name_full"]
    )
    candidates = candidates.drop(columns=["candidate_hero_name_full"])

    candidates["candidate_matches_played"] = candidates["candidate_matches_played"].fillna(0)
    candidates["candidate_pick_rate"] = candidates["candidate_pick_rate"].fillna(0)
    candidates["candidate_ban_rate"] = candidates["candidate_ban_rate"].fillna(0)
    candidates["candidate_pick_or_ban_rate"] = candidates["candidate_pick_or_ban_rate"].fillna(0)

    global_winrate = heroes_stats["candidate_winrate"].mean()
    candidates["candidate_winrate"] = candidates["candidate_winrate"].fillna(global_winrate)

    pick = candidates[candidates["action_type"] == "pick"].copy()
    ban = candidates[candidates["action_type"] == "ban"].copy()

    pick.to_parquet(ML_DIR / "draft_candidates_pick.parquet", index=False)
    ban.to_parquet(ML_DIR / "draft_candidates_ban.parquet", index=False)


if __name__ == "__main__":
    main()
