from pathlib import Path
import pandas as pd


PATCH = 60
BASE = Path(f"data/patch_{PATCH}")
ML_DIR = BASE / "ml"


def phase(order):
    if order <= 8:
        return 1
    if order <= 16:
        return 2
    return 3


def main():
    events = pd.read_parquet(ML_DIR / "draft_events.parquet")
    heroes = pd.read_csv("data/heroes.csv")

    all_heroes = set(heroes["id"])
    rows = []

    for match_id, g in events.groupby("match_id"):
        g = g.sort_values("order")

        used = set()

        team_data = {}

        for _, row in g.iterrows():
            acting = row["acting_team_id"]
            opponent = row["opponent_team_id"]

            if acting not in team_data:
                team_data[acting] = {"picks": [], "bans": []}
            if opponent not in team_data:
                team_data[opponent] = {"picks": [], "bans": []}

            ally_picks = team_data[acting]["picks"].copy()
            ally_bans = team_data[acting]["bans"].copy()
            enemy_picks = team_data[opponent]["picks"].copy()
            enemy_bans = team_data[opponent]["bans"].copy()

            available = sorted(list(all_heroes - used))

            rows.append({
                "state_id": row["state_id"],
                "match_id": match_id,
                "order": row["order"],
                "draft_phase": phase(row["order"]),
                "action_type": row["action_type"],
                "is_pick": row["is_pick"],
                "chosen_hero_id": row["hero_id"],
                "chosen_hero_name": row["hero_name"],

                "acting_side": row["acting_side"],
                "acting_team_id": row["acting_team_id"],
                "acting_team_name": row["acting_team_name"],
                "opponent_team_id": row["opponent_team_id"],
                "opponent_team_name": row["opponent_team_name"],

                "patch": row["patch"],
                "league_name": row["league_name"],
                "start_time": row["start_time"],
                "acting_team_win": row["acting_team_win"],

                "ally_picks_before": ally_picks,
                "enemy_picks_before": enemy_picks,
                "ally_bans_before": ally_bans,
                "enemy_bans_before": enemy_bans,

                "n_ally_picks_before": len(ally_picks),
                "n_enemy_picks_before": len(enemy_picks),
                "n_ally_bans_before": len(ally_bans),
                "n_enemy_bans_before": len(enemy_bans),

                "available_heroes": available,
                "available_hero_count": len(available),
            })

            hero = row["hero_id"]
            used.add(hero)

            if row["is_pick"]:
                team_data[acting]["picks"].append(hero)
            else:
                team_data[acting]["bans"].append(hero)

    states = pd.DataFrame(rows)
    states.to_parquet(ML_DIR / "draft_states.parquet", index=False)


if __name__ == "__main__":
    main()