import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths


HEROES_PATH = "data/heroes.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build draft state table.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def phase(order):
    if order <= 8:
        return 1
    if order <= 16:
        return 2
    return 3


def _hero_id(value):
    if pd.isna(value):
        return None
    return int(value)


def build_draft_states(patch_label=PATCH_LABEL):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)

    events = pd.read_parquet(ml_dir / "draft_events.parquet")
    heroes = pd.read_csv(HEROES_PATH)
    all_heroes = set(heroes["id"].astype(int))

    rows = []
    validation_errors = []

    for match_id, group in events.groupby("match_id"):
        group = group.sort_values("order")
        if len(group) != 24:
            validation_errors.append(f"match_id={match_id}: expected 24 events, got {len(group)}")
            continue

        used = set()
        team_data = {
            "radiant": {"picks": [], "bans": []},
            "dire": {"picks": [], "bans": []},
        }

        for _, row in group.iterrows():
            acting_key = row["acting_side"]
            opponent_key = row["opponent_side"]
            hero = _hero_id(row["hero_id"])

            if acting_key not in team_data or opponent_key not in team_data:
                validation_errors.append(
                    f"match_id={match_id}, order={row['order']}: invalid sides "
                    f"{acting_key}/{opponent_key}"
                )
                continue

            ally_picks = team_data[acting_key]["picks"].copy()
            ally_bans = team_data[acting_key]["bans"].copy()
            enemy_picks = team_data[opponent_key]["picks"].copy()
            enemy_bans = team_data[opponent_key]["bans"].copy()
            available = sorted(all_heroes - used)

            if hero is None:
                validation_errors.append(f"match_id={match_id}, order={row['order']}: missing hero_id")
            elif hero not in available:
                validation_errors.append(
                    f"match_id={match_id}, order={row['order']}: chosen hero {hero} "
                    "is not available before action"
                )

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

            if hero is not None:
                used.add(hero)
                if row["is_pick"]:
                    team_data[acting_key]["picks"].append(hero)
                else:
                    team_data[acting_key]["bans"].append(hero)

                if hero in (all_heroes - used):
                    validation_errors.append(
                        f"match_id={match_id}, order={row['order']}: chosen hero {hero} "
                        "is still available after action"
                    )

    if validation_errors:
        preview = "\n".join(validation_errors[:10])
        raise ValueError(f"Draft state validation failed:\n{preview}")

    states = pd.DataFrame(rows)
    counts = states.groupby("match_id").size()
    bad_counts = counts[counts != 24]
    if not bad_counts.empty:
        raise ValueError(f"Some matches do not have 24 states: {bad_counts.head().to_dict()}")

    out_path = ml_dir / "draft_states.parquet"
    states.to_parquet(out_path, index=False)

    print(f"draft_states: {states.shape}")
    print(f"matches: {states['match_id'].nunique() if not states.empty else 0}")
    print(f"saved: {out_path}")
    return states


def main(argv=None):
    args = parse_args(argv)
    return build_draft_states(args.patch_label)


if __name__ == "__main__":
    main()
