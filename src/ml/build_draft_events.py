from pathlib import Path
import pandas as pd


PATCH = 60
BASE = Path(f"data/patch_{PATCH}")
ML_DIR = BASE / "ml"
ML_DIR.mkdir(exist_ok=True)


def get_team_info(row):
    if row["team"] == 0:
        return pd.Series({
            "acting_side": "radiant",
            "opponent_side": "dire",
            "acting_team_id": row["radiant_team_id"],
            "acting_team_name": row["radiant_team_name"],
            "opponent_team_id": row["dire_team_id"],
            "opponent_team_name": row["dire_team_name"],
            "acting_team_win": row["radiant_win"],
        })
    else:
        return pd.Series({
            "acting_side": "dire",
            "opponent_side": "radiant",
            "acting_team_id": row["dire_team_id"],
            "acting_team_name": row["dire_team_name"],
            "opponent_team_id": row["radiant_team_id"],
            "opponent_team_name": row["radiant_team_name"],
            "acting_team_win": not row["radiant_win"],
        })


def main():
    picks = pd.read_parquet(BASE / "picks_bans.parquet")
    matches = pd.read_parquet(BASE / "matches.parquet")

    cols = [
        "match_id", "start_time", "patch", "league_name",
        "radiant_win",
        "radiant_team_id", "radiant_team_name",
        "dire_team_id", "dire_team_name",
    ]

    df = picks.merge(matches[cols], on="match_id", how="left")
    info = df.apply(get_team_info, axis=1)
    df = pd.concat([df, info], axis=1)
    cnt = df.groupby("match_id").size()
    good_matches = cnt[cnt == 24].index
    df = df[df["match_id"].isin(good_matches)].copy()
    df["action_type"] = df["is_pick"].map({True: "pick", False: "ban"})
    df["state_id"] = df["match_id"].astype(str) + "_" + df["order"].astype(str)

    df = df.sort_values(["match_id", "order"])

    df.to_parquet(ML_DIR / "draft_events.parquet", index=False)


if __name__ == "__main__":
    main()