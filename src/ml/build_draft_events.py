import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build enriched draft event table.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


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

    return pd.Series({
        "acting_side": "dire",
        "opponent_side": "radiant",
        "acting_team_id": row["dire_team_id"],
        "acting_team_name": row["dire_team_name"],
        "opponent_team_id": row["radiant_team_id"],
        "opponent_team_name": row["radiant_team_name"],
        "acting_team_win": None if pd.isna(row["radiant_win"]) else not row["radiant_win"],
    })


def build_draft_events(patch_label=PATCH_LABEL):
    _, base_dir, _, ml_dir, _ = get_patch_paths(patch_label)
    ml_dir.mkdir(parents=True, exist_ok=True)

    picks = pd.read_parquet(base_dir / "picks_bans.parquet")
    matches = pd.read_parquet(base_dir / "matches.parquet")

    cols = [
        "match_id",
        "start_time",
        "patch",
        "league_name",
        "radiant_win",
        "radiant_team_id",
        "radiant_team_name",
        "dire_team_id",
        "dire_team_name",
    ]

    df = picks.merge(matches[cols], on="match_id", how="left")
    team_info = df.apply(get_team_info, axis=1)
    df = pd.concat([df, team_info], axis=1)

    actions_per_match = df.groupby("match_id").size()
    complete_matches = actions_per_match[actions_per_match == 24].index
    df = df[df["match_id"].isin(complete_matches)].copy()

    df["action_type"] = df["is_pick"].map({True: "pick", False: "ban"})
    df["state_id"] = df["match_id"].astype(str) + "_" + df["order"].astype(str)
    df = df.sort_values(["match_id", "order"])

    out_path = ml_dir / "draft_events.parquet"
    df.to_parquet(out_path, index=False)

    print(f"draft_events: {df.shape}")
    print(f"matches: {df['match_id'].nunique() if not df.empty else 0}")
    print(f"saved: {out_path}")
    return df


def main(argv=None):
    args = parse_args(argv)
    return build_draft_events(args.patch_label)


if __name__ == "__main__":
    main()
