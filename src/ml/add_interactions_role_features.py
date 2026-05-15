import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths
from src.ml.feature_sets import ROLE_FEATURES
from src.ml.role_utils import build_role_lookup, candidate_role_features, load_hero_roles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add team-agnostic role features to interactions candidate tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    return parser.parse_args(argv)


def _state_lookup(ml_dir):
    states = pd.read_parquet(
        ml_dir / "draft_states.parquet",
        columns=["state_id", "ally_picks_before", "enemy_picks_before"],
    )
    return states.set_index("state_id").to_dict("index")


def _features_for_row(row, state_lookup, role_lookup):
    state = state_lookup.get(row["state_id"], {})
    output = candidate_role_features(
        int(row["candidate_hero_id"]),
        state.get("ally_picks_before", []),
        state.get("enemy_picks_before", []),
        role_lookup,
    )
    output["own_best_player_candidate_role_fit"] = 0.0
    output["own_mean_player_candidate_role_fit"] = 0.0
    output["opponent_best_player_candidate_role_fit"] = 0.0
    output["opponent_mean_player_candidate_role_fit"] = 0.0
    return output


def _validate_output(input_df, output_df):
    if len(input_df) != len(output_df):
        raise ValueError(f"row count changed: input={len(input_df)}, output={len(output_df)}")
    if input_df["state_id"].nunique() != output_df["state_id"].nunique():
        raise ValueError("state_id count changed")
    target_sum = output_df.groupby("state_id")["target"].sum()
    bad = target_sum[target_sum != 1]
    if not bad.empty:
        raise ValueError(f"target sum must equal 1 per state_id: {bad.head().to_dict()}")
    missing = [col for col in ROLE_FEATURES if col not in output_df.columns]
    if missing:
        raise ValueError(f"missing role columns: {missing}")
    if output_df[ROLE_FEATURES].fillna(0).abs().sum().sum() == 0:
        raise ValueError("all role features are zero")


def add_interactions_role_features(patch_label=PATCH_LABEL):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)
    state_lookup = _state_lookup(ml_dir)
    role_lookup = build_role_lookup(load_hero_roles())

    for action in ["pick", "ban"]:
        path = ml_dir / f"draft_candidates_{action}_interactions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing interactions candidate table: {path}")
        candidates = pd.read_parquet(path)
        feature_part = candidates.apply(
            lambda row: pd.Series(_features_for_row(row, state_lookup, role_lookup)),
            axis=1,
        )
        out = pd.concat([candidates.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1)
        _validate_output(candidates, out)
        out_path = ml_dir / f"draft_candidates_{action}_interactions_role.parquet"
        out.to_parquet(out_path, index=False)
        print(f"draft_candidates_{action}_interactions_role: {out.shape} -> {out_path}")


def main(argv=None):
    args = parse_args(argv)
    return add_interactions_role_features(args.patch_label)


if __name__ == "__main__":
    main()
