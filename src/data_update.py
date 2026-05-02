import argparse

import pandas as pd

from src.config import PATCH_LABEL, get_patch_paths
from src.convert_to_parquet import run_conversion
from src.ml.build_draft_candidates import build_draft_candidates
from src.ml.build_draft_events import build_draft_events
from src.ml.build_draft_states import build_draft_states
from src.ml.build_player_stats import build_player_stats
from src.parser_explorer import run_parser


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch/convert/build local Dota draft data.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--fetch", action="store_true", help="Fetch raw JSON files from OpenDota.")
    parser.add_argument("--convert", action="store_true", help="Convert raw JSON files to parquet tables.")
    parser.add_argument("--build-ml", action="store_true", help="Build draft_events/states/candidates tables.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild parquet tables from all JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Fetch only first N matches.")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Fetch JSON again even if it already exists locally.",
    )
    return parser.parse_args(argv)


def _shape(path):
    if not path.exists():
        return None
    return pd.read_parquet(path).shape


def print_summary(patch_label, include_ml=False):
    _, base_dir, match_dir, ml_dir, _ = get_patch_paths(patch_label)
    json_count = len(list(match_dir.glob("*.json"))) if match_dir.exists() else 0

    print("summary")
    print(f"json files: {json_count}")
    print(f"matches.parquet: {_shape(base_dir / 'matches.parquet')}")
    print(f"players.parquet: {_shape(base_dir / 'players.parquet')}")
    print(f"picks_bans.parquet: {_shape(base_dir / 'picks_bans.parquet')}")
    print(f"heroes_stats.parquet: {_shape(base_dir / 'heroes_stats.parquet')}")

    if include_ml:
        print(f"draft_events.parquet: {_shape(ml_dir / 'draft_events.parquet')}")
        print(f"draft_states.parquet: {_shape(ml_dir / 'draft_states.parquet')}")
        print(f"draft_candidates_pick.parquet: {_shape(ml_dir / 'draft_candidates_pick.parquet')}")
        print(f"draft_candidates_ban.parquet: {_shape(ml_dir / 'draft_candidates_ban.parquet')}")
        print(f"player_stats.parquet: {_shape(base_dir / 'player_stats.parquet')}")
        print(f"player_hero_stats.parquet: {_shape(base_dir / 'player_hero_stats.parquet')}")


def run_data_update(
    patch_label=PATCH_LABEL,
    fetch=False,
    convert=False,
    build_ml=False,
    rebuild=False,
    limit=None,
    refresh_existing=False,
):
    if fetch:
        run_parser(
            patch_label=patch_label,
            limit=limit,
            refresh_existing=refresh_existing,
        )

    if convert:
        run_conversion(
            patch_label=patch_label,
            rebuild=rebuild,
            validate=True,
        )

    if build_ml:
        build_draft_events(patch_label)
        build_draft_states(patch_label)
        build_player_stats(patch_label, all_patches=False, alltime=False)
        build_draft_candidates(patch_label)

    print_summary(patch_label, include_ml=build_ml)


def main(argv=None):
    args = parse_args(argv)
    return run_data_update(
        patch_label=args.patch_label,
        fetch=args.fetch,
        convert=args.convert,
        build_ml=args.build_ml,
        rebuild=args.rebuild,
        limit=args.limit,
        refresh_existing=args.refresh_existing,
    )


if __name__ == "__main__":
    main()
