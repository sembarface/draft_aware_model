import argparse
from pathlib import Path


GENERATED_PATTERNS = [
    "data/patch_*/ml/*.parquet",
    "data/patch_*/player_stats.parquet",
    "data/patch_*/player_hero_stats.parquet",
    "data/alltime/player_stats.parquet",
    "data/alltime/player_hero_stats.parquet",
    "models/patch_*/*_features.json",
    "models/patch_*/*_metrics.json",
    "models/patch_*/*_feature_importance.csv",
    "models/patch_*/*.csv",
    "models/patch_*/*.md",
    "reports/metrics/patch_*/*",
    "reports/errors/patch_*/*",
    "reports/recommendations/patch_*/*",
]

KEEP_NAMES = {
    "matches.parquet",
    "players.parquet",
    "picks_bans.parquet",
    "heroes_stats.parquet",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dry-run or remove generated parquet/model artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Default is dry-run.")
    return parser.parse_args(argv)


def find_generated_files():
    files = []
    for pattern in GENERATED_PATTERNS:
        for path in Path(".").glob(pattern):
            if path.is_file() and path.name not in KEEP_NAMES:
                files.append(path)
    return sorted(set(files))


def cleanup_tables(apply=False):
    files = find_generated_files()
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"cleanup generated tables ({mode})")
    print("Raw JSON, heroes.csv and raw parquet tables matches/players/picks_bans/heroes_stats are not removed.")
    if not files:
        print("No generated files found.")
        return files
    for path in files:
        print(path)
        if apply:
            path.unlink()
    print(f"files: {len(files)}")
    return files


def main(argv=None):
    args = parse_args(argv)
    return cleanup_tables(apply=args.apply)


if __name__ == "__main__":
    main()
