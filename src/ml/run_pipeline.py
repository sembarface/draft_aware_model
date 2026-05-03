import argparse

from src.config import PATCH_LABEL
from src.ml.add_interaction_features import add_interaction_features
from src.ml.add_player_features import add_player_features
from src.ml.build_draft_candidates import build_draft_candidates
from src.ml.build_draft_events import build_draft_events
from src.ml.build_draft_states import build_draft_states
from src.ml.build_player_stats import build_player_stats
from src.ml.evaluate import evaluate
from src.ml.export_reports import export_reports
from src.ml.feature_sets import DATASET_CHOICES
from src.ml.train_catboost import train_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run draft-aware ML pipeline.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="base", help="Dataset variant to train/evaluate.")
    parser.add_argument("--skip-train", action="store_true", help="Build tables only; do not train or evaluate.")
    return parser.parse_args(argv)


def run_pipeline(patch_label=PATCH_LABEL, dataset="base", skip_train=False):
    build_draft_events(patch_label)
    build_draft_states(patch_label)
    build_player_stats(patch_label, all_patches=True, alltime=True)
    build_draft_candidates(patch_label)

    if dataset in {"interactions", "players", "players_smooth"}:
        add_interaction_features(patch_label)
    if dataset in {"players", "players_smooth"}:
        add_player_features(patch_label)

    if skip_train:
        print(f"Built tables for dataset={dataset}; skipped training/evaluation.")
        return None

    for action in ["pick", "ban"]:
        train_model(action=action, dataset=dataset, patch_label=patch_label)
        evaluate(action=action, dataset=dataset, patch_label=patch_label)
    export_reports(patch_label)


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(args.patch_label, args.dataset, args.skip_train)


if __name__ == "__main__":
    main()
