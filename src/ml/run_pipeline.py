import argparse
import time

from src.config import PATCH_LABEL
from src.ml.add_interaction_features import add_interaction_features
from src.ml.add_interactions_role_features import add_interactions_role_features
from src.ml.add_player_features import add_player_features
from src.ml.add_role_features import add_role_features
from src.ml.add_team_priority_features import add_team_priority_features
from src.ml.analyze_metrics_by_context import analyze_metrics_by_context
from src.ml.build_draft_candidates import build_draft_candidates
from src.ml.build_draft_events import build_draft_events
from src.ml.build_draft_states import build_draft_states
from src.ml.build_player_stats import build_player_stats
from src.ml.evaluate import evaluate
from src.ml.export_reports import export_reports
from src.ml.feature_sets import DATASET_CHOICES
from src.ml.train_catboost import train_model


CHECK_DATASETS = ["base", "interactions", "interactions_role", "players_team", "players_team_role"]


class PipelineProgress:
    def __init__(self, total):
        self.total = max(1, int(total))
        self.current = 0
        self.started_at = time.time()

    def step(self, label):
        self.current += 1
        width = 24
        done = min(width, int(width * self.current / self.total))
        bar = "#" * done + "-" * (width - done)
        elapsed = time.time() - self.started_at
        print(f"\n[{bar}] {self.current}/{self.total} {label} | elapsed={elapsed:.1f}s", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run draft-aware ML pipeline.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_CHOICES, "role", "all"],
        default="base",
        help="Dataset variant to train/evaluate. Use role as alias for players_team_role, or all for the active check models.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Build tables only; do not train or evaluate.")
    return parser.parse_args(argv)


def _normalize_dataset(dataset):
    return "players_team_role" if dataset == "role" else dataset


def _datasets_to_train(dataset):
    dataset = _normalize_dataset(dataset)
    return CHECK_DATASETS if dataset == "all" else [dataset]


def _required_feature_level(datasets):
    if "players_team_role" in datasets:
        return "role"
    if "players_team" in datasets:
        return "team"
    if "interactions_role" in datasets:
        return "interactions_role"
    if "interactions" in datasets:
        return "interactions"
    return "base"


def _planned_steps(datasets, skip_train):
    level = _required_feature_level(datasets)
    total = 3
    if level in {"interactions", "interactions_role", "team", "role"}:
        total += 1
    if "interactions_role" in datasets:
        total += 1
    if level in {"team", "role"}:
        total += 3
    if level == "role":
        total += 1
    if not skip_train:
        total += len(datasets) * 2
        total += 1
        if "players_team_role" in datasets:
            total += 2
    return total


def run_pipeline(patch_label=PATCH_LABEL, dataset="base", skip_train=False):
    dataset = _normalize_dataset(dataset)
    datasets = _datasets_to_train(dataset)
    level = _required_feature_level(datasets)
    progress = PipelineProgress(_planned_steps(datasets, skip_train))

    progress.step("build draft_events")
    build_draft_events(patch_label)
    progress.step("build draft_states")
    build_draft_states(patch_label)
    progress.step("build base draft_candidates")
    build_draft_candidates(patch_label)

    if level in {"interactions", "interactions_role", "team", "role"}:
        progress.step("add interaction features")
        add_interaction_features(patch_label)
    if "interactions_role" in datasets:
        progress.step("add interactions role features")
        add_interactions_role_features(patch_label)
    if level in {"team", "role"}:
        progress.step("build player stats")
        build_player_stats(patch_label, all_patches=True, alltime=True)
        progress.step("add player features")
        add_player_features(patch_label)
        progress.step("add team priority features")
        add_team_priority_features(patch_label)
    if level == "role":
        progress.step("add role features")
        add_role_features(patch_label)

    if skip_train:
        print(f"Built tables for datasets={datasets}; skipped training/evaluation.")
        return None

    for current_dataset in datasets:
        for action in ["pick", "ban"]:
            progress.step(f"train {action} {current_dataset}")
            train_model(action=action, dataset=current_dataset, patch_label=patch_label)
            progress.step(f"evaluate {action} {current_dataset}")
            evaluate(action=action, dataset=current_dataset, patch_label=patch_label)
    progress.step("export aggregate reports")
    export_reports(patch_label)
    if "players_team_role" in datasets:
        progress.step("analyze context metrics for players_team_role")
        analyze_metrics_by_context(patch_label=patch_label, dataset="players_team_role")
        progress.step("error analysis players_team vs players_team_role")
        from src.ml.error_analysis import run_error_analysis

        run_error_analysis(patch_label=patch_label, old_dataset="players_team", new_dataset="players_team_role")


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(args.patch_label, args.dataset, args.skip_train)


if __name__ == "__main__":
    main()
