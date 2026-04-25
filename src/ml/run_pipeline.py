import argparse

from src.config import PATCH_LABEL
from src.ml.add_interaction_features import add_interaction_features
from src.ml.build_draft_candidates import build_draft_candidates
from src.ml.build_draft_events import build_draft_events
from src.ml.build_draft_states import build_draft_states
from src.ml.evaluate import evaluate
from src.ml.interaction_tables import save_interaction_tables
from src.ml.train_catboost import train_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run draft-aware ML pipeline.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument(
        "--dataset",
        choices=["base", "interactions"],
        default="base",
        help="Dataset variant to train/evaluate.",
    )
    return parser.parse_args(argv)


def run_pipeline(patch_label=PATCH_LABEL, dataset="base"):
    build_draft_events(patch_label)
    build_draft_states(patch_label)
    build_draft_candidates(patch_label)

    if dataset == "interactions":
        save_interaction_tables(patch_label)
        add_interaction_features(patch_label)

    for action in ["pick", "ban"]:
        train_model(action=action, dataset=dataset, patch_label=patch_label)
        evaluate(action=action, dataset=dataset, patch_label=patch_label)


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(args.patch_label, args.dataset)


if __name__ == "__main__":
    main()
