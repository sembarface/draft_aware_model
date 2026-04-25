import subprocess


commands = [
    "python src/ml/build_draft_events.py",
    "python src/ml/build_draft_states.py",
    "python src/ml/build_draft_candidates.py",
    "python src/ml/train_catboost.py",
    "python src/ml/evaluate.py",
]

for cmd in commands:
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)