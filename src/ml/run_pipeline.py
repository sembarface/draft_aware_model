import subprocess


commands = [
    ["python", "src/ml/build_draft_events.py"],
    ["python", "src/ml/build_draft_states.py"],
    ["python", "src/ml/build_draft_candidates.py"],
    ["python", "src/ml/train_catboost.py", "--action", "pick"],
    ["python", "src/ml/evaluate.py", "--action", "pick"],
    ["python", "src/ml/train_catboost.py", "--action", "ban"],
    ["python", "src/ml/evaluate.py", "--action", "ban"],
]

for cmd in commands:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

status_command = ["python", "src/ml/make_project_status.py"]

try:
    print(" ".join(status_command))
    subprocess.run(status_command, check=True)
except subprocess.CalledProcessError as exc:
    print(f"Project status generation skipped: {exc}")
