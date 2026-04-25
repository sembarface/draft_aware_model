from pathlib import Path


PATCH_LABEL = "7.41"

PATCH_MAP = {
    "7.39": 58,
    "7.40": 59,
    "7.41": 60,
}

PATCH_NUM = PATCH_MAP[PATCH_LABEL]

BASE_DIR = Path(f"data/patch_{PATCH_NUM}")
MATCH_DIR = BASE_DIR / "matches"
ML_DIR = BASE_DIR / "ml"
MODEL_DIR = Path(f"models/patch_{PATCH_NUM}")

TEAM_IDS = [
    7119388,  # Team Spirit
    9247354,  # Team Falcons
    8291895,  # Tundra
    9572001,  # PARIVISION
    2163,     # Team Liquid
    8255888,  # BetBoom Team
    9823272,  # Yandex
    2586976,  # OG
    9338413,  # MOUZ
    8261500,  # Xtreme Gaming
    36,       # NaVi
    8599101,  # Gaimin Gladiators
    9640842,  # Tidebound
    9467224,  # Aurora
]

REQUEST_DELAY = 0.35
REQUEST_TIMEOUT = 20


def get_patch_num(patch_label=None):
    label = patch_label or PATCH_LABEL
    if label not in PATCH_MAP:
        known = ", ".join(sorted(PATCH_MAP))
        raise ValueError(f"Unknown patch label '{label}'. Known patch labels: {known}")
    return PATCH_MAP[label]


def get_patch_paths(patch_label=None):
    patch_num = get_patch_num(patch_label)
    base_dir = Path(f"data/patch_{patch_num}")
    match_dir = base_dir / "matches"
    ml_dir = base_dir / "ml"
    model_dir = Path(f"models/patch_{patch_num}")
    return patch_num, base_dir, match_dir, ml_dir, model_dir
