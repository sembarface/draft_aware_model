from pathlib import Path

import numpy as np
import pandas as pd


HERO_ROLES_PATH = Path("data/hero_roles.csv")

POSITION_COLS = [
    "pos1_prob",
    "pos2_prob",
    "pos3_prob",
    "pos4_prob",
    "pos5_prob",
]

ROLE_COLS = POSITION_COLS + [
    "core_prob",
    "support_prob",
    "flex_score",
]

CORE_PRIOR = 0.6
SUPPORT_PRIOR = 0.4

ROLE_PRESETS = {
    "carry": (0.90, 0.05, 0.03, 0.01, 0.01),
    "mid": (0.05, 0.85, 0.05, 0.03, 0.02),
    "offlane": (0.03, 0.05, 0.82, 0.07, 0.03),
    "soft_support": (0.02, 0.03, 0.10, 0.70, 0.15),
    "hard_support": (0.01, 0.02, 0.04, 0.25, 0.68),
    "flex_core": (0.35, 0.30, 0.25, 0.07, 0.03),
    "flex_support": (0.02, 0.05, 0.08, 0.45, 0.40),
    "roaming_flex": (0.05, 0.08, 0.12, 0.55, 0.20),
    "offlane_mid_flex": (0.03, 0.35, 0.45, 0.15, 0.02),
    "offlane_support_flex": (0.12, 0.08, 0.58, 0.17, 0.05),
    "mid_carry_flex": (0.18, 0.65, 0.12, 0.03, 0.02),
    "universal": (0.20, 0.20, 0.20, 0.20, 0.20),
}

HERO_ROLE_HINTS = {
    "Anti-Mage": "carry",
    "Axe": "offlane",
    "Bane": "hard_support",
    "Bloodseeker": "flex_core",
    "Crystal Maiden": "hard_support",
    "Drow Ranger": "carry",
    "Earthshaker": "soft_support",
    "Juggernaut": "carry",
    "Mirana": "roaming_flex",
    "Morphling": "carry",
    "Shadow Fiend": "mid",
    "Phantom Lancer": "carry",
    "Puck": "mid",
    "Pudge": "roaming_flex",
    "Razor": "flex_core",
    "Sand King": "offlane",
    "Storm Spirit": "mid",
    "Sven": "carry",
    "Tiny": "flex_core",
    "Vengeful Spirit": "flex_support",
    "Windranger": "flex_core",
    "Zeus": "mid",
    "Kunkka": "flex_core",
    "Lina": "mid",
    "Lion": "hard_support",
    "Shadow Shaman": "hard_support",
    "Slardar": "offlane",
    "Tidehunter": "offlane",
    "Witch Doctor": "hard_support",
    "Lich": "hard_support",
    "Riki": "carry",
    "Enigma": "offlane",
    "Tinker": "mid",
    "Sniper": "carry",
    "Necrophos": "flex_core",
    "Warlock": "hard_support",
    "Beastmaster": "offlane",
    "Queen of Pain": "mid",
    "Venomancer": "flex_support",
    "Faceless Void": "carry",
    "Wraith King": "carry",
    "Death Prophet": "flex_core",
    "Phantom Assassin": "carry",
    "Pugna": "flex_support",
    "Templar Assassin": "mid",
    "Viper": "flex_core",
    "Luna": "carry",
    "Dragon Knight": "flex_core",
    "Dazzle": "hard_support",
    "Clockwerk": "soft_support",
    "Leshrac": "mid",
    "Nature's Prophet": "flex_core",
    "Lifestealer": "carry",
    "Dark Seer": "offlane",
    "Clinkz": "carry",
    "Omniknight": "hard_support",
    "Enchantress": "flex_support",
    "Huskar": "mid_carry_flex",
    "Night Stalker": "offlane",
    "Broodmother": "mid",
    "Bounty Hunter": "soft_support",
    "Weaver": "carry",
    "Jakiro": "hard_support",
    "Batrider": "flex_core",
    "Chen": "hard_support",
    "Spectre": "carry",
    "Ancient Apparition": "hard_support",
    "Doom": "offlane",
    "Ursa": "carry",
    "Spirit Breaker": "soft_support",
    "Gyrocopter": "carry",
    "Alchemist": "carry",
    "Invoker": "mid",
    "Silencer": "flex_support",
    "Outworld Destroyer": "mid",
    "Lycan": "offlane",
    "Brewmaster": "offlane",
    "Shadow Demon": "hard_support",
    "Lone Druid": "carry",
    "Chaos Knight": "carry",
    "Meepo": "mid",
    "Treant Protector": "hard_support",
    "Ogre Magi": "hard_support",
    "Undying": "hard_support",
    "Rubick": "soft_support",
    "Disruptor": "hard_support",
    "Nyx Assassin": "soft_support",
    "Naga Siren": "carry",
    "Keeper of the Light": "flex_support",
    "Io": "hard_support",
    "Visage": "flex_core",
    "Slark": "carry",
    "Medusa": "carry",
    "Troll Warlord": "carry",
    "Centaur Warrunner": "offlane",
    "Magnus": "offlane",
    "Timbersaw": "offlane",
    "Bristleback": "offlane",
    "Tusk": "soft_support",
    "Skywrath Mage": "soft_support",
    "Abaddon": "flex_support",
    "Elder Titan": "hard_support",
    "Legion Commander": "offlane",
    "Techies": "soft_support",
    "Ember Spirit": "mid",
    "Earth Spirit": "soft_support",
    "Underlord": "offlane",
    "Terrorblade": "carry",
    "Phoenix": "soft_support",
    "Oracle": "hard_support",
    "Winter Wyvern": "hard_support",
    "Arc Warden": "carry",
    "Monkey King": "flex_core",
    "Dark Willow": "soft_support",
    "Pangolier": "offlane_mid_flex",
    "Grimstroke": "hard_support",
    "Hoodwink": "soft_support",
    "Void Spirit": "mid",
    "Snapfire": "soft_support",
    "Mars": "offlane",
    "Ringmaster": "soft_support",
    "Dawnbreaker": "offlane_support_flex",
    "Marci": "roaming_flex",
    "Primal Beast": "offlane",
    "Muerta": "carry",
    "Kez": "carry",
    "Largo": "universal",
}


def default_role_vector():
    return {
        "pos1_prob": 0.2,
        "pos2_prob": 0.2,
        "pos3_prob": 0.2,
        "pos4_prob": 0.2,
        "pos5_prob": 0.2,
        "core_prob": CORE_PRIOR,
        "support_prob": SUPPORT_PRIOR,
        "flex_score": 1.0,
    }


def _complete_role_row(row):
    values = {col: float(row.get(col, default_role_vector()[col])) for col in POSITION_COLS}
    total = sum(max(0.0, values[col]) for col in POSITION_COLS)
    if total <= 0:
        values.update({col: 0.2 for col in POSITION_COLS})
    else:
        values.update({col: max(0.0, values[col]) / total for col in POSITION_COLS})

    core_prob = values["pos1_prob"] + values["pos2_prob"] + values["pos3_prob"]
    support_prob = values["pos4_prob"] + values["pos5_prob"]
    values["core_prob"] = float(row.get("core_prob", core_prob))
    values["support_prob"] = float(row.get("support_prob", support_prob))
    values["flex_score"] = float(row.get("flex_score", 1.0 - max(values[col] for col in POSITION_COLS)))
    return values


def _default_roles_from_heroes(heroes_path=Path("data/heroes.csv")):
    heroes = pd.read_csv(heroes_path)
    rows = []
    for _, hero in heroes.iterrows():
        hero_name = hero["name"]
        # Main workflow builds empirical priors from local OpenDota data. If that
        # file is missing, fall back to a uniform vector rather than manual hero
        # role hints so the main pipeline does not silently use hand-authored roles.
        preset = "universal"
        pos_values = ROLE_PRESETS[preset]
        role_row = dict(zip(POSITION_COLS, pos_values))
        role_row = _complete_role_row(role_row)
        rows.append({"hero_id": int(hero["id"]), "hero_name": hero_name, **role_row})
    return pd.DataFrame(rows)


def load_hero_roles(path=HERO_ROLES_PATH):
    path = Path(path)
    if not path.exists():
        # Main workflow: python -m src.ml.build_hero_roles --patch-labels 7.36 7.37 7.38 7.39 7.40 7.41
        # This fallback keeps the UI/pipeline usable when empirical priors have not been built yet.
        path.parent.mkdir(parents=True, exist_ok=True)
        roles = _default_roles_from_heroes()
        roles.to_csv(path, index=False)
    roles = pd.read_csv(path)
    if "hero_id" not in roles.columns and "id" in roles.columns:
        roles = roles.rename(columns={"id": "hero_id"})
    if "hero_name" not in roles.columns and "name" in roles.columns:
        roles = roles.rename(columns={"name": "hero_name"})
    for col in ROLE_COLS:
        if col not in roles.columns:
            roles[col] = default_role_vector()[col]
        roles[col] = pd.to_numeric(roles[col], errors="coerce").fillna(default_role_vector()[col])
    roles["hero_id"] = pd.to_numeric(roles["hero_id"], errors="coerce").astype("Int64")
    roles = roles.dropna(subset=["hero_id"]).copy()
    roles["hero_id"] = roles["hero_id"].astype(int)
    return roles[["hero_id", "hero_name", *ROLE_COLS]]


def build_role_lookup(hero_roles):
    return {
        int(row["hero_id"]): {col: float(row[col]) for col in ROLE_COLS}
        for _, row in hero_roles.iterrows()
    }


def hero_role_vector(hero_id, role_lookup):
    try:
        return role_lookup.get(int(hero_id), default_role_vector()).copy()
    except Exception:
        return default_role_vector()


def _hero_ids(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    output = []
    for value in values:
        try:
            if not pd.isna(value):
                output.append(int(value))
        except Exception:
            continue
    return output


def role_state_from_picks(hero_ids, role_lookup, prefix):
    filled = {col: 0.0 for col in POSITION_COLS}
    core_count = 0.0
    support_count = 0.0
    for hero_id in _hero_ids(hero_ids):
        role = hero_role_vector(hero_id, role_lookup)
        for col in POSITION_COLS:
            filled[col] += role[col]
        core_count += role["core_prob"]
        support_count += role["support_prob"]
    return {
        f"{prefix}pos1_filled": filled["pos1_prob"],
        f"{prefix}pos2_filled": filled["pos2_prob"],
        f"{prefix}pos3_filled": filled["pos3_prob"],
        f"{prefix}pos4_filled": filled["pos4_prob"],
        f"{prefix}pos5_filled": filled["pos5_prob"],
        f"{prefix}core_count_soft": core_count,
        f"{prefix}support_count_soft": support_count,
    }


def _role_need(state, prefix):
    return {
        "pos1_prob": max(0.0, 1.0 - state[f"{prefix}pos1_filled"]),
        "pos2_prob": max(0.0, 1.0 - state[f"{prefix}pos2_filled"]),
        "pos3_prob": max(0.0, 1.0 - state[f"{prefix}pos3_filled"]),
        "pos4_prob": max(0.0, 1.0 - state[f"{prefix}pos4_filled"]),
        "pos5_prob": max(0.0, 1.0 - state[f"{prefix}pos5_filled"]),
        "core": max(0.0, 3.0 - state[f"{prefix}core_count_soft"]),
        "support": max(0.0, 2.0 - state[f"{prefix}support_count_soft"]),
    }


def _role_fit(candidate_role, state, prefix):
    need = _role_need(state, prefix)
    fit = sum(candidate_role[col] * need[col] for col in POSITION_COLS)
    conflict = sum(candidate_role[col] * state[f"{prefix}{col[:4]}_filled"] for col in POSITION_COLS)
    return fit, conflict, candidate_role["core_prob"] * need["core"], candidate_role["support_prob"] * need["support"]


def candidate_role_features(candidate_hero_id, ally_picks, enemy_picks, role_lookup):
    candidate_role = hero_role_vector(candidate_hero_id, role_lookup)
    ally_state = role_state_from_picks(ally_picks, role_lookup, "ally_")
    enemy_state = role_state_from_picks(enemy_picks, role_lookup, "enemy_")
    own_fit, own_conflict, own_core_fit, own_support_fit = _role_fit(candidate_role, ally_state, "ally_")
    enemy_fit, enemy_conflict, enemy_core_fit, enemy_support_fit = _role_fit(candidate_role, enemy_state, "enemy_")

    return {
        "candidate_pos1_prob": candidate_role["pos1_prob"],
        "candidate_pos2_prob": candidate_role["pos2_prob"],
        "candidate_pos3_prob": candidate_role["pos3_prob"],
        "candidate_pos4_prob": candidate_role["pos4_prob"],
        "candidate_pos5_prob": candidate_role["pos5_prob"],
        "candidate_core_prob": candidate_role["core_prob"],
        "candidate_support_prob": candidate_role["support_prob"],
        "candidate_flex_score": candidate_role["flex_score"],
        **ally_state,
        **enemy_state,
        "candidate_own_role_fit_score": own_fit,
        "candidate_own_role_conflict_score": own_conflict,
        "candidate_own_core_fit": own_core_fit,
        "candidate_own_support_fit": own_support_fit,
        "candidate_enemy_role_fit_score": enemy_fit,
        "candidate_enemy_role_conflict_score": enemy_conflict,
        "candidate_enemy_core_fit": enemy_core_fit,
        "candidate_enemy_support_fit": enemy_support_fit,
    }


def player_position_prior_from_stats(player_stats_row):
    if not player_stats_row:
        return {"core_prob": 0.0, "support_prob": 0.0}
    gpm = float(player_stats_row.get("avg_gold_per_min", 0.0) or 0.0)
    xpm = float(player_stats_row.get("avg_xp_per_min", 0.0) or 0.0)
    farm_score = gpm + 0.5 * xpm
    if farm_score >= 850:
        core_prob = 0.85
    elif farm_score >= 700:
        core_prob = 0.70
    elif farm_score >= 550:
        core_prob = 0.50
    else:
        core_prob = 0.30
    return {"core_prob": core_prob, "support_prob": 1.0 - core_prob}


def player_candidate_role_fit(player_prior, candidate_role):
    return (
        player_prior["core_prob"] * candidate_role["core_prob"]
        + player_prior["support_prob"] * candidate_role["support_prob"]
    )


def roster_candidate_role_fit(roster, candidate_hero_id, role_lookup, player_stats_lookup):
    if not roster:
        return {"best_player_candidate_role_fit": 0.0, "mean_player_candidate_role_fit": 0.0}
    candidate_role = hero_role_vector(candidate_hero_id, role_lookup)
    fits = []
    for player in roster:
        account_id = player.get("account_id") if isinstance(player, dict) else player
        try:
            account_id = int(account_id)
        except Exception:
            continue
        stats = player_stats_lookup.get(account_id) or player_stats_lookup.get((account_id,)) or {}
        fits.append(player_candidate_role_fit(player_position_prior_from_stats(stats), candidate_role))
    return {
        "best_player_candidate_role_fit": float(max(fits)) if fits else 0.0,
        "mean_player_candidate_role_fit": float(np.mean(fits)) if fits else 0.0,
    }
