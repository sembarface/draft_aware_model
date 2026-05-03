import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRanker, Pool

from src.config import get_ml_report_dirs, get_patch_num, get_patch_paths
from src.ui.explanations import explain_recommendation, explain_recommendation_markdown


HEROES_PATH = Path("data/heroes.csv")


def _other_side(side):
    return "dire" if side == "radiant" else "radiant"


def _as_str_id(value):
    if pd.isna(value):
        return "unknown"
    try:
        return str(int(value))
    except Exception:
        return str(value)


@st.cache_data(show_spinner=False)
def load_heroes():
    heroes = pd.read_csv(HEROES_PATH)[["id", "name"]].rename(columns={"id": "hero_id", "name": "hero_name"})
    heroes["hero_id"] = heroes["hero_id"].astype(int)
    return heroes.sort_values("hero_name").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_teams(patch_label="7.41"):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    matches_path = base_dir / "matches.parquet"
    if not matches_path.exists():
        return pd.DataFrame(columns=["team_id", "team_name"])
    matches = pd.read_parquet(matches_path)
    rows = []
    for side in ["radiant", "dire"]:
        id_col = f"{side}_team_id"
        name_col = f"{side}_team_name"
        if id_col in matches.columns:
            part = matches[[id_col, name_col]].rename(columns={id_col: "team_id", name_col: "team_name"})
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["team_id", "team_name"])
    teams = pd.concat(rows, ignore_index=True).dropna(subset=["team_id"]).drop_duplicates("team_id")
    teams["team_id"] = teams["team_id"].astype(int)
    teams["team_name"] = teams["team_name"].fillna(teams["team_id"].astype(str))
    return teams.sort_values("team_name").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_players(patch_label="7.41"):
    _, base_dir, _, _, _ = get_patch_paths(patch_label)
    players_path = base_dir / "players.parquet"
    matches_path = base_dir / "matches.parquet"
    if not players_path.exists():
        return pd.DataFrame()
    players = pd.read_parquet(players_path)
    if matches_path.exists() and "start_time" not in players.columns:
        matches = pd.read_parquet(matches_path)[["match_id", "start_time"]]
        players = players.merge(matches, on="match_id", how="left")
    return players


@st.cache_data(show_spinner=False)
def load_player_stats(patch_label="7.41"):
    patch_num = get_patch_num(patch_label)
    paths = {
        "alltime_player": Path("data/alltime/player_stats.parquet"),
        "alltime_player_hero": Path("data/alltime/player_hero_stats.parquet"),
        "patch_player": Path(f"data/patch_{patch_num}/player_stats.parquet"),
        "patch_player_hero": Path(f"data/patch_{patch_num}/player_hero_stats.parquet"),
    }
    data = {}
    for key, path in paths.items():
        data[key] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return data


@st.cache_data(show_spinner=False)
def load_candidate_reference(patch_label="7.41", dataset="players"):
    _, _, _, ml_dir, _ = get_patch_paths(patch_label)
    frames = []
    for action in ["pick", "ban"]:
        suffix = "" if dataset == "base" else f"_{dataset}"
        path = ml_dir / f"draft_candidates_{action}{suffix}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["_action"] = action
            frames.append(df)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    key_cols = {"candidate_hero_id", "match_id", "state_id", "target", "order"}
    numeric_cols = [col for col in df.select_dtypes(include=["number", "bool"]).columns if col not in key_cols]
    means = df.groupby(["_action", "candidate_hero_id"])[list(numeric_cols)].mean(numeric_only=True).reset_index()
    return df, means


@st.cache_resource(show_spinner=False)
def load_model(patch_label="7.41", dataset="players", action_type="pick"):
    _, _, _, _, model_dir = get_patch_paths(patch_label)
    model_path = model_dir / f"{action_type}_{dataset}_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"missing model: {model_path}")
    model = CatBoostRanker()
    model.load_model(model_path)
    return model


@st.cache_data(show_spinner=False)
def load_feature_config(patch_label="7.41", dataset="players", action_type="pick"):
    _, _, _, _, model_dir = get_patch_paths(patch_label)
    report_dirs = get_ml_report_dirs(patch_label)
    path = report_dirs["features"] / f"{action_type}_{dataset}_features.json"
    if not path.exists():
        path = model_dir / f"{action_type}_{dataset}_features.json"
    if not path.exists():
        raise FileNotFoundError(f"missing feature config for {action_type}/{dataset}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_latest_team_roster(team_id, patch_label="7.41"):
    players = load_players(patch_label)
    if players.empty or "team_id" not in players.columns:
        return []
    team_id = int(team_id)
    part = players[players["team_id"] == team_id].copy()
    if part.empty:
        return []
    sort_cols = ["start_time", "match_id"] if "start_time" in part.columns else ["match_id"]
    latest_match = part.sort_values(sort_cols).iloc[-1]["match_id"]
    roster = part[part["match_id"] == latest_match].copy()
    rows = []
    for _, row in roster.iterrows():
        if pd.isna(row.get("account_id")):
            continue
        player_name = None
        for name_col in ["nickname", "personaname", "name", "player_name"]:
            if name_col in row.index and pd.notna(row.get(name_col)):
                player_name = str(row.get(name_col))
                break
        rows.append({
            "account_id": int(row["account_id"]),
            "player_name": player_name or str(int(row["account_id"])),
        })
    return rows[:5]


def _stats_lookup(df, keys):
    if df.empty:
        return {}
    return {tuple(row[key] for key in keys): row.to_dict() for _, row in df.iterrows()}


@st.cache_data(show_spinner=False)
def load_player_lookups(patch_label="7.41"):
    stats = load_player_stats(patch_label)
    return {
        "alltime_player": _stats_lookup(stats["alltime_player"], ["account_id"]),
        "alltime_player_hero": _stats_lookup(stats["alltime_player_hero"], ["account_id", "hero_id"]),
        "patch_player": _stats_lookup(stats["patch_player"], ["account_id"]),
        "patch_player_hero": _stats_lookup(stats["patch_player_hero"], ["account_id", "hero_id"]),
    }


def _player_summary(row):
    if not row:
        return {"matches": 0, "winrate": 0.5, "avg_kda": 0.0, "avg_gold_per_min": 0.0, "avg_xp_per_min": 0.0}
    return {
        "matches": row.get("matches", 0),
        "winrate": row.get("winrate", 0.5),
        "avg_kda": row.get("avg_kda", 0.0),
        "avg_gold_per_min": row.get("avg_gold_per_min", 0.0),
        "avg_xp_per_min": row.get("avg_xp_per_min", 0.0),
    }


def _player_hero_summary(row):
    if not row:
        return {
            "games": 0,
            "winrate": 0.5,
            "avg_kda": 0.0,
            "avg_gold_per_min": 0.0,
            "avg_xp_per_min": 0.0,
            "avg_hero_damage": 0.0,
            "avg_tower_damage": 0.0,
        }
    return {
        "games": row.get("games", 0),
        "winrate": row.get("winrate", 0.5),
        "avg_kda": row.get("avg_kda", 0.0),
        "avg_gold_per_min": row.get("avg_gold_per_min", 0.0),
        "avg_xp_per_min": row.get("avg_xp_per_min", 0.0),
        "avg_hero_damage": row.get("avg_hero_damage", 0.0),
        "avg_tower_damage": row.get("avg_tower_damage", 0.0),
    }


def _max(values, default=0.0):
    return float(np.max(values)) if values else default


def _mean(values, default=0.0):
    return float(np.mean(values)) if values else default


def _roster_feature_values(roster, hero_id, lookups, prefix, scope):
    p_lookup = lookups[f"{scope}_player"]
    ph_lookup = lookups[f"{scope}_player_hero"]
    hero_values = {name: [] for name in ["games", "winrate", "avg_kda", "avg_gold_per_min", "avg_xp_per_min", "avg_hero_damage", "avg_tower_damage"]}
    player_values = {name: [] for name in ["matches", "winrate", "avg_kda", "avg_gold_per_min", "avg_xp_per_min"]}
    best_player = None
    best_key = (-1.0, -1.0, -1.0)
    for player in roster:
        account_id = player["account_id"]
        ph = _player_hero_summary(ph_lookup.get((account_id, hero_id), {}))
        ps = _player_summary(p_lookup.get((account_id,), {}))
        for name, value in ph.items():
            hero_values[name].append(value)
        for name, value in ps.items():
            player_values[name].append(value)
        key = (
            float(ph.get("games", 0) or 0),
            float(ph.get("winrate", 0.5) or 0.5),
            float(ph.get("avg_kda", 0.0) or 0.0),
        )
        if key > best_key:
            best_key = key
            best_player = {
                "account_id": account_id,
                "player_name": player.get("player_name") or str(account_id),
                **ph,
            }
    out = {}
    for name, values in hero_values.items():
        if scope == "alltime":
            out[f"{prefix}player_hero_{name}_{scope}_mean"] = _mean(values, 0.5 if name == "winrate" else 0.0)
        out[f"{prefix}player_hero_{name}_{scope}_max"] = _max(values, 0.5 if name == "winrate" else 0.0)
    for name, values in player_values.items():
        out[f"{prefix}roster_player_{name}_{scope}_mean"] = _mean(values, 0.5 if name == "winrate" else 0.0)
    if best_player:
        out[f"{prefix}best_player_hero_account_id_{scope}"] = best_player["account_id"]
        out[f"{prefix}best_player_hero_name_{scope}"] = best_player["player_name"]
        out[f"{prefix}best_player_hero_games_{scope}"] = best_player.get("games", 0)
        out[f"{prefix}best_player_hero_winrate_{scope}"] = best_player.get("winrate", 0.5)
        out[f"{prefix}best_player_hero_avg_kda_{scope}"] = best_player.get("avg_kda", 0.0)
    return out


def _player_features_from_rosters(hero_id, own_roster, opponent_roster, lookups):
    out = {}
    for scope in ["alltime", "patch"]:
        out.update(_roster_feature_values(own_roster, hero_id, lookups, "own_", scope))
        out.update(_roster_feature_values(opponent_roster, hero_id, lookups, "opponent_", scope))
    return out


def _reference_defaults(patch_label, action_type, dataset):
    _, means = load_candidate_reference(patch_label=patch_label, dataset=dataset)
    if means.empty:
        return pd.DataFrame()
    return means[means["_action"] == action_type].copy()


def recommend(
    patch_label,
    dataset,
    action_type,
    acting_team_id,
    opponent_team_id,
    acting_team_name,
    opponent_team_name,
    acting_side,
    order,
    draft_phase,
    ally_picks_before,
    enemy_picks_before,
    ally_bans_before,
    enemy_bans_before,
    unavailable_heroes,
    top_k=10,
):
    heroes = load_heroes()
    cfg = load_feature_config(patch_label, dataset, action_type)
    features = cfg["features"]
    cat_features = cfg.get("cat_features", [])
    model = load_model(patch_label, dataset, action_type)
    ref = _reference_defaults(patch_label, action_type, dataset)
    ref_lookup = ref.set_index("candidate_hero_id").to_dict("index") if not ref.empty else {}
    available = heroes[~heroes["hero_id"].isin(set(unavailable_heroes))].copy()
    rows = []
    patch_num = get_patch_num(patch_label)
    lookups = load_player_lookups(patch_label)
    own_roster = get_latest_team_roster(acting_team_id, patch_label)
    opponent_roster = get_latest_team_roster(opponent_team_id, patch_label)

    for _, hero in available.iterrows():
        hero_id = int(hero["hero_id"])
        row = dict(ref_lookup.get(hero_id, {}))
        row.update({
            "order": order,
            "draft_phase": draft_phase,
            "action_type": action_type,
            "acting_side": acting_side,
            "acting_team_id": _as_str_id(acting_team_id),
            "opponent_team_id": _as_str_id(opponent_team_id),
            "patch": str(patch_num),
            "league_name": "manual_ui",
            "n_ally_picks_before": len(ally_picks_before),
            "n_enemy_picks_before": len(enemy_picks_before),
            "n_ally_bans_before": len(ally_bans_before),
            "n_enemy_bans_before": len(enemy_bans_before),
            "available_hero_count": len(available),
            "candidate_hero_id": hero_id,
            "candidate_hero_name": hero["hero_name"],
            "acting_team_name": acting_team_name,
            "opponent_team_name": opponent_team_name,
        })
        player_feature_values = _player_features_from_rosters(hero_id, own_roster, opponent_roster, lookups)
        row.update(player_feature_values)
        rows.append(row)

    df = pd.DataFrame(rows)
    for feature in features:
        if feature not in df.columns:
            df[feature] = "unknown" if feature in cat_features else 0
    for feature in cat_features:
        df[feature] = df[feature].fillna("unknown").astype(str)
    for feature in [col for col in features if col not in cat_features]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)

    pool = Pool(df[features], cat_features=cat_features)
    df["score"] = model.predict(pool)
    df = df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["explanation"] = df.apply(lambda row: explain_recommendation(row, action_type), axis=1)
    df["explanation_markdown"] = df.apply(lambda row: explain_recommendation_markdown(row, action_type), axis=1)
    df["key_factors"] = df["explanation"]
    output_cols = [
        "rank",
        "candidate_hero_id",
        "candidate_hero_name",
        "score",
        "explanation",
        "explanation_markdown",
        "key_factors",
    ]
    explanation_cols = [col for col in df.columns if "best_player_hero" in col]
    return df[output_cols + explanation_cols], own_roster, opponent_roster


@st.cache_data(show_spinner=False)
def recommend_cached(
    patch_label,
    dataset,
    action_type,
    acting_team_id,
    opponent_team_id,
    acting_team_name,
    opponent_team_name,
    acting_side,
    order,
    draft_phase,
    ally_picks_before,
    enemy_picks_before,
    ally_bans_before,
    enemy_bans_before,
    unavailable_heroes,
):
    return recommend(
        patch_label=patch_label,
        dataset=dataset,
        action_type=action_type,
        acting_team_id=acting_team_id,
        opponent_team_id=opponent_team_id,
        acting_team_name=acting_team_name,
        opponent_team_name=opponent_team_name,
        acting_side=acting_side,
        order=order,
        draft_phase=draft_phase,
        ally_picks_before=list(ally_picks_before),
        enemy_picks_before=list(enemy_picks_before),
        ally_bans_before=list(ally_bans_before),
        enemy_bans_before=list(enemy_bans_before),
        unavailable_heroes=set(unavailable_heroes),
        top_k=20,
    )
