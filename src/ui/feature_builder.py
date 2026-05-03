import pandas as pd

from src.config import get_patch_num
from src.ui.explanations import explain_recommendation, explain_recommendation_markdown


def _as_str_id(value):
    if pd.isna(value):
        return "unknown"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def build_recommendation_rows(
    heroes,
    ref_lookup,
    unavailable_heroes,
    patch_label,
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
    player_feature_fn,
):
    available = heroes[~heroes["hero_id"].isin(set(unavailable_heroes))].copy()
    rows = []
    patch_num = get_patch_num(patch_label)

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
        row.update(player_feature_fn(hero_id))
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_model_frame(df, features, cat_features):
    for feature in features:
        if feature not in df.columns:
            df[feature] = "unknown" if feature in cat_features else 0
    for feature in cat_features:
        df[feature] = df[feature].fillna("unknown").astype(str)
    for feature in [col for col in features if col not in cat_features]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)
    return df


def attach_explanations(df, action_type):
    df["explanation"] = df.apply(lambda row: explain_recommendation(row, action_type), axis=1)
    df["explanation_markdown"] = df.apply(lambda row: explain_recommendation_markdown(row, action_type), axis=1)
    df["key_factors"] = df["explanation"]
    return df
