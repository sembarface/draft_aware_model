import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PATCH_LABEL
from src.ui.draft_order import get_draft_order, get_empty_draft_table
from src.ui.recommender import load_heroes, load_teams, recommend


st.set_page_config(page_title="Dota Draft Recommender", layout="wide")


def _team_option(row):
    return {"team_id": int(row["team_id"]), "team_name": str(row["team_name"])}


def _team_label(team):
    return f"{team['team_name']} ({team['team_id']})"


def _other_side(side):
    return "dire" if side == "radiant" else "radiant"


def _hero_label(row):
    return f"{row['hero_name']} ({row['hero_id']})"


def _hero_label_with_rank(row):
    if row.get("recommendation_rank"):
        return f"#{int(row['recommendation_rank'])} {row['hero_name']} ({row['hero_id']})"
    return _hero_label(row)


def _draft_signature(patch_label, dataset, own_team, opponent_team, own_side, first_pick):
    return (patch_label, dataset, own_team["team_id"], opponent_team["team_id"], own_side, first_pick)


def _reset_draft():
    st.session_state["draft_actions"] = []


def _current_order(first_pick):
    actions = st.session_state.get("draft_actions", [])
    order = get_draft_order(first_pick)
    if len(actions) >= len(order):
        return None
    return order[len(actions)]


def _team_for_role(role, own_team, opponent_team):
    return own_team if role == "own" else opponent_team


def _actions_by_role(role, action_type):
    return [
        action["hero_name"]
        for action in st.session_state.get("draft_actions", [])
        if action["team_role"] == role and action["action_type"] == action_type
    ]


def _hero_lists_for_current(team_role):
    actions = st.session_state.get("draft_actions", [])
    ally_picks = [a["hero_id"] for a in actions if a["team_role"] == team_role and a["action_type"] == "pick"]
    enemy_picks = [a["hero_id"] for a in actions if a["team_role"] != team_role and a["action_type"] == "pick"]
    ally_bans = [a["hero_id"] for a in actions if a["team_role"] == team_role and a["action_type"] == "ban"]
    enemy_bans = [a["hero_id"] for a in actions if a["team_role"] != team_role and a["action_type"] == "ban"]
    return ally_picks, enemy_picks, ally_bans, enemy_bans


def _draft_table(first_pick, own_team, opponent_team):
    table = get_empty_draft_table(first_pick, own_team, opponent_team)
    actions = st.session_state.get("draft_actions", [])
    for idx, action in enumerate(actions):
        table.loc[idx, "hero_id"] = action["hero_id"]
        table.loc[idx, "hero_name"] = action["hero_name"]
    return table[["order", "phase", "action_type", "team_role", "team_name", "hero_name"]]


def main():
    st.title("Dota 2 Draft Recommender")
    st.caption("Experimental local UI. Recommendations use available aggregate features and current saved ranker models.")

    if "draft_actions" not in st.session_state:
        st.session_state["draft_actions"] = []

    with st.sidebar:
        patch_label = st.selectbox("Patch", [PATCH_LABEL], index=0)
        dataset = st.selectbox("Dataset", ["players", "players_smooth", "interactions", "base"], index=0)
        teams = load_teams(patch_label)
        if teams.empty:
            st.error("No teams found. Build matches.parquet first.")
            return
        team_options = [_team_option(row) for _, row in teams.iterrows()]
        own_team = st.selectbox("Our team", team_options, format_func=_team_label, index=0)
        opponent_default = 1 if len(team_options) > 1 else 0
        opponent_team = st.selectbox("Opponent team", team_options, format_func=_team_label, index=opponent_default)
        own_side = st.radio("Our side", ["radiant", "dire"], horizontal=True)
        first_pick = st.radio("First pick", ["own", "opponent"], format_func=lambda x: "our team" if x == "own" else "opponent")
        top_k = st.slider("Top K", 5, 20, 10)
        signature = _draft_signature(patch_label, dataset, own_team, opponent_team, own_side, first_pick)
        if st.session_state.get("draft_signature") != signature:
            st.session_state["draft_signature"] = signature
            _reset_draft()
        if st.button("Reset draft", use_container_width=True):
            _reset_draft()
            st.rerun()

    if own_team["team_id"] == opponent_team["team_id"]:
        st.error("Choose different teams.")
        return

    current = _current_order(first_pick)
    own_side_for_role = {"own": own_side, "opponent": _other_side(own_side)}

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Draft table")
        st.dataframe(_draft_table(first_pick, own_team, opponent_team), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Draft panels")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{own_team['team_name']}**")
            st.write("Picks:", ", ".join(_actions_by_role("own", "pick")) or "-")
            st.write("Bans:", ", ".join(_actions_by_role("own", "ban")) or "-")
        with c2:
            st.markdown(f"**{opponent_team['team_name']}**")
            st.write("Picks:", ", ".join(_actions_by_role("opponent", "pick")) or "-")
            st.write("Bans:", ", ".join(_actions_by_role("opponent", "ban")) or "-")

    if current is None:
        st.success("Draft is complete.")
        return

    current_team = _team_for_role(current["team_role"], own_team, opponent_team)
    opponent_for_current = opponent_team if current["team_role"] == "own" else own_team
    acting_side = own_side_for_role[current["team_role"]]

    st.subheader(
        f"Current action: order {current['order']} | phase {current['draft_phase']} | "
        f"{current['action_type']} | {current_team['team_name']}"
    )

    heroes = load_heroes()
    unavailable = {action["hero_id"] for action in st.session_state.get("draft_actions", [])}
    available = heroes[~heroes["hero_id"].isin(unavailable)].copy()
    if available.empty:
        st.warning("No available heroes.")
        return

    ally_picks, enemy_picks, ally_bans, enemy_bans = _hero_lists_for_current(current["team_role"])

    recs = None
    own_roster = []
    opponent_roster = []
    recommendation_error = None
    try:
        recs, own_roster, opponent_roster = recommend(
            patch_label=patch_label,
            dataset=dataset,
            action_type=current["action_type"],
            acting_team_id=current_team["team_id"],
            opponent_team_id=opponent_for_current["team_id"],
            acting_team_name=current_team["team_name"],
            opponent_team_name=opponent_for_current["team_name"],
            acting_side=acting_side,
            order=current["order"],
            draft_phase=current["draft_phase"],
            ally_picks_before=ally_picks,
            enemy_picks_before=enemy_picks,
            ally_bans_before=ally_bans,
            enemy_bans_before=enemy_bans,
            unavailable_heroes=unavailable,
            top_k=max(top_k, 10),
        )
    except Exception as exc:
        recommendation_error = exc

    hero_records = available.to_dict("records")
    if recs is not None and not recs.empty:
        rank_lookup = {
            int(row["candidate_hero_id"]): int(row["rank"])
            for _, row in recs.head(10).iterrows()
        }
        by_id = {int(row["hero_id"]): row for row in hero_records}
        ordered_ids = [hero_id for hero_id in rank_lookup if hero_id in by_id]
        ordered_ids.extend([int(row["hero_id"]) for row in hero_records if int(row["hero_id"]) not in rank_lookup])
        hero_records = []
        for hero_id in ordered_ids:
            row = dict(by_id[hero_id])
            row["recommendation_rank"] = rank_lookup.get(hero_id)
            hero_records.append(row)

    selected = st.selectbox("Hero for current action", hero_records, format_func=_hero_label_with_rank)
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Add action", type="primary", use_container_width=True):
            if int(selected["hero_id"]) in unavailable:
                st.error("This hero is already unavailable.")
            else:
                st.session_state["draft_actions"].append({
                    "order": current["order"],
                    "action_type": current["action_type"],
                    "team_role": current["team_role"],
                    "team_id": current_team["team_id"],
                    "team_name": current_team["team_name"],
                    "hero_id": int(selected["hero_id"]),
                    "hero_name": selected["hero_name"],
                })
                st.rerun()
    with b2:
        if st.button("Undo last action", use_container_width=True):
            if st.session_state["draft_actions"]:
                st.session_state["draft_actions"].pop()
                st.rerun()

    st.subheader("Recommendations")
    if recs is not None:
        if not own_roster:
            st.warning(f"Roster was not found for {current_team['team_name']}; player features use defaults.")
        if not opponent_roster:
            st.warning(f"Roster was not found for {opponent_for_current['team_name']}; opponent player features use defaults.")
        st.dataframe(
            recs.head(top_k).rename(columns={"candidate_hero_name": "hero_name"})[["rank", "hero_name", "score", "explanation"]],
            use_container_width=True,
            hide_index=True,
        )
        for _, row in recs.head(3).iterrows():
            with st.expander(f"#{int(row['rank'])} {row['candidate_hero_name']}"):
                st.write(row["explanation"])
                st.json({"hero_id": int(row["candidate_hero_id"]), "score": float(row["score"])})
    elif isinstance(recommendation_error, FileNotFoundError):
        st.error(str(recommendation_error))
        st.info("Train the corresponding ranker model first, for example `python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset players`.")
    elif recommendation_error is not None:
        st.error(f"Recommendation failed: {recommendation_error}")


if __name__ == "__main__":
    main()
