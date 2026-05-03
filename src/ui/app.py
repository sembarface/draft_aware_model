import sys
import base64
import time
import urllib.request
from html import escape
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PATCH_LABEL
from src.ui.draft_order import get_draft_order, get_empty_draft_table
from src.ui.recommender import load_heroes, load_teams, recommend_cached


st.set_page_config(page_title="Dota Draft Recommender", layout="wide")


HERO_ICON_DIR = ROOT_DIR / "data" / "hero_icons"


HERO_IMAGE_KEY_OVERRIDES = {
    "Anti-Mage": "antimage",
    "Centaur Warrunner": "centaur",
    "Clockwerk": "rattletrap",
    "Doom": "doom_bringer",
    "Io": "wisp",
    "Lifestealer": "life_stealer",
    "Magnus": "magnataur",
    "Nature's Prophet": "furion",
    "Necrophos": "necrolyte",
    "Outworld Destroyer": "obsidian_destroyer",
    "Queen of Pain": "queenofpain",
    "Shadow Fiend": "nevermore",
    "Timbersaw": "shredder",
    "Treant Protector": "treant",
    "Underlord": "abyssal_underlord",
    "Vengeful Spirit": "vengefulspirit",
    "Windranger": "windrunner",
    "Wraith King": "skeleton_king",
    "Zeus": "zuus",
}


def _team_option(row):
    return {"team_id": int(row["team_id"]), "team_name": str(row["team_name"])}


def _team_label(team):
    return f"{team['team_name']} ({team['team_id']})"


def _other_side(side):
    return "dire" if side == "radiant" else "radiant"


def _hero_label(row):
    return f"{row['hero_name']} ({row['hero_id']})"


def _hero_image_key(hero_name):
    if not hero_name or pd.isna(hero_name):
        return ""
    hero_name = str(hero_name)
    if hero_name in HERO_IMAGE_KEY_OVERRIDES:
        return HERO_IMAGE_KEY_OVERRIDES[hero_name]
    return (
        hero_name.lower()
        .replace("'", "")
        .replace("-", "")
        .replace(" ", "_")
    )


def get_hero_image_url(hero_image_key):
    if not hero_image_key:
        return ""
    return f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{hero_image_key}.png"


@st.cache_data(show_spinner=False)
def get_hero_image_src(hero_image_key):
    if not hero_image_key:
        return ""
    HERO_ICON_DIR.mkdir(parents=True, exist_ok=True)
    icon_path = HERO_ICON_DIR / f"{hero_image_key}.png"
    if not icon_path.exists():
        try:
            with urllib.request.urlopen(get_hero_image_url(hero_image_key), timeout=2) as response:
                icon_path.write_bytes(response.read())
        except Exception:
            return get_hero_image_url(hero_image_key)
    try:
        encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return get_hero_image_url(hero_image_key)


def _hero_label_with_rank(row):
    if row.get("recommendation_rank"):
        return f"#{int(row['recommendation_rank'])} {row['hero_name']} ({row['hero_id']})"
    return _hero_label(row)


def _draft_signature(patch_label, dataset, own_team, opponent_team, own_side, first_pick):
    return (patch_label, dataset, own_team["team_id"], opponent_team["team_id"], own_side, first_pick)


def _reset_draft():
    st.session_state["draft_actions"] = []


def _add_action(action):
    unavailable = {item["hero_id"] for item in st.session_state.get("draft_actions", [])}
    if action["hero_id"] not in unavailable:
        st.session_state["draft_actions"].append(action)


def _undo_last_action():
    if st.session_state.get("draft_actions"):
        st.session_state["draft_actions"].pop()


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


def _action_records_by_role(role, action_type):
    return [
        action
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
    table["hero_image"] = ""
    actions = st.session_state.get("draft_actions", [])
    for idx, action in enumerate(actions):
        table.loc[idx, "hero_id"] = action["hero_id"]
        table.loc[idx, "hero_name"] = action["hero_name"]
        table.loc[idx, "hero_image"] = get_hero_image_src(_hero_image_key(action["hero_name"]))
    return table[["order", "phase", "action_type", "team_role", "team_name", "hero_image", "hero_name"]]


def _render_draft_table(table):
    rows = []
    for _, row in table.iterrows():
        filled = bool(row.get("hero_name"))
        action_type = row.get("action_type")
        if filled and action_type == "pick":
            background = "#f0f9f2"
            border = "#63a879"
        elif filled and action_type == "ban":
            background = "#fff1f1"
            border = "#c97070"
        else:
            background = "#f8fafc"
            border = "#d1d5db"
        hero_image = row.get("hero_image") or ""
        image_html = (
            f'<img src="{escape(hero_image)}" alt="" class="draft-hero-img">'
            if hero_image
            else ""
        )
        hero_html = (
            f'<div class="draft-hero-cell">{image_html}<span>{escape(str(row.get("hero_name") or ""))}</span></div>'
            if filled
            else ""
        )
        rows.append(
            "<tr style=\"background:{background};border-left:4px solid {border};\">"
            "<td>{order}</td>"
            "<td>{phase}</td>"
            "<td>{action}</td>"
            "<td>{team_role}</td>"
            "<td>{team_name}</td>"
            "<td>{hero}</td>"
            "</tr>".format(
                background=background,
                border=border,
                order=escape(str(row.get("order", ""))),
                phase=escape(str(row.get("phase", ""))),
                action=escape(str(row.get("action_type", ""))),
                team_role=escape(str(row.get("team_role", ""))),
                team_name=escape(str(row.get("team_name", ""))),
                hero=hero_html,
            )
        )
    html = dedent("""
    <style>
      .draft-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        color: #111827;
      }
      .draft-table th, .draft-table td {
        border-bottom: 1px solid rgba(49, 51, 63, 0.18);
        padding: 0.32rem 0.45rem;
        vertical-align: middle;
      }
      .draft-table th {
        text-align: left;
        font-weight: 600;
        color: #111827;
        background: #f3f4f6;
      }
      .draft-hero-cell {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-height: 34px;
        font-weight: 600;
      }
      .draft-hero-img {
        width: 58px;
        height: 32px;
        object-fit: cover;
        border-radius: 5px;
        box-shadow: 0 0 0 1px rgba(17, 24, 39, 0.12);
      }
    </style>
    <table class="draft-table">
      <thead>
        <tr>
          <th>order</th>
          <th>phase</th>
          <th>action</th>
          <th>role</th>
          <th>team</th>
          <th>hero</th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
    """).replace("__ROWS__", "\n".join(rows))
    st.markdown(html, unsafe_allow_html=True)


def _render_action_chips(title, actions, action_type):
    chips = []
    for action in actions:
        hero_name = action.get("hero_name") or ""
        image_url = get_hero_image_src(_hero_image_key(hero_name))
        image_html = (
            f'<img src="{escape(image_url)}" alt="" class="draft-chip-img">'
            if image_url
            else ""
        )
        chips.append(
            '<div class="draft-chip draft-chip-{action_type}">'
            '{image}<span>{hero}</span>'
            '</div>'.format(
                action_type=escape(action_type),
                image=image_html,
                hero=escape(hero_name),
            )
        )
    body = "\n".join(chips) if chips else '<div class="draft-chip-empty">-</div>'
    html = dedent("""
    <style>
      .draft-chip-title {
        margin: 0.45rem 0 0.25rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: #4b5563;
      }
      .draft-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.45rem;
      }
      .draft-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        max-width: 100%;
        padding: 0.18rem 0.42rem 0.18rem 0.22rem;
        border-radius: 6px;
        border: 1px solid rgba(17, 24, 39, 0.12);
        color: #111827;
        font-size: 0.82rem;
        font-weight: 600;
      }
      .draft-chip-pick {
        background: #f0f9f2;
        border-color: #b7dcc2;
      }
      .draft-chip-ban {
        background: #fff1f1;
        border-color: #edcaca;
      }
      .draft-chip-img {
        width: 38px;
        height: 22px;
        object-fit: cover;
        border-radius: 4px;
      }
      .draft-chip-empty {
        color: #6b7280;
        font-size: 0.9rem;
      }
    </style>
    <div class="draft-chip-title">__TITLE__</div>
    <div class="draft-chip-wrap">__CHIPS__</div>
    """).replace("__TITLE__", escape(title)).replace("__CHIPS__", body)
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.title("Dota 2 Draft Recommender")
    st.caption("Experimental local UI. Recommendations use available aggregate features and current saved ranker models.")

    if "draft_actions" not in st.session_state:
        st.session_state["draft_actions"] = []

    with st.sidebar:
        patch_label = st.selectbox("Patch", [PATCH_LABEL], index=0)
        dataset = st.selectbox("Dataset", ["players", "interactions", "base"], index=0)
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
        show_debug_timings = st.checkbox("Show debug timings", value=False)
        signature = _draft_signature(patch_label, dataset, own_team, opponent_team, own_side, first_pick)
        if st.session_state.get("draft_signature") != signature:
            st.session_state["draft_signature"] = signature
            _reset_draft()
        st.button("Reset draft", use_container_width=True, on_click=_reset_draft)

    if own_team["team_id"] == opponent_team["team_id"]:
        st.error("Choose different teams.")
        return

    current = _current_order(first_pick)
    own_side_for_role = {"own": own_side, "opponent": _other_side(own_side)}

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Draft table")
        _render_draft_table(_draft_table(first_pick, own_team, opponent_team))

    with right:
        st.subheader("Draft panels")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{own_team['team_name']}**")
            _render_action_chips("Picks", _action_records_by_role("own", "pick"), "pick")
            _render_action_chips("Bans", _action_records_by_role("own", "ban"), "ban")
        with c2:
            st.markdown(f"**{opponent_team['team_name']}**")
            _render_action_chips("Picks", _action_records_by_role("opponent", "pick"), "pick")
            _render_action_chips("Bans", _action_records_by_role("opponent", "ban"), "ban")

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
    recommendation_elapsed = None
    try:
        started_at = time.perf_counter()
        recs, own_roster, opponent_roster = recommend_cached(
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
            ally_picks_before=tuple(ally_picks),
            enemy_picks_before=tuple(enemy_picks),
            ally_bans_before=tuple(ally_bans),
            enemy_bans_before=tuple(enemy_bans),
            unavailable_heroes=tuple(sorted(unavailable)),
        )
        recommendation_elapsed = time.perf_counter() - started_at
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
    pending_action = {
        "order": current["order"],
        "action_type": current["action_type"],
        "team_role": current["team_role"],
        "team_id": current_team["team_id"],
        "team_name": current_team["team_name"],
        "hero_id": int(selected["hero_id"]),
        "hero_name": selected["hero_name"],
    }
    b1, b2 = st.columns([1, 1])
    with b1:
        st.button(
            "Add action",
            type="primary",
            use_container_width=True,
            on_click=_add_action,
            args=(pending_action,),
        )
    with b2:
        st.button("Undo last action", use_container_width=True, on_click=_undo_last_action)

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
                st.markdown(row.get("explanation_markdown") or row["explanation"])
    elif isinstance(recommendation_error, FileNotFoundError):
        st.error(str(recommendation_error))
        st.info("Train the corresponding ranker model first, for example `python -m src.ml.train_catboost --patch-label 7.41 --action pick --dataset players`.")
    elif recommendation_error is not None:
        st.error(f"Recommendation failed: {recommendation_error}")

    if show_debug_timings:
        st.sidebar.caption(
            "Recommendations: "
            f"{recommendation_elapsed:.3f}s | "
            f"available heroes: {len(available)} | "
            f"draft actions: {len(st.session_state.get('draft_actions', []))}"
            if recommendation_elapsed is not None
            else f"Recommendations: failed | available heroes: {len(available)} | draft actions: {len(st.session_state.get('draft_actions', []))}"
        )


if __name__ == "__main__":
    main()
