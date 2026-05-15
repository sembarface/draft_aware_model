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


def _render_draft_table(table, own_side="radiant", current_order=None):
    radiant_role = "own" if own_side == "radiant" else "opponent"
    dire_role = "opponent" if radiant_role == "own" else "own"

    def _team_name(role):
        part = table[table["team_role"] == role]
        if part.empty:
            return "unknown"
        return str(part.iloc[0].get("team_name") or "unknown")

    def _slot(row, side_role):
        if row.get("team_role") != side_role:
            return '<div class="dota-slot dota-slot-spacer"></div>'

        filled = bool(row.get("hero_name"))
        action_type = str(row.get("action_type") or "")
        current = int(row.get("order")) == current_order if current_order is not None else False
        classes = ["dota-slot", f"dota-slot-{action_type}"]
        if filled:
            classes.append("dota-slot-filled")
        if current:
            classes.append("dota-slot-current")

        hero_name = str(row.get("hero_name") or "")
        hero_image = row.get("hero_image") or ""
        image_html = f'<img src="{escape(hero_image)}" alt="" class="dota-hero-img">' if hero_image else ""
        ban_mark = '<div class="dota-ban-mark">×</div>' if action_type == "ban" and filled else ""
        empty_label = "" if filled else '<div class="dota-empty-fill"></div>'
        title = escape(hero_name or f"order {row.get('order')} {action_type}")

        return (
            f'<div class="{" ".join(classes)}" title="{title}">'
            f"{image_html}{empty_label}{ban_mark}"
            "</div>"
        )

    rows = []
    for _, row in table.iterrows():
        order = int(row.get("order"))
        phase_break = " dota-row-phase-break" if order in {8, 10, 13, 19, 23} else ""
        current_class = " dota-order-current" if current_order == order else ""
        rows.append(
            '<div class="dota-draft-row{phase_break}">'
            '<div class="dota-side dota-side-left">{left}</div>'
            '<div class="dota-order-line dota-order-line-left"></div>'
            '<div class="dota-order{current_class}">{order}</div>'
            '<div class="dota-order-line dota-order-line-right"></div>'
            '<div class="dota-side dota-side-right">{right}</div>'
            "</div>".format(
                phase_break=phase_break,
                left=_slot(row, radiant_role),
                current_class=current_class,
                order=order,
                right=_slot(row, dire_role),
            )
        )

    html = dedent("""
    <style>
      .dota-draft-board {
        width: min(100%, 430px);
        margin: 0 auto 0.75rem;
        padding: 0.75rem 0.8rem 0.9rem;
        border: 1px solid rgba(185, 210, 230, 0.18);
        border-radius: 4px;
        background:
          linear-gradient(90deg, rgba(30, 75, 108, 0.55), rgba(21, 30, 40, 0.94) 49%, rgba(49, 43, 34, 0.92)),
          radial-gradient(circle at 15% 15%, rgba(87, 168, 210, 0.22), transparent 32%);
        color: #e5edf3;
        box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.35), 0 12px 30px rgba(0, 0, 0, 0.18);
      }
      .dota-board-header {
        display: grid;
        grid-template-columns: 1fr 44px 1fr;
        align-items: start;
        margin-bottom: 0.55rem;
        column-gap: 0.35rem;
      }
      .dota-faction-title {
        font-size: 1.05rem;
        line-height: 1.05;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0;
        color: #f2efe7;
      }
      .dota-faction-title-radiant {
        color: #a7ff6e;
        text-shadow: 0 0 7px rgba(101, 255, 76, 0.9);
      }
      .dota-faction-title-dire {
        text-align: right;
      }
      .dota-team-subtitle {
        margin-top: 0.22rem;
        max-width: 160px;
        font-size: 0.68rem;
        line-height: 1.15;
        color: rgba(230, 239, 247, 0.76);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .dota-faction-title-dire .dota-team-subtitle {
        margin-left: auto;
      }
      .dota-draft-row {
        display: grid;
        grid-template-columns: 1fr 18px 24px 18px 1fr;
        align-items: center;
        min-height: 32px;
      }
      .dota-row-phase-break {
        margin-top: 0.28rem;
      }
      .dota-side {
        display: flex;
        align-items: center;
      }
      .dota-side-left {
        justify-content: flex-end;
      }
      .dota-side-right {
        justify-content: flex-start;
      }
      .dota-order {
        position: relative;
        z-index: 1;
        width: 24px;
        text-align: center;
        font-size: 0.75rem;
        font-weight: 800;
        color: #e7edf2;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.9);
      }
      .dota-order-current {
        color: #ffffff;
        text-shadow: 0 0 6px rgba(111, 190, 255, 0.95);
      }
      .dota-order-line {
        height: 1px;
        background: rgba(206, 218, 226, 0.45);
      }
      .dota-slot {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        background: rgba(8, 12, 15, 0.72);
        border: 1px solid rgba(215, 225, 230, 0.26);
        overflow: hidden;
        box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.55);
      }
      .dota-slot-spacer {
        opacity: 0;
        pointer-events: none;
      }
      .dota-slot-ban {
        width: 58px;
        height: 30px;
      }
      .dota-slot-pick {
        width: 86px;
        height: 48px;
      }
      .dota-slot-current {
        border-color: rgba(144, 205, 255, 0.95);
        box-shadow: 0 0 0 1px rgba(144, 205, 255, 0.45), 0 0 14px rgba(78, 160, 245, 0.5);
      }
      .dota-hero-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: saturate(1.08) contrast(1.04);
      }
      .dota-empty-fill {
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(34, 42, 48, 0.88), rgba(8, 10, 12, 0.9));
      }
      .dota-ban-mark {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ff321f;
        font-size: 2.1rem;
        line-height: 1;
        font-weight: 900;
        text-shadow: 0 1px 0 #520900, 0 0 6px rgba(255, 40, 20, 0.7);
      }
      @media (max-width: 640px) {
        .dota-draft-board {
          width: 100%;
          padding-left: 0.45rem;
          padding-right: 0.45rem;
        }
        .dota-slot-pick {
          width: 76px;
          height: 43px;
        }
        .dota-slot-ban {
          width: 52px;
          height: 28px;
        }
        .dota-faction-title {
          font-size: 0.9rem;
        }
      }
    </style>
    <div class="dota-draft-board">
      <div class="dota-board-header">
        <div class="dota-faction-title dota-faction-title-radiant">
          <div>СИЛЫ</div><div>СВЕТА</div>
          <div class="dota-team-subtitle">__RADIANT_TEAM__</div>
        </div>
        <div></div>
        <div class="dota-faction-title dota-faction-title-dire">
          <div>СИЛЫ ТЬМЫ</div>
          <div class="dota-team-subtitle">__DIRE_TEAM__</div>
        </div>
      </div>
      __ROWS__
    </div>
    """)
    html = (
        html.replace("__RADIANT_TEAM__", escape(_team_name(radiant_role)))
        .replace("__DIRE_TEAM__", escape(_team_name(dire_role)))
        .replace("__ROWS__", "\n".join(rows))
    )
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


def _factor_items(explanation, limit=3):
    text = str(explanation or "").replace("Положительные факторы:", "").strip()
    if not text:
        return []
    items = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            title, detail = part.split(":", 1)
            detail = detail.strip().split(";")[0].strip()
            items.append((title.strip(), _compact_factor_detail(detail)))
        else:
            items.append(("Фактор", _compact_factor_detail(part.split(";")[0].strip())))
    return items[:limit]


def _percent_suffix(text):
    marker = text.rfind(":")
    if marker >= 0:
        value = text[marker + 1 :].strip()
        if value.endswith("%"):
            return value
    for part in reversed(text.split()):
        if part.endswith("%"):
            return part
    return ""


def _compact_factor_detail(detail, max_len=64):
    detail = str(detail or "").strip()
    percent = _percent_suffix(detail)
    if "comfort-сигнал" in detail:
        compact = detail.replace("сильный comfort-сигнал: ", "")
    elif "игр на этом герое" in detail or "большой опыт" in detail:
        compact = detail.replace("игрок вашей команды ", "").replace("игрок соперника ", "")
    elif "недостающую роль" in detail:
        compact = "закрывает недостающую роль"
    elif "core-роль" in detail:
        compact = "закрывает core-роль"
    elif "support-роль" in detail:
        compact = "закрывает support-роль"
    elif "часто банили" in detail:
        compact = f"часто банят против команды: {percent}" if percent else "часто банят против команды"
    elif "часто выбирала" in detail or "часто выбирали" in detail:
        compact = f"часто выбирали в патче: {percent}" if percent else "часто выбирали в патче"
    elif "contested rate" in detail:
        compact = f"высокий contested rate: {percent}" if percent else "высокий contested rate"
    elif "часто забирали рано" in detail:
        compact = f"часто забирали рано: {percent}" if percent else "часто забирали рано"
    elif "гибок по ролям" in detail:
        compact = "гибкий по ролям герой"
    else:
        compact = detail

    if len(compact) > max_len:
        compact = compact[: max_len - 1].rstrip() + "…"
    return compact


def _player_context_rows(row):
    rows = []
    for prefix, side in [("own_", "Наша команда"), ("opponent_", "Соперник")]:
        for scope, scope_title in [("patch", "Текущий патч"), ("alltime", "Вся статистика")]:
            games = row.get(f"{prefix}best_player_hero_games_{scope}")
            try:
                games_value = float(games)
            except Exception:
                games_value = 0.0
            if games_value <= 0:
                continue

            winrate = row.get(f"{prefix}best_player_hero_winrate_{scope}")
            kda = row.get(f"{prefix}best_player_hero_avg_kda_{scope}")
            try:
                winrate_value = float(winrate)
                winrate_text = f"{float(winrate) * 100:.1f}%"
            except Exception:
                winrate_value = 0.0
                winrate_text = "-"
            try:
                kda_text = f"{float(kda):.2f}"
            except Exception:
                kda_text = "-"

            rows.append(
                {
                    "Сигнал": "сильный" if games_value >= 3 and winrate_value >= 0.55 else "есть опыт",
                    "Сторона": side,
                    "Период": scope_title,
                    "Игрок": row.get(f"{prefix}best_player_hero_name_{scope}") or "-",
                    "Игры": int(round(games_value)),
                    "WR": winrate_text,
                    "KDA": kda_text,
                }
            )
    return pd.DataFrame(rows)


def _render_recommendation_detail_body(row):
    if row is None:
        return
    st.markdown(row.get("explanation_markdown") or row["explanation"])
    player_context = _player_context_rows(row)
    if not player_context.empty:
        st.caption("Статистика игроков на этом герое")
        st.dataframe(
            player_context,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Сигнал": st.column_config.TextColumn(width="small"),
                "Сторона": st.column_config.TextColumn(width="small"),
                "Период": st.column_config.TextColumn(width="medium"),
                "Игрок": st.column_config.TextColumn(width="medium"),
                "Игры": st.column_config.NumberColumn(width="small"),
                "WR": st.column_config.TextColumn(width="small"),
                "KDA": st.column_config.TextColumn(width="small"),
            },
        )


def _open_recommendation_dialog(row):
    title = f"#{int(row['rank'])} {row['candidate_hero_name']}"
    dialog = getattr(st, "dialog", None)
    if dialog is None:
        st.markdown(f"#### {title}")
        _render_recommendation_detail_body(row)
        return

    @dialog(title, width="large")
    def _dialog():
        st.markdown(
            """
            <style>
              div[data-testid="stDialog"] div[role="dialog"] {
                width: min(96vw, 1080px);
                max-width: 96vw;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _render_recommendation_detail_body(row)

    _dialog()


def _render_factor_chips(row):
    factors = _factor_items(row.get("explanation"), limit=3)
    if not factors:
        factors = [("Фактор", "Суммарная оценка ranker-модели")]
    for title, detail in factors:
        st.markdown(
            (
                '<span class="rec-factor">'
                f'<span class="rec-factor-title">{escape(title)}</span>'
                f'<span>{escape(detail)}</span>'
                "</span>"
            ),
            unsafe_allow_html=True,
        )


def _render_recommendation_cards(recs, top_k):
    selected_row = None
    html = dedent("""
    <style>
      .rec-card-img {
        width: 100%;
        height: 118px;
        object-fit: cover;
        border-radius: 7px;
        filter: saturate(1.07) contrast(1.04);
      }
      .rec-factor-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.45rem;
      }
      .rec-factor {
        display: inline-flex;
        gap: 0.28rem;
        align-items: center;
        min-width: 0;
        max-width: 100%;
        padding: 0.18rem 0.42rem;
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.76);
        border: 1px solid rgba(147, 197, 253, 0.16);
        font-size: 0.78rem;
        line-height: 1.25;
        color: #cbd5e1;
      }
      .rec-factor-title {
        flex: 0 0 auto;
        color: #93c5fd;
        font-weight: 800;
      }
    </style>
    """)
    st.markdown(html, unsafe_allow_html=True)

    rows = recs.head(top_k).reset_index(drop=True)
    for start in range(0, len(rows), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, rows.iloc[start : start + 4].iterrows()):
            hero_name = str(row.get("candidate_hero_name") or "")
            with col.container(border=True):
                image_src = get_hero_image_src(_hero_image_key(hero_name))
                if image_src:
                    st.markdown(
                        f'<img src="{escape(image_src)}" alt="" class="rec-card-img">',
                        unsafe_allow_html=True,
                    )
                if st.button(
                    f"#{int(row['rank'])} {hero_name}",
                    key=f"rec_detail_{int(row['candidate_hero_id'])}_{int(row['rank'])}",
                    use_container_width=True,
                ):
                    selected_row = row
                st.caption(f"score {float(row['score']):.3f}")
                st.markdown('<div class="rec-factor-wrap">', unsafe_allow_html=True)
                _render_factor_chips(row)
                st.markdown("</div>", unsafe_allow_html=True)
    return selected_row


def main():
    st.title("Dota 2 Draft Recommender")
    st.caption("Experimental local UI. Recommendations use available aggregate features and current saved ranker models.")

    if "draft_actions" not in st.session_state:
        st.session_state["draft_actions"] = []

    with st.sidebar:
        patch_label = st.selectbox("Patch", [PATCH_LABEL], index=0)
        dataset = st.selectbox(
            "Dataset",
            ["players_team_role", "players_team", "interactions_role", "interactions", "base"],
            index=0,
        )
        st.caption("Main model: players_team_role. Universal/no-roster model: interactions_role.")
        needs_known_teams = dataset in {"players_team", "players_team_role"}
        if needs_known_teams:
            st.caption("Team-specific model: uses local team/player history.")
            teams = load_teams(patch_label)
            if teams.empty:
                st.error("No teams found. Build matches.parquet first.")
                return
            team_options = [_team_option(row) for _, row in teams.iterrows()]
            own_team = st.selectbox("Our team", team_options, format_func=_team_label, index=0)
            opponent_default = 1 if len(team_options) > 1 else 0
            opponent_team = st.selectbox("Opponent team", team_options, format_func=_team_label, index=opponent_default)
        else:
            st.caption("Universal model: no known rosters required. Uses generic Team A vs Team B.")
            own_team = {"team_id": -1, "team_name": "Team A"}
            opponent_team = {"team_id": -2, "team_name": "Team B"}
        own_side = st.radio("Our side", ["radiant", "dire"], horizontal=True)
        first_pick = st.radio("First pick", ["own", "opponent"], format_func=lambda x: "our team" if x == "own" else "opponent")
        top_k = st.slider("Top K", 5, 20, 10)
        show_debug_timings = st.checkbox("Show debug timings", value=False)
        signature = _draft_signature(patch_label, dataset, own_team, opponent_team, own_side, first_pick)
        if st.session_state.get("draft_signature") != signature:
            st.session_state["draft_signature"] = signature
            _reset_draft()
        st.button("Reset draft", use_container_width=True, on_click=_reset_draft)

    if needs_known_teams and own_team["team_id"] == opponent_team["team_id"]:
        st.error("Choose different teams.")
        return

    current = _current_order(first_pick)
    own_side_for_role = {"own": own_side, "opponent": _other_side(own_side)}

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Draft board")
        _render_draft_table(
            _draft_table(first_pick, own_team, opponent_team),
            own_side=own_side,
            current_order=current["order"] if current is not None else None,
        )

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
    if current["action_type"] == "pick":
        action_button_bg = "#16803c"
        action_button_hover = "#116b31"
        action_button_border = "#0f5f2c"
    else:
        action_button_bg = "#b42318"
        action_button_hover = "#981b13"
        action_button_border = "#86180f"
    st.markdown(
        f"""
        <style>
          div[data-testid="stButton"] button[kind="primary"] {{
            background: {action_button_bg};
            border-color: {action_button_border};
            color: white;
          }}
          div[data-testid="stButton"] button[kind="primary"]:hover {{
            background: {action_button_hover};
            border-color: {action_button_border};
            color: white;
          }}
          div[data-testid="stButton"] button[kind="primary"]:focus {{
            box-shadow: 0 0 0 0.18rem color-mix(in srgb, {action_button_bg} 30%, transparent);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns([1, 1])
    with b1:
        st.button(
            "Add pick" if current["action_type"] == "pick" else "Add ban",
            type="primary",
            use_container_width=True,
            on_click=_add_action,
            args=(pending_action,),
        )
    with b2:
        st.button("Undo last action", use_container_width=True, on_click=_undo_last_action)

    st.subheader("Recommendations")
    if recs is not None:
        if needs_known_teams and not own_roster:
            st.warning(f"Roster was not found for {current_team['team_name']}; player features use defaults.")
        if needs_known_teams and not opponent_roster:
            st.warning(f"Roster was not found for {opponent_for_current['team_name']}; opponent player features use defaults.")
        selected_row = _render_recommendation_cards(recs, top_k)
        if selected_row is not None:
            _open_recommendation_dialog(selected_row)
        with st.expander("Таблица ранжирования"):
            rec_table = recs.head(top_k).rename(columns={"candidate_hero_name": "hero_name"})
            st.dataframe(
                rec_table[["rank", "hero_name", "score"]],
                use_container_width=True,
                hide_index=True,
            )
    elif isinstance(recommendation_error, FileNotFoundError):
        st.error(str(recommendation_error))
        st.info(
            "Train the corresponding ranker model first, for example "
            f"`python -m src.ml.train_catboost --patch-label {patch_label} --action {current['action_type']} --dataset {dataset}`."
        )
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
