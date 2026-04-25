#streamlit run src/gui/app.py
import streamlit as st

from data_loader import load_data
from distributions import show_distributions
from correlations import show_correlations
from hero_interactions import show_hero_interactions


st.set_page_config(
    page_title="Многомерный статистический анализ матчей Dota 2",
    layout="wide"
)

st.title("Многомерный статистический анализ матчей Dota 2")

st.sidebar.header("Параметры анализа")

PATCH = st.sidebar.selectbox("Патч", [58,59,60])

players, matches, picks, heroes_stats = load_data(PATCH)

SECTION = st.sidebar.radio(
    "Раздел анализа (глава 3)",
    [
        "Табличные данные",
        "3.1 Распределения",
        "3.2 Корреляционный анализ",
        "3.3 Синергия и контрпики героев"
    ]
)

if SECTION == "Табличные данные":
    table = st.selectbox("Таблица", ["players", "matches", "picks_bans", "heroes_stats"])
    if table == "players":
        st.dataframe(players)
    elif table == "matches":
        st.dataframe(matches)
    elif table == "picks_bans":
        st.dataframe(picks)
    else:
        st.dataframe(heroes_stats)


elif SECTION == "3.1 Распределения":
    show_distributions(players, matches)

elif SECTION == "3.2 Корреляционный анализ":
    show_correlations(players)

elif SECTION == "3.3 Синергия и контрпики героев":
    show_hero_interactions(matches, heroes_stats, picks)

