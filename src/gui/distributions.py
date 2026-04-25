import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

PLAYER_COLS = [
    "kills", "deaths", "assists",
    "last_hits", "denies",
    "teamfight_participation", "level",
    "gold_per_min", "xp_per_min",
    "kills_per_min",
    "hero_damage", "tower_damage", "hero_healing", "net_worth"
]

def show_distributions(players, matches):
    st.subheader("Анализ распределений игровых параметров")

    source = st.radio("Источник данных", ["players", "matches"])

    if source == "players":
        col = st.selectbox("Параметр", PLAYER_COLS)
        data = players[col].dropna()
    else:
        col = "duration"
        data = matches[col].dropna()

    mean = data.mean()
    median = data.median()
    mode = data.mode().iloc[0] if not data.mode().empty else np.nan
    std = data.std()

    show_stats = st.multiselect(
        "Отображать на графике:",
        ["Среднее", "Медиана", "Мода"],
        default=["Среднее", "Медиана", "Мода"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data, kde=True, ax=ax)

    if "Среднее" in show_stats:
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f}")
    if "Медиана" in show_stats:
        ax.axvline(median, color="green", linestyle="--", label=f"Median = {median:.2f}")
    if "Мода" in show_stats:
        ax.axvline(mode, color="blue", linestyle="--", label=f"Mode = {mode:.2f}")

    if show_stats:
        ax.legend()

    ax.set_title(f"Распределение параметра: {col}")

    ax.text(
        0.95, 0.95,
        f"σ = {std:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    st.pyplot(fig)

    st.markdown("### Q–Q диаграмма")
    fig, ax = plt.subplots(figsize=(4, 4))
    stats.probplot(data, dist="norm", plot=ax)
    st.pyplot(fig)
