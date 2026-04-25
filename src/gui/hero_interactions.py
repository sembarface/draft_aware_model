from pathlib import Path

import pandas as pd
import streamlit as st

from src.ml.interaction_tables import (
    build_conditional_bans,
    build_hero_matchups,
    build_hero_synergy,
)


def _load_or_build(path, build_func, save=False):
    if path.exists():
        return pd.read_parquet(path)

    df = build_func()
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def _filter_table(df, min_games, only_significant, significant_col):
    if df.empty:
        return df
    if "games" in df.columns:
        df = df[df["games"] >= min_games]
    elif "picked_games" in df.columns:
        df = df[df["picked_games"] >= min_games]
    if only_significant and significant_col in df.columns:
        df = df[df[significant_col]]
    return df


def show_hero_interactions(matches, heroes_stats, picks_bans, patch_num=None):
    st.subheader("Синергия и контрпики героев")

    if patch_num is None:
        patch_num = int(matches["patch"].dropna().iloc[0]) if "patch" in matches.columns else 60

    ml_dir = Path(f"data/patch_{patch_num}/ml")
    save_computed = st.checkbox("Save computed interaction tables", value=False)
    min_games = st.slider("Min games", min_value=1, max_value=100, value=20)
    only_significant = st.checkbox("Only significant", value=False)

    synergy = _load_or_build(
        ml_dir / "hero_synergy.parquet",
        lambda: build_hero_synergy(
            matches,
            heroes_stats,
            min_games=min_games,
            apply_min_games=False,
        ),
        save=save_computed,
    )
    matchups = _load_or_build(
        ml_dir / "hero_matchups.parquet",
        lambda: build_hero_matchups(
            matches,
            heroes_stats,
            min_games=min_games,
            apply_min_games=False,
        ),
        save=save_computed,
    )
    conditional_bans = _load_or_build(
        ml_dir / "hero_conditional_bans.parquet",
        lambda: build_conditional_bans(picks_bans, min_games=5),
        save=save_computed,
    )

    st.markdown("## Синергия героев")
    synergy_sort = st.selectbox(
        "Synergy sort",
        ["synergy_delta", "pair_winrate", "games", "p_value"],
        index=0,
    )
    synergy_view = _filter_table(synergy, min_games, only_significant, "significant")
    if not synergy_view.empty:
        st.dataframe(
            synergy_view[
                [
                    "hero1_name",
                    "hero2_name",
                    "games",
                    "pair_pick_freq",
                    "hero1_wr",
                    "hero2_wr",
                    "baseline_winrate",
                    "pair_winrate",
                    "synergy_delta",
                    "delta_hero1",
                    "delta_hero2",
                    "p_value",
                    "significant",
                ]
            ]
            .sort_values(synergy_sort, ascending=synergy_sort == "p_value")
            .round(4)
        )
    else:
        st.info("No synergy rows for selected filters.")

    st.markdown("## Контрпики")
    counter_sort = st.selectbox(
        "Counter sort",
        ["counter_delta", "matchup_winrate", "games", "p_value_win"],
        index=0,
    )
    counter_view = _filter_table(matchups, min_games, only_significant, "significant_win")
    if not counter_view.empty:
        st.dataframe(
            counter_view[
                [
                    "hero_name",
                    "vs_hero_name",
                    "games",
                    "hero_winrate",
                    "matchup_winrate",
                    "counter_delta",
                    "p_value_win",
                    "significant_win",
                ]
            ]
            .sort_values(counter_sort, ascending=counter_sort == "p_value_win")
            .round(4)
        )
    else:
        st.info("No matchup rows for selected filters.")

    if not conditional_bans.empty:
        st.markdown("## Условные баны")
        st.dataframe(
            conditional_bans[conditional_bans["picked_games"] >= min_games][
                [
                    "picked_hero_name",
                    "banned_hero_name",
                    "picked_games",
                    "bans_by_picked_team",
                    "bans_by_opponent_team",
                    "ban_rate_by_picked_team",
                    "ban_rate_by_opponent_team",
                ]
            ]
            .sort_values("ban_rate_by_opponent_team", ascending=False)
            .round(4)
        )
