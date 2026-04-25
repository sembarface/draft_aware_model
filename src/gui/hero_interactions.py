import streamlit as st
import pandas as pd
import itertools
from scipy.stats import binomtest

MIN_GAMES = 20
ALPHA = 0.05


def binom_pvalue(successes, trials, p0):
    if (
        trials is None
        or successes is None
        or p0 is None
        or pd.isna(p0)
        or trials <= 0
    ):
        return None
    successes = int(successes)
    trials = int(trials)
    if successes > trials:
        return None
    return binomtest(successes, trials, p0, alternative="two-sided").pvalue


def show_hero_interactions(matches, heroes_stats, picks_bans):
    st.subheader("Синергия и контрпики героев")

    total_matches = matches["match_id"].nunique()

    hero_name = dict(zip(heroes_stats["hero_id"], heroes_stats["hero_name"]))
    hero_wr = dict(zip(heroes_stats["hero_id"], heroes_stats["winrate"]))
    hero_ban_rate = dict(zip(heroes_stats["hero_id"], heroes_stats["ban_rate"]))

    # =================================================
    # СИНЕРГИЯ ГЕРОЕВ + СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ
    # =================================================
    st.markdown("## Синергия героев (со статистической значимостью)")

    synergy_records = []

    for _, row in matches.iterrows():
        radiant = [row[f"radiant_hero_{i}"] for i in range(1, 6) if pd.notna(row[f"radiant_hero_{i}"])]
        dire = [row[f"dire_hero_{i}"] for i in range(1, 6) if pd.notna(row[f"dire_hero_{i}"])]

        radiant_win = bool(row["radiant_win"])

        for team, win in [(radiant, radiant_win), (dire, not radiant_win)]:
            for h1, h2 in itertools.combinations(sorted(team), 2):
                synergy_records.append((h1, h2, win))

    synergy_df = pd.DataFrame(synergy_records, columns=["hero1", "hero2", "win"])

    synergy = (
        synergy_df.groupby(["hero1", "hero2"])
        .agg(
            games=("win", "count"),
            wins=("win", "sum")
        )
        .reset_index()
    )

    # фильтр по минимальному числу матчей
    synergy = synergy[synergy["games"] >= MIN_GAMES]

    # винрейты
    synergy["pair_winrate"] = synergy["wins"] / synergy["games"]

    synergy["hero1_wr"] = synergy["hero1"].map(hero_wr)
    synergy["hero2_wr"] = synergy["hero2"].map(hero_wr)

    # разница с базовым винрейтом
    synergy["delta_hero1"] = synergy["pair_winrate"] - synergy["hero1_wr"]
    synergy["delta_hero2"] = synergy["pair_winrate"] - synergy["hero2_wr"]

    # частота совместного пика
    synergy["pair_pick_freq"] = synergy["games"] / total_matches

    # базовый ожидаемый винрейт пары (симметрично)
    synergy["baseline_wr"] = (
        synergy["hero1_wr"] + synergy["hero2_wr"]
    ) / 2

    # p-value (биномиальный тест)
    synergy["p_value"] = synergy.apply(
        lambda r: binom_pvalue(
            successes=r["wins"],
            trials=r["games"],
            p0=r["baseline_wr"]
        ),
        axis=1
    )

    # статистическая значимость
    synergy["significant"] = synergy["p_value"] < ALPHA


    # имена героев
    synergy["hero1_name"] = synergy["hero1"].map(hero_name)
    synergy["hero2_name"] = synergy["hero2"].map(hero_name)

    # вывод
    st.dataframe(
        synergy[
            [
                "hero1_name", "hero2_name",
                "games",
                "pair_pick_freq",
                "hero1_wr", "hero2_wr",
                "baseline_wr",
                "pair_winrate",
                "delta_hero1", "delta_hero2",
                "p_value", "significant"
            ]
        ]
        .sort_values("delta_hero1", ascending=False)
        .round(4)
    )



    # ===============================
    # КОНТРПИКИ
    # ===============================
    st.markdown("## Контрпики с условной частотой банов")

    counter_records = []

    for _, row in matches.iterrows():
        radiant = [row[f"radiant_hero_{i}"] for i in range(1, 6) if pd.notna(row[f"radiant_hero_{i}"])]
        dire = [row[f"dire_hero_{i}"] for i in range(1, 6) if pd.notna(row[f"dire_hero_{i}"])]
        radiant_win = bool(row["radiant_win"])
        for r in radiant:
            for d in dire:
                counter_records.append((r, d, radiant_win))
                counter_records.append((d, r, not radiant_win))

    df = pd.DataFrame(counter_records, columns=["hero", "vs_hero", "win"])

    counters = (
        df.groupby(["hero", "vs_hero"])
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )
    counters = counters[counters["games"] >= MIN_GAMES]
    counters["matchup_winrate"] = counters["wins"] / counters["games"]
    counters["hero_winrate"] = counters["hero"].map(hero_wr)
    counters["delta_vs"] = counters["matchup_winrate"] - counters["hero_winrate"]

    # ---------- p-value по винрейту ----------
    counters["p_value_win"] = counters.apply(
        lambda r: binom_pvalue(r["wins"], r["games"], r["hero_winrate"]),
        axis=1
    )

    # -------------------------------------------------
    # Условная частота банов + p-value
    # -------------------------------------------------

    def compute_ban_stats(hero_id, vs_hero_id):
        """
        Возвращает:
        - vs_pick_games: сколько матчей, где vs_hero был пикнут
        - vs_ban_games: сколько раз hero был забанен в этих матчах
        """

        # матчи, где vs_hero был пикнут
        vs_pick_matches = picks_bans[
            (picks_bans["is_pick"]) &
            (picks_bans["hero_id"] == vs_hero_id)
        ][["match_id", "team"]]

        if vs_pick_matches.empty:
            return 0, 0

        match_ids = vs_pick_matches["match_id"].unique()

        # баны hero в этих матчах (на противоположной команде)
        hero_bans = picks_bans[
            (picks_bans["match_id"].isin(match_ids)) &
            (~picks_bans["is_pick"]) &
            (picks_bans["hero_id"] == hero_id)
        ]

        return len(match_ids), len(hero_bans)


    # считаем значения
    ban_stats = counters.apply(
        lambda r: compute_ban_stats(r["hero"], r["vs_hero"]),
        axis=1,
        result_type="expand"
    )

    ban_stats.columns = ["vs_pick_games", "vs_ban_games"]

    counters = pd.concat([counters, ban_stats], axis=1)

    # частота банов при наличии контр-героя
    counters["vs_ban_rate"] = (
        counters["vs_ban_games"] / counters["vs_pick_games"]
    )

    # p-value по банам
    counters["p_value_ban"] = counters.apply(
        lambda r: binom_pvalue(
            successes=r["vs_ban_games"],
            trials=r["vs_pick_games"],
            p0=hero_ban_rate.get(r["hero"], 0)
        ),
        axis=1
    )


    counters["significant_win"] = counters["p_value_win"] < ALPHA
    counters["significant_ban"] = counters["p_value_ban"] < ALPHA
    counters["significant_overall"] = counters["significant_win"] | counters["significant_ban"]

    counters["hero_name"] = counters["hero"].map(hero_name)
    counters["vs_hero_name"] = counters["vs_hero"].map(hero_name)

    st.dataframe(
        counters[
            [
                "hero_name", "vs_hero_name",
                "games",
                "hero_winrate", "matchup_winrate", "delta_vs",
                "vs_ban_rate",
                "p_value_win", "p_value_ban",
                "significant_overall"
            ]
        ]
        .sort_values("delta_vs", ascending=False)
        .round(4)
    )
