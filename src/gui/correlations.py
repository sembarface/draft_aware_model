import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


NUM_COLS = [
    "kills", "deaths", "assists",
    "last_hits", "denies",
    "teamfight_participation", "level",
    "gold_per_min", "xp_per_min",
    "kills_per_min",
    "hero_damage", "tower_damage", "hero_healing",
    "net_worth"
]

TARGET_CORR_MIN = 0
INDEP_CORR_MAX = 0.4
ALPHA = 0.05


def corr_with_pvalues(df, method):
    corr = pd.DataFrame(index=NUM_COLS, columns=NUM_COLS, dtype=float)
    pvals = pd.DataFrame(index=NUM_COLS, columns=NUM_COLS, dtype=float)

    for i in NUM_COLS:
        for j in NUM_COLS:
            x = df[i]
            y = df[j]
            if method == "pearson":
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)
            corr.loc[i, j] = r
            pvals.loc[i, j] = p

    return corr, pvals


def multiple_correlation(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    r, _ = stats.pearsonr(y, y_pred)
    return r


def show_correlations(players):
    st.subheader("Корреляционный анализ игровых параметров")

    method = st.radio("Метод корреляции", ["spearman", "pearson",])

    df = players[NUM_COLS].dropna()

    # =========================================================
    # 1. ПАРНЫЕ КОРРЕЛЯЦИИ (ИСХОДНЫЙ КОД)
    # =========================================================
    corr, pvals = corr_with_pvalues(df, method)

    st.markdown("### Корреляционная матрица")
    st.dataframe(corr.round(3))

    st.markdown("### p-value матрица")
    st.dataframe(pvals.round(5))

    st.markdown("### Тепловая карта корреляций")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    st.pyplot(fig)

    # =========================================================
    # 2. МНОГОФАКТОРНЫЕ КОРРЕЛЯЦИОННЫЕ ЗАВИСИМОСТИ
    # =========================================================
    st.markdown("## Многофакторные корреляционные зависимости")

    results = []

    for target in NUM_COLS:
        # 1. факторы, сильно связанные с target
        candidates = [
            col for col in NUM_COLS
            if col != target
            and abs(corr.loc[target, col]) >= TARGET_CORR_MIN
            and pvals.loc[target, col] < ALPHA
        ]

        independent_vars = []

        for cand in candidates:
            is_independent = True
            for chosen in independent_vars:
                if (
                    abs(corr.loc[cand, chosen]) > INDEP_CORR_MAX
                    or pvals.loc[cand, chosen] >= ALPHA
                ):
                    is_independent = False
                    break

            if is_independent:
                independent_vars.append(cand)

        if len(independent_vars) >= 2:
            r_multi = multiple_correlation(
                df[target].values,
                df[independent_vars].values
            )

            results.append({
                "dependent_variable": target,
                "independent_variables": ", ".join(independent_vars),
                "multiple_correlation": r_multi
            })

    if results:
        res_df = pd.DataFrame(results).sort_values(
            "multiple_correlation", ascending=False
        )
        st.dataframe(res_df.round(4))
    else:
        st.info("Подходящих многофакторных зависимостей не обнаружено.")

    st.markdown("""
    **Интерпретация:**  
    Многофакторная корреляционная зависимость выявляется в случае,
    когда несколько параметров обладают сильной связью с исследуемой величиной,
    но при этом слабо коррелируют между собой, что позволяет рассматривать их
    как независимые факторы.
    """)
