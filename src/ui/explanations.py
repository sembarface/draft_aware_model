IGNORE_EXPLANATION_FEATURES = {
    "candidate_hero_id",
    "acting_team_id",
    "opponent_team_id",
    "patch",
}

HIGH_TEAM_PICK_RATE = 0.08
HIGH_TEAM_BAN_AGAINST_RATE = 0.08
HIGH_TEAM_CONTESTED_RATE = 0.12
HIGH_TEAM_EARLY_PICK_RATE = 0.50
MIN_TEAM_COUNT = 2

HIGH_ROLE_FIT = 0.6
HIGH_ROLE_CONFLICT = 1.2
HIGH_CORE_FIT = 0.8
HIGH_SUPPORT_FIT = 0.8
HIGH_FLEX_SCORE = 0.7


def _num(value, default=0.0):
    try:
        if value is None:
            return default
        number = float(value)
        if number != number:
            return default
        return number
    except Exception:
        return default


def _pct(value):
    return f"{_num(value) * 100:.1f}%"


def _games_text(value):
    games = int(round(_num(value)))
    if games % 10 == 1 and games % 100 != 11:
        word = "игра"
    elif games % 10 in {2, 3, 4} and games % 100 not in {12, 13, 14}:
        word = "игры"
    else:
        word = "игр"
    return f"{games} {word}"


def _add(items, value):
    if value:
        items.append(value)


def _first_existing(row, names, default=0.0):
    for name in names:
        if name in IGNORE_EXPLANATION_FEATURES:
            continue
        value = row.get(name)
        if value is not None:
            return value
    return default


def _rate_signal(row, names, label, threshold):
    value = _num(_first_existing(row, names))
    if value <= threshold:
        return None
    return f"{label}: {_pct(value)}"


def _delta_signal(row, names, label, threshold=0.0):
    value = _num(_first_existing(row, names))
    if value <= threshold:
        return None
    return f"{label}: {value:.3f}"


def _player_signal(row, prefix, scope, team_text):
    games = _num(row.get(f"{prefix}best_player_hero_games_{scope}"))
    if games <= 0:
        return None

    name = row.get(f"{prefix}best_player_hero_name_{scope}") or "один из игроков"
    winrate = row.get(f"{prefix}best_player_hero_winrate_{scope}")
    kda = row.get(f"{prefix}best_player_hero_avg_kda_{scope}")
    scope_text = "в текущем патче" if scope == "patch" else "по всей доступной статистике"
    winrate_value = _num(winrate)

    if games >= 3 and winrate_value >= 0.55:
        return (
            f"сильный comfort-сигнал: {team_text} {name} имеет {_games_text(games)} "
            f"на этом герое {scope_text}, winrate {_pct(winrate)}, KDA {_num(kda):.2f}"
        )
    if games >= 3:
        return (
            f"{team_text} {name}: большой опыт на этом герое {scope_text} "
            f"({_games_text(games)}), winrate {_pct(winrate)}, KDA {_num(kda):.2f}"
        )
    return (
        f"{team_text} {name}: {_games_text(games)} на этом герое {scope_text}, "
        f"winrate {_pct(winrate)}, KDA {_num(kda):.2f}"
    )


def meta_signals(row, action_type):
    signals = []
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_pick_rate", "candidate_pick_rate_alltime", "candidate_pick_rate_patch"],
            "героя часто выбирали",
            0.03,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_ban_rate", "candidate_ban_rate_alltime", "candidate_ban_rate_patch"],
            "героя часто банили",
            0.03 if action_type == "ban" else 0.05,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_pick_or_ban_rate", "candidate_pick_or_ban_rate_alltime", "candidate_pick_or_ban_rate_patch"],
            "высокий общий pick-or-ban rate",
            0.05,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_winrate", "candidate_winrate_alltime", "candidate_winrate_patch"],
            "высокий winrate героя",
            0.52,
        ),
    )

    matches = _num(_first_existing(row, ["candidate_matches_played", "candidate_matches", "candidate_games"]))
    if matches > 0:
        signals.append(f"статистика героя основана на {int(round(matches))} матчах")
    return signals


def interaction_signals(row, action_type):
    signals = []
    if action_type == "pick":
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_ally_synergy_mean", "candidate_ally_synergy_max"],
                "есть положительная синергия с уже выбранными союзниками",
            ),
        )
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_vs_enemy_counter_mean", "candidate_vs_enemy_counter_max"],
                "есть хороший матчап против уже выбранных героев соперника",
            ),
        )
    else:
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_enemy_synergy_mean", "candidate_enemy_synergy_max"],
                "герой может усилить уже выбранный драфт соперника",
            ),
        )
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_vs_ally_counter_mean", "candidate_vs_ally_counter_max"],
                "герой опасен против ваших уже выбранных героев",
            ),
        )
    return signals


def player_signals(row, action_type):
    signals = []
    if action_type == "pick":
        _add(signals, _player_signal(row, "own_", "patch", "игрок вашей команды"))
        _add(signals, _player_signal(row, "own_", "alltime", "игрок вашей команды"))
    else:
        _add(signals, _player_signal(row, "opponent_", "patch", "игрок соперника"))
        _add(signals, _player_signal(row, "opponent_", "alltime", "игрок соперника"))
    return signals


def team_priority_signals(row, action_type):
    signals = []
    prefix = "own_" if action_type == "pick" else "opponent_"
    team_text = "вашей команды" if action_type == "pick" else "соперника"
    against_text = "против вашей команды" if action_type == "pick" else "против соперника"

    pick_count = _num(row.get(f"{prefix}team_candidate_pick_count_patch"))
    pick_rate = _num(row.get(f"{prefix}team_candidate_pick_rate_patch"))
    ban_against_count = _num(row.get(f"{prefix}team_candidate_ban_against_count_patch"))
    ban_against_rate = _num(row.get(f"{prefix}team_candidate_ban_against_rate_patch"))
    contested_count = _num(row.get(f"{prefix}team_candidate_contested_count_patch"))
    contested_rate = _num(row.get(f"{prefix}team_candidate_contested_rate_patch"))
    early_pick_rate = _num(row.get(f"{prefix}team_candidate_early_pick_rate_patch"))

    if pick_count >= MIN_TEAM_COUNT and pick_rate >= HIGH_TEAM_PICK_RATE:
        signals.append(f"команда часто выбирала этого героя в текущем патче: {_pct(pick_rate)}")
    if ban_against_count >= MIN_TEAM_COUNT and ban_against_rate >= HIGH_TEAM_BAN_AGAINST_RATE:
        signals.append(f"этого героя часто банили {against_text}: {_pct(ban_against_rate)}")
    if contested_count >= MIN_TEAM_COUNT and contested_rate >= HIGH_TEAM_CONTESTED_RATE:
        signals.append(f"герой выглядит приоритетным для {team_text}: contested rate {_pct(contested_rate)}")
    if early_pick_rate >= HIGH_TEAM_EARLY_PICK_RATE and pick_count >= MIN_TEAM_COUNT:
        signals.append(f"когда герой доставался {team_text}, его часто забирали рано: {_pct(early_pick_rate)}")
    return signals


def role_signals(row, action_type):
    signals = []
    if action_type == "pick":
        role_fit = _num(row.get("candidate_own_role_fit_score"))
        core_fit = _num(row.get("candidate_own_core_fit"))
        support_fit = _num(row.get("candidate_own_support_fit"))
        conflict = _num(row.get("candidate_own_role_conflict_score"))

        if role_fit >= HIGH_ROLE_FIT:
            signals.append("герой хорошо закрывает недостающую роль в вашем текущем драфте")
        if core_fit >= HIGH_CORE_FIT:
            signals.append("герой может закрыть недостающую core-роль")
        if support_fit >= HIGH_SUPPORT_FIT:
            signals.append("герой может закрыть недостающую support-роль")
        if conflict >= HIGH_ROLE_CONFLICT:
            signals.append("есть риск дублирования уже закрытой роли")
    else:
        role_fit = _num(row.get("candidate_enemy_role_fit_score"))
        core_fit = _num(row.get("candidate_enemy_core_fit"))
        support_fit = _num(row.get("candidate_enemy_support_fit"))

        if role_fit >= HIGH_ROLE_FIT:
            signals.append("герой может закрыть недостающую роль в драфте соперника")
        if core_fit >= HIGH_CORE_FIT:
            signals.append("герой может закрыть для соперника недостающую core-роль")
        if support_fit >= HIGH_SUPPORT_FIT:
            signals.append("герой может закрыть для соперника недостающую support-роль")

    if _num(row.get("candidate_flex_score")) >= HIGH_FLEX_SCORE:
        if action_type == "pick":
            signals.append("герой гибок по ролям и оставляет пространство для дальнейшего драфта")
        else:
            signals.append("герой гибок по ролям и может упростить сопернику дальнейший драфт")
    return signals


def grouped_signals(row, action_type):
    groups = [
        ("Игроки", player_signals(row, action_type), 2),
        ("Роль в драфте", role_signals(row, action_type), 4),
        ("Приоритет команды", team_priority_signals(row, action_type), 4),
        ("Мета героя", meta_signals(row, action_type), 4),
        ("Синергия и матчапы", interaction_signals(row, action_type), 2),
    ]
    return [(title, signals[:limit]) for title, signals, limit in groups if signals]


def explain_recommendation(row, action_type):
    groups = grouped_signals(row, action_type)
    if not groups:
        return (
            "Положительные факторы: нет одного доминирующего понятного фактора; "
            "рекомендация основана на суммарной оценке ranker-модели."
        )

    parts = []
    for title, signals in groups[:4]:
        parts.append(f"{title}: " + "; ".join(signals[:2]))
    return "Положительные факторы: " + " | ".join(parts)


def explain_recommendation_markdown(row, action_type):
    groups = grouped_signals(row, action_type)
    if not groups:
        return (
            "### Почему герой в рекомендации\n\n"
            "- Нет одного доминирующего понятного фактора. Рекомендация основана на суммарной оценке ranker-модели."
        )

    blocks = ["### Почему герой в рекомендации"]
    for title, signals in groups:
        section = [f"**{title}**"]
        section.extend(f"- {signal}" for signal in signals)
        blocks.append("\n".join(section))
    return "\n\n".join(blocks)
