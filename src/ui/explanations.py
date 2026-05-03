IGNORE_EXPLANATION_FEATURES = {
    "candidate_hero_id",
    "acting_team_id",
    "opponent_team_id",
    "patch",
}


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
    name = row.get(f"{prefix}best_player_hero_name_{scope}") or "одного из игроков"
    winrate = row.get(f"{prefix}best_player_hero_winrate_{scope}")
    kda = row.get(f"{prefix}best_player_hero_avg_kda_{scope}")
    scope_text = "в текущем патче" if scope == "patch" else "в доступной прошлой статистике"
    return (
        f"{team_text} {name}: {int(round(games))} матч(ей) на этом герое {scope_text}, "
        f"winrate {_pct(winrate)}, KDA {_num(kda):.2f}"
    )


def meta_signals(row, action_type):
    signals = []
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_pick_rate", "candidate_pick_rate_alltime", "candidate_pick_rate_patch"],
            "герой часто выбирался до этого драфта",
            0.03,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_ban_rate", "candidate_ban_rate_alltime", "candidate_ban_rate_patch"],
            "герой часто банился до этого драфта",
            0.03 if action_type == "ban" else 0.05,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_pick_or_ban_rate", "candidate_pick_or_ban_rate_alltime", "candidate_pick_or_ban_rate_patch"],
            "у героя высокий pick-or-ban rate до этого драфта",
            0.05,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_winrate", "candidate_winrate_alltime", "candidate_winrate_patch"],
            "у героя высокий winrate до этого драфта",
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
                "положительная синергия с уже выбранными союзниками",
            ),
        )
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_vs_enemy_counter_mean", "candidate_vs_enemy_counter_max"],
                "хороший matchup против уже выбранных героев соперника",
            ),
        )
    else:
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_enemy_synergy_mean", "candidate_enemy_synergy_max"],
                "может усилить уже выбранный драфт соперника",
            ),
        )
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_vs_ally_counter_mean", "candidate_vs_ally_counter_max"],
                "опасен против ваших уже выбранных героев",
            ),
        )
    return signals


def player_signals(row, action_type):
    signals = []
    if action_type == "pick":
        _add(signals, _player_signal(row, "own_", "patch", "у игрока вашей команды"))
        _add(signals, _player_signal(row, "own_", "alltime", "у игрока вашей команды"))
    else:
        _add(signals, _player_signal(row, "opponent_", "patch", "у игрока соперника"))
        _add(signals, _player_signal(row, "opponent_", "alltime", "у игрока соперника"))
    return signals


def explain_recommendation(row, action_type):
    meta = meta_signals(row, action_type)
    players = player_signals(row, action_type)
    interactions = interaction_signals(row, action_type)
    parts = []

    if meta:
        parts.append("Мета героя: " + "; ".join(meta[:3]))
    if players:
        parts.append("Игроки: " + "; ".join(players[:2]))
    if interactions:
        parts.append("Состояние драфта: " + "; ".join(interactions[:2]))

    if not parts:
        parts.append("нет одного доминирующего понятного фактора; рекомендация основана на суммарной оценке ranker-модели")

    return "Положительные факторы: " + " | ".join(parts)


def explain_recommendation_markdown(row, action_type):
    meta = meta_signals(row, action_type)
    players = player_signals(row, action_type)
    interactions = interaction_signals(row, action_type)
    blocks = ["**Положительные факторы**"]

    if meta:
        blocks.append("**Мета героя.** " + "; ".join(meta[:4]) + ".")
    if players:
        blocks.append("**Игроки.** " + "; ".join(players[:2]) + ".")
    if interactions:
        blocks.append("**Состояние драфта.** " + "; ".join(interactions[:2]) + ".")

    if len(blocks) == 1:
        blocks.append(
            "Нет одного доминирующего понятного фактора. "
            "Рекомендация основана на суммарной оценке ranker-модели."
        )

    return "\n\n".join(blocks)
