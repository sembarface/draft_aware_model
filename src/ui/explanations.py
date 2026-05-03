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
    name = row.get(f"{prefix}best_player_hero_name_{scope}") or "one player"
    winrate = row.get(f"{prefix}best_player_hero_winrate_{scope}")
    kda = row.get(f"{prefix}best_player_hero_avg_kda_{scope}")
    scope_text = "in this patch" if scope == "patch" else "in available past data"
    return (
        f"{team_text} {name}: {int(round(games))} games on this hero {scope_text}, "
        f"winrate {_pct(winrate)}, KDA {_num(kda):.2f}"
    )


def meta_signals(row, action_type):
    signals = []
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_pick_rate", "candidate_pick_rate_alltime", "candidate_pick_rate_patch"],
            "pick rate before this draft",
            0.03,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_ban_rate", "candidate_ban_rate_alltime", "candidate_ban_rate_patch"],
            "ban rate before this draft",
            0.03 if action_type == "ban" else 0.05,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_pick_or_ban_rate", "candidate_pick_or_ban_rate_alltime", "candidate_pick_or_ban_rate_patch"],
            "pick-or-ban rate before this draft",
            0.05,
        ),
    )
    _add(
        signals,
        _rate_signal(
            row,
            ["candidate_winrate", "candidate_winrate_alltime", "candidate_winrate_patch"],
            "winrate before this draft",
            0.52,
        ),
    )
    matches = _num(_first_existing(row, ["candidate_matches_played", "candidate_matches", "candidate_games"]))
    if matches > 0:
        signals.append(f"sample size for hero stats: {int(round(matches))} matches")
    return signals


def interaction_signals(row, action_type):
    signals = []
    if action_type == "pick":
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_ally_synergy_mean", "candidate_ally_synergy_max"],
                "positive synergy with already picked allies",
            ),
        )
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_vs_enemy_counter_mean", "candidate_vs_enemy_counter_max"],
                "positive matchup against enemy picks",
            ),
        )
    else:
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_enemy_synergy_mean", "candidate_enemy_synergy_max"],
                "can strengthen the opponent draft",
            ),
        )
        _add(
            signals,
            _delta_signal(
                row,
                ["candidate_vs_ally_counter_mean", "candidate_vs_ally_counter_max"],
                "dangerous against your current picks",
            ),
        )
    return signals


def player_signals(row, action_type):
    signals = []
    if action_type == "pick":
        _add(signals, _player_signal(row, "own_", "patch", "your player"))
        _add(signals, _player_signal(row, "own_", "alltime", "your player"))
    else:
        _add(signals, _player_signal(row, "opponent_", "patch", "opponent player"))
        _add(signals, _player_signal(row, "opponent_", "alltime", "opponent player"))
    return signals


def explain_recommendation(row, action_type):
    parts = []
    meta = meta_signals(row, action_type)
    players = player_signals(row, action_type)
    interactions = interaction_signals(row, action_type)

    if meta:
        parts.append("Meta: " + "; ".join(meta[:3]))
    if players:
        parts.append("Players: " + "; ".join(players[:2]))
    if interactions:
        parts.append("Draft fit: " + "; ".join(interactions[:2]))

    if not parts:
        parts.append("No single readable factor dominates; the score comes from the combined ranker signal.")

    return "Positive factors: " + " | ".join(parts)
