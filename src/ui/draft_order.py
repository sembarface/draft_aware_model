import pandas as pd


# Patch 60 Captain's Mode order inferred from local draft_events:
# actor is relative to the team that makes the first pick.
DRAFT_ORDER = [
    (1, "ban", "first"),
    (2, "ban", "first"),
    (3, "ban", "second"),
    (4, "ban", "second"),
    (5, "ban", "first"),
    (6, "ban", "second"),
    (7, "ban", "second"),
    (8, "pick", "first"),
    (9, "pick", "second"),
    (10, "ban", "first"),
    (11, "ban", "first"),
    (12, "ban", "second"),
    (13, "pick", "second"),
    (14, "pick", "first"),
    (15, "pick", "first"),
    (16, "pick", "second"),
    (17, "pick", "second"),
    (18, "pick", "first"),
    (19, "ban", "first"),
    (20, "ban", "second"),
    (21, "ban", "first"),
    (22, "ban", "second"),
    (23, "pick", "first"),
    (24, "pick", "second"),
]


def draft_phase(order):
    if order <= 8:
        return 1
    if order <= 16:
        return 2
    return 3


def get_draft_order(first_pick_team="own"):
    if first_pick_team not in {"own", "opponent"}:
        raise ValueError("first_pick_team must be 'own' or 'opponent'")
    first_role = first_pick_team
    second_role = "opponent" if first_pick_team == "own" else "own"
    rows = []
    for order, action_type, actor in DRAFT_ORDER:
        rows.append({
            "order": order,
            "action_type": action_type,
            "actor": actor,
            "team_role": first_role if actor == "first" else second_role,
            "draft_phase": draft_phase(order),
        })
    return rows


def get_empty_draft_table(first_pick_team, own_team, opponent_team):
    rows = []
    for row in get_draft_order(first_pick_team):
        team = own_team if row["team_role"] == "own" else opponent_team
        rows.append({
            "order": row["order"],
            "phase": row["draft_phase"],
            "action_type": row["action_type"],
            "actor": row["actor"],
            "team_role": row["team_role"],
            "team_name": team.get("team_name", "unknown"),
            "hero_id": None,
            "hero_name": None,
        })
    return pd.DataFrame(rows)
