import os
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------
# Настройки
# ---------------------------
PATCH_NUM = 60
BASE_DIR = Path(f"data/patch_{PATCH_NUM}")
MATCH_DIR = BASE_DIR / "matches"

PLAYERS_FILE = BASE_DIR / "players.parquet"
MATCHES_FILE = BASE_DIR / "matches.parquet"
PICKS_FILE = BASE_DIR / "picks_bans.parquet"
HERO_STATS_FILE = BASE_DIR / "heroes_stats.parquet"

HEROES_CSV = Path("data") / "heroes.csv"   # id,name

# ---------------------------
# Утилиты
# ---------------------------
def safe_read_parquet(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pq.read_table(path).to_pandas()
    except Exception:
        return pd.read_parquet(path)

def safe_write_parquet(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    table = pa.Table.from_pandas(df)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)

def load_heroes_map(path: Path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["id"], df["name"]))

# ---------------------------
# Обработка одного матча
# ---------------------------
def process_match_json(match_obj, heroes_map):
    mid = match_obj.get("match_id")
    duration = match_obj.get("duration")

    # -------- heroes per team --------
    radiant_heroes = []
    dire_heroes = []

    for p in match_obj.get("players", []):
        if p.get("player_slot", 0) < 128:
            radiant_heroes.append(p.get("hero_id"))
        else:
            dire_heroes.append(p.get("hero_id"))

    # -------- matches row --------
    match_row = {
        "match_id": mid,
        "start_time": match_obj.get("start_time"),
        "duration": duration,
        "radiant_win": match_obj.get("radiant_win"),
        "patch": match_obj.get("patch"),
        "radiant_score": match_obj.get("radiant_score"),
        "dire_score": match_obj.get("dire_score"),
        "radiant_team_id": match_obj.get("radiant_team_id"),
        "radiant_team_name": match_obj.get("radiant_name") or match_obj.get("radiant_team_name"),
        "dire_team_id": match_obj.get("dire_team_id"),
        "dire_team_name": match_obj.get("dire_name") or match_obj.get("dire_team_name"),
        "league_name": (match_obj.get("league") or {}).get("name"),
    }

    for i in range(5):
        match_row[f"radiant_hero_{i+1}"] = radiant_heroes[i] if i < len(radiant_heroes) else None
        match_row[f"dire_hero_{i+1}"] = dire_heroes[i] if i < len(dire_heroes) else None

    # -------- players --------
    players_rows = []
    dur_min = duration / 60 if duration else None

    for p in match_obj.get("players", []):
        kills = p.get("kills") or 0
        kpm = p.get("kills_per_min") or (kills / dur_min if dur_min else None)

        players_rows.append({
            "match_id": mid,
            "account_id": p.get("account_id"),
            "nickname": p.get("name"),
            "player_slot": p.get("player_slot"),
            "hero_id": p.get("hero_id"),
            "hero_name": heroes_map.get(p.get("hero_id")),
            "kills": kills,
            "deaths": p.get("deaths"),
            "assists": p.get("assists"),
            "last_hits": p.get("last_hits"),
            "denies": p.get("denies"),
            "teamfight_participation": p.get("teamfight_participation"),
            "level": p.get("level"),
            "kills_per_min": kpm,
            "net_worth": p.get("net_worth"),
            "gold_per_min": p.get("gold_per_min") or p.get("gpm"),
            "xp_per_min": p.get("xp_per_min") or p.get("xpm"),
            "hero_damage": p.get("hero_damage"),
            "tower_damage": p.get("tower_damage"),
            "hero_healing": p.get("hero_healing"),
        })

    # -------- picks & bans --------
    pbs_rows = []
    draft_timings = match_obj.get("draft_timings") or {}
    timing_map = {dt["order"] : dt.get("total_time_taken") for dt in draft_timings}

    for pb in match_obj.get("picks_bans", []):
        order = pb.get("order")
        pbs_rows.append({
            "match_id": mid,
            "order": order+1,
            "is_pick": pb.get("is_pick"),
            "hero_id": pb.get("hero_id"),
            "hero_name": heroes_map.get(pb.get("hero_id")),
            "team": pb.get("team"),
            "total_time_taken": timing_map.get(order+1),
        })

    return match_row, players_rows, pbs_rows

# ---------------------------
# Построение heroes_stats.parquet
# ---------------------------
def build_heroes_stats(players, picks, matches):
    df = players.merge(matches[["match_id", "radiant_win"]], on="match_id")

    df["win"] = (
        ((df["player_slot"] < 128) & df["radiant_win"]) |
        ((df["player_slot"] >= 128) & (~df["radiant_win"]))
    )

    hero_base = df.groupby(["hero_id", "hero_name"]).agg(
        matches_played=("match_id", "nunique"),
        wins=("win", "sum")
    ).reset_index()

    hero_base["winrate"] = hero_base["wins"] / hero_base["matches_played"]

    total_matches = matches["match_id"].nunique()

    picks_cnt = picks[picks["is_pick"]].groupby("hero_id").size()
    bans_cnt = picks[~picks["is_pick"]].groupby("hero_id").size()

    hero_base["pick_rate"] = hero_base["hero_id"].map(picks_cnt).fillna(0) / total_matches
    hero_base["ban_rate"] = hero_base["hero_id"].map(bans_cnt).fillna(0) / total_matches
    hero_base["pick_or_ban_rate"] = hero_base["pick_rate"] + hero_base["ban_rate"]

    return hero_base

# ---------------------------
# Main
# ---------------------------
def main():
    if not MATCH_DIR.exists():
        print(f"No match directory: {MATCH_DIR}")
        return

    heroes_map = load_heroes_map(HEROES_CSV)

    df_matches_existing = safe_read_parquet(MATCHES_FILE)
    df_players_existing = safe_read_parquet(PLAYERS_FILE)
    df_picks_existing = safe_read_parquet(PICKS_FILE)

    existing_ids = set(df_matches_existing["match_id"]) if not df_matches_existing.empty else set()

    new_matches, new_players, new_picks = [], [], []

    for path in MATCH_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as f:
            match = json.load(f)

        mid = match.get("match_id")
        if mid in existing_ids:
            continue

        m, p, pb = process_match_json(match, heroes_map)
        new_matches.append(m)
        new_players.extend(p)
        new_picks.extend(pb)

    if not new_matches:
        print("No new matches.")
        return

    final_matches = pd.concat([df_matches_existing, pd.DataFrame(new_matches)], ignore_index=True)
    final_players = pd.concat([df_players_existing, pd.DataFrame(new_players)], ignore_index=True)
    final_picks = pd.concat([df_picks_existing, pd.DataFrame(new_picks)], ignore_index=True)

    safe_write_parquet(final_matches, MATCHES_FILE)
    safe_write_parquet(final_players, PLAYERS_FILE)
    safe_write_parquet(final_picks, PICKS_FILE)

    heroes_stats = build_heroes_stats(final_players, final_picks, final_matches)
    safe_write_parquet(heroes_stats, HERO_STATS_FILE)

    print(f"Done. Total matches: {len(final_matches)}")

if __name__ == "__main__":
    main()
