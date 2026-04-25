import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import PATCH_LABEL, get_patch_paths


HEROES_CSV = Path("data") / "heroes.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert raw OpenDota JSON files to parquet tables.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild parquet tables from all JSON files.")
    parser.add_argument("--only-new", action="store_true", help="Convert only JSON files absent in matches.parquet.")
    parser.add_argument("--validate", action="store_true", help="Validate basic JSON structure before conversion.")
    parser.add_argument("--strict", action="store_true", help="Raise an error on invalid JSON when validating.")
    return parser.parse_args(argv)


def safe_read_parquet(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pq.read_table(path).to_pandas()
    except Exception:
        return pd.read_parquet(path)


def safe_write_parquet(df, path):
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def load_heroes_df(path=HEROES_CSV):
    if not path.exists():
        return pd.DataFrame(columns=["hero_id", "hero_name"])
    heroes = pd.read_csv(path)
    return heroes.rename(columns={"id": "hero_id", "name": "hero_name"})[
        ["hero_id", "hero_name"]
    ]


def load_heroes_map(path=HEROES_CSV):
    heroes = load_heroes_df(path)
    return dict(zip(heroes["hero_id"], heroes["hero_name"]))


def validate_match_obj(match_obj, expected_patch_num=None):
    if not isinstance(match_obj, dict):
        return False, "json_is_not_object"
    if not match_obj.get("match_id"):
        return False, "missing_match_id"
    if expected_patch_num is not None and match_obj.get("patch") != expected_patch_num:
        return False, "wrong_json_patch"
    if len(match_obj.get("players") or []) != 10:
        return False, "players_len_not_10"
    if len(match_obj.get("picks_bans") or []) == 0:
        return False, "missing_picks_bans"
    return True, "ok"


def _player_side(player_slot):
    return "radiant" if player_slot is not None and player_slot < 128 else "dire"


def _team_context(match_obj, side):
    radiant_win = match_obj.get("radiant_win")
    if side == "radiant":
        win = bool(radiant_win) if radiant_win is not None else None
        return (
            match_obj.get("radiant_team_id"),
            match_obj.get("radiant_name") or match_obj.get("radiant_team_name"),
            win,
        )

    win = (not bool(radiant_win)) if radiant_win is not None else None
    return (
        match_obj.get("dire_team_id"),
        match_obj.get("dire_name") or match_obj.get("dire_team_name"),
        win,
    )


def process_match_json(match_obj, heroes_map):
    match_id = match_obj.get("match_id")
    duration = match_obj.get("duration")

    radiant_heroes = []
    dire_heroes = []

    for player in match_obj.get("players", []):
        side = _player_side(player.get("player_slot"))
        if side == "radiant":
            radiant_heroes.append(player.get("hero_id"))
        else:
            dire_heroes.append(player.get("hero_id"))

    match_row = {
        "match_id": match_id,
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
        "league_id": match_obj.get("leagueid"),
        "league_name": (match_obj.get("league") or {}).get("name"),
        "series_id": match_obj.get("series_id"),
        "series_type": match_obj.get("series_type"),
        "game_mode": match_obj.get("game_mode"),
        "lobby_type": match_obj.get("lobby_type"),
    }

    for i in range(5):
        match_row[f"radiant_hero_{i + 1}"] = radiant_heroes[i] if i < len(radiant_heroes) else None
        match_row[f"dire_hero_{i + 1}"] = dire_heroes[i] if i < len(dire_heroes) else None

    players_rows = []
    duration_min = duration / 60 if duration else None

    for player in match_obj.get("players", []):
        player_slot = player.get("player_slot")
        side = _player_side(player_slot)
        is_radiant = side == "radiant"
        team_id, team_name, win = _team_context(match_obj, side)
        kills = player.get("kills") or 0
        kills_per_min = player.get("kills_per_min")
        if kills_per_min is None and duration_min:
            kills_per_min = kills / duration_min

        players_rows.append({
            "match_id": match_id,
            "account_id": player.get("account_id"),
            "nickname": player.get("name"),
            "player_slot": player_slot,
            "is_radiant": is_radiant,
            "side": side,
            "team_id": team_id,
            "team_name": team_name,
            "win": win,
            "hero_id": player.get("hero_id"),
            "hero_name": heroes_map.get(player.get("hero_id")),
            "kills": kills,
            "deaths": player.get("deaths"),
            "assists": player.get("assists"),
            "last_hits": player.get("last_hits"),
            "denies": player.get("denies"),
            "teamfight_participation": player.get("teamfight_participation"),
            "level": player.get("level"),
            "kills_per_min": kills_per_min,
            "net_worth": player.get("net_worth"),
            "gold_per_min": player.get("gold_per_min") or player.get("gpm"),
            "xp_per_min": player.get("xp_per_min") or player.get("xpm"),
            "hero_damage": player.get("hero_damage"),
            "tower_damage": player.get("tower_damage"),
            "hero_healing": player.get("hero_healing"),
        })

    picks_rows = []
    draft_timings = match_obj.get("draft_timings") or []
    if not isinstance(draft_timings, list):
        draft_timings = []
    timing_map = {
        timing.get("order"): timing.get("total_time_taken")
        for timing in draft_timings
        if isinstance(timing, dict)
    }

    for pick_ban in match_obj.get("picks_bans", []):
        raw_order = pick_ban.get("order")
        saved_order = raw_order + 1 if raw_order is not None else None
        time_taken = timing_map.get(saved_order)
        if time_taken is None:
            time_taken = timing_map.get(raw_order)

        hero_id = pick_ban.get("hero_id")
        picks_rows.append({
            "match_id": match_id,
            "order": saved_order,
            "is_pick": pick_ban.get("is_pick"),
            "hero_id": hero_id,
            "hero_name": heroes_map.get(hero_id),
            "team": pick_ban.get("team"),
            "total_time_taken": time_taken,
        })

    return match_row, players_rows, picks_rows


def build_heroes_stats(players, picks, matches):
    heroes = load_heroes_df(HEROES_CSV)
    total_matches = matches["match_id"].nunique() if not matches.empty else 0

    if players.empty:
        hero_base = pd.DataFrame(columns=["hero_id", "matches_played", "wins"])
    else:
        hero_base = (
            players.groupby("hero_id")
            .agg(
                matches_played=("match_id", "nunique"),
                wins=("win", "sum"),
            )
            .reset_index()
        )

    all_hero_ids = set(hero_base["hero_id"].dropna().astype(int).tolist())
    if not heroes.empty:
        all_hero_ids.update(heroes["hero_id"].dropna().astype(int).tolist())

    result = pd.DataFrame({"hero_id": sorted(all_hero_ids)})
    if not heroes.empty:
        result = result.merge(heroes, on="hero_id", how="left")
    else:
        result["hero_name"] = None

    if not players.empty and "hero_name" in players.columns:
        played_names = (
            players[["hero_id", "hero_name"]]
            .dropna()
            .drop_duplicates("hero_id")
        )
        result = result.merge(
            played_names.rename(columns={"hero_name": "played_hero_name"}),
            on="hero_id",
            how="left",
        )
        result["hero_name"] = result["hero_name"].fillna(result["played_hero_name"])
        result = result.drop(columns=["played_hero_name"])

    result = result.merge(hero_base, on="hero_id", how="left")
    result["matches_played"] = result["matches_played"].fillna(0).astype(int)
    result["wins"] = result["wins"].fillna(0).astype(int)
    denominator = result["matches_played"].where(result["matches_played"] > 0)
    result["winrate"] = result["wins"] / denominator

    if total_matches > 0 and not picks.empty:
        picks_count = picks[picks["is_pick"]].groupby("hero_id").size()
        bans_count = picks[~picks["is_pick"]].groupby("hero_id").size()
        result["pick_rate"] = result["hero_id"].map(picks_count).fillna(0) / total_matches
        result["ban_rate"] = result["hero_id"].map(bans_count).fillna(0) / total_matches
    else:
        result["pick_rate"] = 0.0
        result["ban_rate"] = 0.0

    result["pick_or_ban_rate"] = result["pick_rate"] + result["ban_rate"]
    return result[
        [
            "hero_id",
            "hero_name",
            "matches_played",
            "wins",
            "winrate",
            "pick_rate",
            "ban_rate",
            "pick_or_ban_rate",
        ]
    ]


def _read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_conversion(
    patch_label=PATCH_LABEL,
    rebuild=False,
    only_new=False,
    validate=False,
    strict=False,
):
    if rebuild and only_new:
        raise ValueError("--rebuild and --only-new cannot be used together")

    patch_num, base_dir, match_dir, _, _ = get_patch_paths(patch_label)
    base_dir.mkdir(parents=True, exist_ok=True)

    players_file = base_dir / "players.parquet"
    matches_file = base_dir / "matches.parquet"
    picks_file = base_dir / "picks_bans.parquet"
    hero_stats_file = base_dir / "heroes_stats.parquet"

    if not match_dir.exists():
        print(f"No match directory: {match_dir}")
        return {}

    if rebuild:
        existing_matches = pd.DataFrame()
        existing_players = pd.DataFrame()
        existing_picks = pd.DataFrame()
        existing_ids = set()
    else:
        existing_matches = safe_read_parquet(matches_file)
        existing_players = safe_read_parquet(players_file)
        existing_picks = safe_read_parquet(picks_file)
        existing_ids = (
            set(existing_matches["match_id"].dropna().astype(int))
            if not existing_matches.empty
            else set()
        )

    heroes_map = load_heroes_map(HEROES_CSV)
    new_matches = []
    new_players = []
    new_picks = []
    skipped_existing = 0
    skipped_invalid = 0

    for path in sorted(match_dir.glob("*.json")):
        match_obj = _read_json(path)
        match_id = match_obj.get("match_id")

        if not rebuild and match_id in existing_ids:
            skipped_existing += 1
            continue

        if validate or strict:
            is_valid, reason = validate_match_obj(match_obj, expected_patch_num=patch_num)
            if not is_valid:
                if strict:
                    raise ValueError(f"Invalid JSON {path}: {reason}")
                skipped_invalid += 1
                continue

        match_row, players_rows, picks_rows = process_match_json(match_obj, heroes_map)
        new_matches.append(match_row)
        new_players.extend(players_rows)
        new_picks.extend(picks_rows)

    final_matches = pd.concat([existing_matches, pd.DataFrame(new_matches)], ignore_index=True)
    final_players = pd.concat([existing_players, pd.DataFrame(new_players)], ignore_index=True)
    final_picks = pd.concat([existing_picks, pd.DataFrame(new_picks)], ignore_index=True)

    if not final_matches.empty:
        final_matches = final_matches.drop_duplicates("match_id", keep="last")
    if not final_players.empty:
        final_players = final_players.drop_duplicates(["match_id", "account_id", "player_slot"], keep="last")
    if not final_picks.empty:
        final_picks = final_picks.drop_duplicates(["match_id", "order"], keep="last")

    heroes_stats = build_heroes_stats(final_players, final_picks, final_matches)

    safe_write_parquet(final_matches, matches_file)
    safe_write_parquet(final_players, players_file)
    safe_write_parquet(final_picks, picks_file)
    safe_write_parquet(heroes_stats, hero_stats_file)

    print(f"patch_label={patch_label}, patch_num={patch_num}")
    print(f"converted_new_matches={len(new_matches)}")
    print(f"skipped_existing={skipped_existing}, skipped_invalid={skipped_invalid}")
    print(f"matches: {final_matches.shape}, unique_matches={final_matches['match_id'].nunique() if not final_matches.empty else 0}")
    print(f"players: {final_players.shape}, unique_matches={final_players['match_id'].nunique() if not final_players.empty else 0}")
    print(f"picks_bans: {final_picks.shape}, unique_matches={final_picks['match_id'].nunique() if not final_picks.empty else 0}")
    print(f"heroes_stats: {heroes_stats.shape}")

    return {
        "patch_num": patch_num,
        "matches": final_matches.shape,
        "players": final_players.shape,
        "picks_bans": final_picks.shape,
        "heroes_stats": heroes_stats.shape,
    }


def main(argv=None):
    args = parse_args(argv)
    return run_conversion(
        patch_label=args.patch_label,
        rebuild=args.rebuild,
        only_new=args.only_new,
        validate=args.validate,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
