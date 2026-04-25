import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    PATCH_LABEL,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    TEAM_IDS,
    get_patch_paths,
)


EXPLORER_URL = "https://api.opendota.com/api/explorer"
MATCH_URL = "https://api.opendota.com/api/matches/{match_id}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch OpenDota match JSON files.")
    parser.add_argument("--patch-label", default=PATCH_LABEL, help="OpenDota patch label, e.g. 7.41.")
    parser.add_argument("--limit", type=int, default=None, help="Download only first N matches.")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Download JSON again even if it already exists locally.",
    )
    parser.add_argument(
        "--allow-incomplete-draft",
        action="store_true",
        help="Allow matches where picks_bans length is not 24.",
    )
    parser.add_argument(
        "--allow-wrong-json-patch",
        action="store_true",
        help="Allow JSON patch id that does not match PATCH_MAP.",
    )
    parser.add_argument(
        "--sleep",
        "--delay",
        dest="delay",
        type=float,
        default=REQUEST_DELAY,
        help="Delay between match API requests.",
    )
    return parser.parse_args(argv)


def get_top_teams(n=30, timeout=REQUEST_TIMEOUT):
    """Optional helper for EDA; the parser defaults to manual TEAM_IDS."""
    response = requests.get("https://api.opendota.com/api/teams", timeout=timeout)
    response.raise_for_status()
    teams = pd.DataFrame(response.json())
    return teams.sort_values("rating", ascending=False).head(n)["team_id"].tolist()


def build_sql(patch_label, team_ids):
    team_ids_sql = ", ".join(str(int(team_id)) for team_id in team_ids)
    return f"""
        SELECT DISTINCT
            matches.match_id,
            matches.start_time,
            matches.radiant_team_id,
            matches.dire_team_id,
            leagues.name AS league_name
        FROM matches
        JOIN match_patch USING(match_id)
        JOIN leagues USING(leagueid)
        WHERE TRUE
          AND match_patch.patch = '{patch_label}'
          AND leagues.tier = 'professional'
          AND (
              matches.radiant_team_id IN ({team_ids_sql})
              OR matches.dire_team_id IN ({team_ids_sql})
          )
        ORDER BY matches.start_time DESC NULLS LAST
    """


def fetch_match_rows(sql, timeout=REQUEST_TIMEOUT):
    response = requests.get(EXPLORER_URL, params={"sql": sql}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("rows", [])


def fetch_match_json(match_id, timeout=REQUEST_TIMEOUT):
    response = requests.get(MATCH_URL.format(match_id=match_id), timeout=timeout)
    response.raise_for_status()
    return response.json()


def _team_in_set(team_id, team_ids):
    if team_id is None or pd.isna(team_id):
        return False
    return int(team_id) in team_ids


def validate_match_json(
    data,
    patch_num,
    team_ids,
    allow_incomplete_draft=False,
    allow_wrong_json_patch=False,
):
    if not isinstance(data, dict):
        return False, "json_is_not_object"

    team_ids = set(int(team_id) for team_id in team_ids)
    json_patch = data.get("patch")
    radiant_team_id = data.get("radiant_team_id")
    dire_team_id = data.get("dire_team_id")
    picks_bans_len = len(data.get("picks_bans") or [])
    players_len = len(data.get("players") or [])

    if not allow_wrong_json_patch and json_patch != patch_num:
        return False, "wrong_json_patch"

    if not (
        _team_in_set(radiant_team_id, team_ids)
        or _team_in_set(dire_team_id, team_ids)
    ):
        return False, "team_not_in_manual_team_ids"

    if not allow_incomplete_draft and picks_bans_len != 24:
        return False, "incomplete_draft"

    if players_len != 10:
        return False, "players_len_not_10"

    return True, "ok"


def make_log_row(match_id, status, reason, data=None):
    data = data or {}
    picks_bans = data.get("picks_bans") if isinstance(data, dict) else None
    return {
        "match_id": match_id,
        "status": status,
        "reason": reason,
        "radiant_team_id": data.get("radiant_team_id") if isinstance(data, dict) else None,
        "dire_team_id": data.get("dire_team_id") if isinstance(data, dict) else None,
        "json_patch": data.get("patch") if isinstance(data, dict) else None,
        "picks_bans_len": len(picks_bans or []) if isinstance(data, dict) else None,
    }


def write_parser_log(path, rows):
    if not rows:
        return
    new_log = pd.DataFrame(rows)
    if path.exists():
        old_log = pd.read_csv(path)
        new_log = pd.concat([old_log, new_log], ignore_index=True)
    new_log.to_csv(path, index=False)


def run_parser(
    patch_label=PATCH_LABEL,
    limit=None,
    refresh_existing=False,
    allow_incomplete_draft=False,
    allow_wrong_json_patch=False,
    delay=REQUEST_DELAY,
):
    patch_num, base_dir, match_dir, _, _ = get_patch_paths(patch_label)
    base_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    sql = build_sql(patch_label, TEAM_IDS)
    rows = fetch_match_rows(sql)
    match_rows = rows[:limit] if limit is not None else rows

    match_ids_path = base_dir / f"match_ids_patch_{patch_num}.csv"
    pd.DataFrame(match_rows).to_csv(match_ids_path, index=False)

    log_rows = []

    for row in match_rows:
        match_id = row.get("match_id")
        out_path = match_dir / f"{match_id}.json"

        if out_path.exists() and not refresh_existing:
            log_rows.append(make_log_row(match_id, "skipped_existing", "json_exists"))
            continue

        try:
            data = fetch_match_json(match_id)
        except Exception as exc:
            log_rows.append(make_log_row(match_id, "failed", str(exc)))
            continue

        is_valid, reason = validate_match_json(
            data=data,
            patch_num=patch_num,
            team_ids=TEAM_IDS,
            allow_incomplete_draft=allow_incomplete_draft,
            allow_wrong_json_patch=allow_wrong_json_patch,
        )

        if not is_valid:
            log_rows.append(make_log_row(match_id, "rejected", reason, data))
            time.sleep(delay)
            continue

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log_rows.append(make_log_row(match_id, "saved", "ok", data))
        time.sleep(delay)

    log_path = base_dir / f"parser_log_patch_{patch_num}.csv"
    write_parser_log(log_path, log_rows)

    saved = sum(row["status"] == "saved" for row in log_rows)
    skipped = sum(row["status"] == "skipped_existing" for row in log_rows)
    rejected = sum(row["status"] == "rejected" for row in log_rows)
    failed = sum(row["status"] == "failed" for row in log_rows)

    print(f"patch_label={patch_label}, patch_num={patch_num}")
    print(f"match_ids: {len(match_rows)} -> {match_ids_path}")
    print(f"saved={saved}, skipped_existing={skipped}, rejected={rejected}, failed={failed}")
    print(f"log: {log_path}")

    return {
        "patch_num": patch_num,
        "match_dir": match_dir,
        "match_ids": len(match_rows),
        "saved": saved,
        "skipped_existing": skipped,
        "rejected": rejected,
        "failed": failed,
    }


def main(argv=None):
    args = parse_args(argv)
    return run_parser(
        patch_label=args.patch_label,
        limit=args.limit,
        refresh_existing=args.refresh_existing,
        allow_incomplete_draft=args.allow_incomplete_draft,
        allow_wrong_json_patch=args.allow_wrong_json_patch,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
