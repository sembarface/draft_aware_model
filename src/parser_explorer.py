import requests
import json
import os
import time
from pathlib import Path
import pandas as pd

# =====================================================
# НАСТРОЙКИ — МЕНЯЙ ЗДЕСЬ
# =====================================================

# Патч в формате как в Explorer (строка)
PATCH_FROM = "7.41"
PATCH_TO = "7.41"

# Патч в формате JSON (число OpenDota)
PATCH_NUM = 60

# Команды
USE_TOP_TEAMS = False
TOP_N_TEAMS = 30

# Если USE_TOP_TEAMS = False, сюда внесёшь команд вручную:
TEAM_IDS = [7119388,    9247354,    8291895,    9572001,    2163,   8255888,    9823272,    2586976,    9338413,    8261500,    36,     8599101,    9640842,     9467224     ]
#           Spirit      Falcons     Tundra      PARI        Liquid  BB Team     Yandex      OG          MOUZ        Xtreme      NaVi    GG          Tidebound    Aurora
# Директория сохранения матчей
OUTPUT_DIR = Path(f"data/patch_{PATCH_NUM}/matches/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def get_top_teams(n=TOP_N_TEAMS):
    url = "https://api.opendota.com/api/teams"
    r = requests.get(url)
    arr = r.json()
    df = pd.DataFrame(arr)
    df = df.sort_values("rating", ascending=False).head(n)
    return df["team_id"].tolist()


def build_sql(p_from, p_to, team_ids):
    team_filter = " OR ".join([f"notable_players.team_id = {tid}" for tid in team_ids])

    sql = f"""
        SELECT DISTINCT
            matches.match_id
        FROM matches
        JOIN match_patch USING(match_id)
        JOIN leagues USING(leagueid)
        JOIN player_matches USING(match_id)
        LEFT JOIN notable_players 
               ON notable_players.account_id = player_matches.account_id
        WHERE TRUE
          AND match_patch.patch >= '{p_from}'
          AND match_patch.patch <= '{p_to}'
          AND leagues.tier = 'professional'
          AND ({team_filter})
        ORDER BY matches.match_id DESC NULLS LAST
    """

    return sql


def fetch_match_ids(sql):
    url = "https://api.opendota.com/api/explorer"
    r = requests.get(url, params={"sql": sql})
    if r.status_code != 200:
        print("SQL:", sql)
        raise RuntimeError(f"Explorer error: {r.text}")

    rows = r.json()["rows"]
    return [row["match_id"] for row in rows]


def fetch_match_json(match_id):
    url = f"https://api.opendota.com/api/matches/{match_id}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Ошибка загрузки матча {match_id}")
        return None
    return r.json()


# =====================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =====================================================

def main():
    print("=== ПАРСИНГ ЧЕРЕЗ OPEN DOTA EXPLORER ===")

    if USE_TOP_TEAMS:
        team_ids = get_top_teams(TOP_N_TEAMS)
    else:
        team_ids = TEAM_IDS

    print("Команды:", team_ids)

    print("\nГенерируем SQL...")
    sql = build_sql(PATCH_FROM, PATCH_TO, team_ids)

    print("\nЗапрашиваем match_id...")
    match_ids = fetch_match_ids(sql)
    print(f"Найдено матчей: {len(match_ids)}")

    saved = 0

    for mid in match_ids:
        out = OUTPUT_DIR / f"{mid}.json"
        if out.exists():
            continue

        data = fetch_match_json(mid)
        if not data:
            continue

        # фильтруем по JSON патчу (числовой)
        if data.get("patch") != PATCH_NUM:
            continue

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        saved += 1
        print(f"[saved] {mid}")
        time.sleep(0.3)

    print("\n=== ГОТОВО ===")
    print(f"Сохранено новых матчей: {saved}")


if __name__ == "__main__":
    main()
