# src/gui/data_loader.py
import pandas as pd
from pathlib import Path

def load_data(patch: int):
    base = Path(f"data/patch_{patch}")

    players = pd.read_parquet(base / "players.parquet")
    matches = pd.read_parquet(base / "matches.parquet")
    picks = pd.read_parquet(base / "picks_bans.parquet")
    heroes_stats = pd.read_parquet(base / "heroes_stats.parquet")

    return players, matches, picks, heroes_stats
