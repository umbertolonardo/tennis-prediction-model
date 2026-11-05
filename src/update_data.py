"""Rebuild the features with TennisMyLife's newer matches, in a separate database.

tennis.db stays as it is, so every earlier notebook keeps its results. Run from the project folder:
    python src/update_data.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis_data import load_all_matches  # noqa: E402
from tennis_domain_features import DOMAIN_GROUPS, build_domain_features  # noqa: E402
from tennis_elo import run_elo  # noqa: E402
from tennis_features import build_features  # noqa: E402
from tennis_predict import ELO_SETTINGS, ELO_WEIGHT  # noqa: E402
from tennis_tml import link_players, players_with_new  # noqa: E402

RAW = ROOT / "data/raw"
OLD_DB = ROOT / "data/processed/tennis.db"
NEW_DB = ROOT / "data/processed/tennis_latest.db"


def largest_change(old, new):
    """Compare every old match with its new version. Adding later matches mustn't change the past."""
    keys = ["tourney_id", "match_num"]
    both = old.merge(new, on=keys, how="left", suffixes=("_old", "_new"), validate="one_to_one", indicator=True)
    largest = 0.0
    for column in [c for c in old.columns if c not in keys and c in new.columns and pd.api.types.is_numeric_dtype(old[c])]:
        a, b = both[f"{column}_old"].to_numpy(dtype=float), both[f"{column}_new"].to_numpy(dtype=float)
        assert (np.isnan(a) == np.isnan(b)).all(), f"missing values changed in {column}"
        largest = max(largest, float(np.nanmax(np.abs(a - b))))
    return int((both["_merge"] != "both").sum()), largest


links = link_players(RAW)
matches = load_all_matches(RAW, include_tml=True)
players = players_with_new(RAW, links)
tournament_info = pd.read_csv(ROOT / "reference/tournament_info.csv")
print(f"{len(matches):,} matches, up to day {pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(matches['day'].max())):%Y-%m-%d}", flush=True)

ratings = run_elo(matches, **ELO_SETTINGS)
features = build_features(matches, ratings, players, tournament_info, ELO_WEIGHT)
domain = build_domain_features(matches, ratings, ELO_WEIGHT)
history = DOMAIN_GROUPS["Tournament history"]
final = features.merge(domain[["tourney_id", "match_num"] + history], on=["tourney_id", "match_num"], validate="one_to_one")

old = pd.read_sql_query("SELECT * FROM final_features", sqlite3.connect(OLD_DB))
missing, change = largest_change(old, final)
print(f"Old matches: {len(old):,}. Missing from the new table: {missing}. Largest change in any value: {change}", flush=True)
assert missing == 0 and change == 0, "the update changed past matches"

with sqlite3.connect(NEW_DB) as conn:
    final.to_sql("final_features", conn, if_exists="replace", index=False)
    links.to_sql("player_links", conn, if_exists="replace", index=False)
print(f"Saved {len(final):,} main-tour matches ({len(final) - len(old):,} new) to {NEW_DB.name}", flush=True)
