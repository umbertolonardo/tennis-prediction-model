"""Load every match from all tour levels and put them in chronological order."""
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_tml import load_new_matches

SOURCES = {
    "main": "atp_matches_[0-9][0-9][0-9][0-9].csv",
    "qual_chall": "atp_matches_qual_chall_*.csv",
    "futures": "atp_matches_futures_*.csv",
}

COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_seed", "winner_entry", "winner_age", "winner_rank",
    "winner_rank_points", "loser_id", "loser_seed", "loser_entry", "loser_age", "loser_rank",
    "loser_rank_points", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]

ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "ER": 4, "RR": 5, "R128": 6, "R64": 7,
               "R32": 8, "R16": 9, "QF": 10, "SF": 11, "BR": 12, "F": 13}

# How many rounds each round is before the final
ROUNDS_BEFORE_FINAL = {"F": 0, "BR": 0, "SF": 1, "QF": 2, "R16": 3, "R32": 4, "R64": 5, "R128": 6}
# Longest believable duration: the 2010 Isner-Mahut match lasted 665 minutes
MAX_MINUTES = 700


def estimate_day_offset(df):
    """Estimate how many days after the tournament's start date each match was played."""
    slam = (df["tourney_level"] == "G") | (df["draw_size"] >= 128)
    big_event = df["draw_size"].between(65, 127)
    last_day = np.select([slam, big_event], [13, 10], default=6)
    round_gap = np.where(slam, 2, 1)

    before_final = df["round"].map(ROUNDS_BEFORE_FINAL)
    offset = pd.Series(last_day - before_final * round_gap, index=df.index)

    # Round robin matches sit in the first part of an event
    offset = offset.mask(df["round"].isin(["RR", "ER"]), 2)

    # Qualifying is played just before the main draw starts
    qualifying = df["round"].map({"Q1": -2, "Q2": -1, "Q3": -1, "Q4": -1})
    slam_qualifying = df["round"].map({"Q1": -6, "Q2": -4, "Q3": -2, "Q4": -1})
    offset = offset.mask(qualifying.notna(), qualifying)
    offset = offset.mask(qualifying.notna() & slam, slam_qualifying)

    # A Davis Cup tie is played over about three days
    offset = offset.mask(df["tourney_level"] == "D", ((df["match_num"] - 1) // 2).clip(0, 2))
    return offset.fillna(0).astype(int)


def load_all_matches(raw_dir, first_year=1990, include_tml=False):
    """Return all matches from every source, sorted in estimated playing order.

    With include_tml, also add the TennisMyLife matches played after Sackmann's last ones.
    """
    raw_dir = Path(raw_dir)
    frames = []
    for source, pattern in SOURCES.items():
        for path in sorted(raw_dir.glob(pattern)):
            frame = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
            frame["source"] = source
            frames.append(frame)
    if include_tml:
        frames.append(load_new_matches(raw_dir)[COLUMNS + ["source"]])
    df = pd.concat(frames, ignore_index=True)

    df["tourney_level"] = df["tourney_level"].astype(str)
    df["tourney_name"] = df["tourney_name"].astype(str).str.strip()
    df["round"] = df["round"].astype(str)
    df["start_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df[df["start_date"].dt.year >= first_year].copy()

    # Days since 1970 are faster to work with than dates
    df["day"] = (df["start_date"] - pd.Timestamp("1970-01-01")).dt.days + estimate_day_offset(df)
    df["year"] = df["start_date"].dt.year

    score = df["score"].astype(str)
    df["walkover"] = score.str.contains(r"W/O|Walkover|^WO$", regex=True) | df["score"].isna()
    df["ended_early"] = score.str.contains(r"RET|DEF|ABD|abandoned|unfinished", regex=True, case=False)

    is_qualifying = df["round"].str.startswith("Q")
    df["tier"] = np.select(
        [(df["source"] == "main"), (df["source"] == "futures"), is_qualifying],
        ["main", "futures", "qualifying"],
        default="challenger",
    )

    # Impossible durations are recording errors, so they are treated as unknown
    df.loc[df["minutes"] > MAX_MINUTES, "minutes"] = np.nan

    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(14)
    df = df.sort_values(
        ["day", "source", "tourney_id", "round_order", "match_num"], kind="stable"
    ).reset_index(drop=True)
    return df
