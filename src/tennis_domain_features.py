"""Features from tennis knowledge: confidence, priorities, tournament history, comebacks, slumps.

Same rule as tennis_features.py: read each player's history first, update it afterwards.
"""
import bisect
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from tennis_elo import combined_difference, win_probability

BIG_TITLE_LEVELS = {"G", "M", "F"}

DOMAIN_GROUPS = {
    "Season confidence": ["d_season_win_share", "d_season_beat_expectation", "d_season_titles"],
    "Grand Slam priority": ["d_slams_this_year", "d_slam_winner_small_event", "d_top_before_slam"],
    "Top player in small event": ["d_top10_in_small_event"],
    "After a big title": ["d_big_title_recent"],
    "Tournament history": ["d_event_win_share", "d_event_titles", "d_defending_champion"],
    "End of season": ["d_late_season_rank"],
    "Comeback": ["d_early_in_comeback"],
    "Slump": ["d_opening_loss_streak"],
}


class Player:
    def __init__(self):
        self.season = None
        self.season_wins = self.season_matches = self.season_titles = self.slams_this_year = 0
        self.season_surprise = 0.0
        self.big_title_day = None
        self.event_wins = defaultdict(int)
        self.event_matches = defaultdict(int)
        self.event_titles = defaultdict(int)
        self.last_title_year = {}
        self.last_day = None
        self.since_return = 99
        self.opening_loss_streak = 0
        self.current_event = None

    def start_season(self, year):
        if self.season != year:
            self.season = year
            self.season_wins = self.season_matches = self.season_titles = self.slams_this_year = 0
            self.season_surprise = 0.0


def tournament_history(p, name, year):
    """A player's record at one tournament, read before the match."""
    return {
        "event_win_share": (p.event_wins[name] + 5) / (p.event_matches[name] + 10),
        "event_titles": p.event_titles[name],
        "defending_champion": float(p.last_title_year.get(name) == year - 1),
    }


def build_domain_features(df, ratings, elo_weight, first_year=2000, return_state=False):
    """Return one row per main-tour match played; d_ columns are winner minus loser.

    With return_state, also return each player's history, to score future matches.
    """
    slam_days = sorted(df.loc[df["tourney_level"] == "G", "start_date"].map(
        lambda d: (d - pd.Timestamp("1970-01-01")).days).unique())
    difference = combined_difference(ratings, elo_weight).to_numpy()
    players = defaultdict(Player)
    rows = []

    for i, row in enumerate(df.to_dict("records")):
        winner, loser = players[row["winner_id"]], players[row["loser_id"]]
        day, year, level, tier = row["day"], row["year"], row["tourney_level"], row["tier"]
        name = row["tourney_name"].replace("Us Open", "US Open")
        event = (row["tourney_id"], row["source"])
        winner.start_season(year)
        loser.start_season(year)
        expected = float(win_probability(difference[i], 0))

        if tier == "main" and year >= first_year and not row["walkover"]:
            month = row["start_date"].month
            small_event = level == "A"
            # The Grand Slam calendar is published in advance, so this isn't future information
            position = bisect.bisect_left(slam_days, day + 1)
            next_slam = slam_days[position] if position < len(slam_days) else None
            before_slam = next_slam is not None and next_slam - day <= 14 and level != "G"

            def read(p, rank):
                top10 = rank == rank and rank <= 10
                top20 = rank == rank and rank <= 20
                gap = day - p.last_day if p.last_day is not None else 0
                # Matches played since returning; a match after a long gap restarts the count
                since_return = 0 if gap > 60 else p.since_return
                return {
                    "season_win_share": (p.season_wins + 5) / (p.season_matches + 10),
                    "season_beat_expectation": p.season_surprise / (p.season_matches + 10),
                    "season_titles": p.season_titles,
                    "slams_this_year": p.slams_this_year,
                    "slam_winner_small_event": float(p.slams_this_year > 0 and level != "G"),
                    "top_before_slam": float(top20 and before_slam),
                    "top10_in_small_event": float(top10 and small_event),
                    "big_title_recent": float(p.big_title_day is not None and day - p.big_title_day <= 14),
                    **tournament_history(p, name, year),
                    "late_season_rank": (math.log(rank) if rank == rank and rank > 0 else 7.0) * (month >= 10),
                    "early_in_comeback": float(since_return < 5),
                    "opening_loss_streak": p.opening_loss_streak,
                }

            w, l = read(winner, row["winner_rank"]), read(loser, row["loser_rank"])
            record = {"tourney_id": row["tourney_id"], "match_num": row["match_num"]}
            for key in w:
                record["d_" + key] = round(float(w[key]) - float(l[key]), 8)
            rows.append(record)

        # Everything below uses the result
        if row["walkover"]:
            continue
        for p, won, exp in ((winner, 1, expected), (loser, 0, 1 - expected)):
            if p.last_day is not None and day - p.last_day > 60:
                p.since_return = 0
            p.since_return = min(p.since_return + 1, 99)
            p.last_day = day
            p.season_matches += 1
            p.season_wins += won
            p.season_surprise += won - exp
            if tier == "main":
                p.event_matches[name] += 1
                p.event_wins[name] += won
                # A player's first match of an event: losing it extends the slump
                if p.current_event != event:
                    p.current_event = event
                    p.opening_loss_streak = 0 if won else p.opening_loss_streak + 1

        if tier == "main" and row["round"] == "F":
            winner.season_titles += 1
            winner.event_titles[name] += 1
            winner.last_title_year[name] = year
            if level in BIG_TITLE_LEVELS:
                winner.big_title_day = day
            if level == "G":
                winner.slams_this_year += 1

    frame = pd.DataFrame(rows)
    return (frame, players) if return_state else frame
