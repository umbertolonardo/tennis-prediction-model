"""Build pre-match features by walking through every match in chronological order.

The rule that prevents data leakage: for each match, features are read from each
player's history FIRST, and that history is updated with the match's result AFTERWARDS.
"""
import math
from collections import defaultdict, deque
from functools import partial

import numpy as np
import pandas as pd

from tennis_elo import combined_difference, win_probability

SERVE_PRIOR = 0.63  # A typical share of service points won
PRIOR_POINTS = 100  # How strongly a player's stats are pulled to average when data is thin
DEFAULT_MINUTES = 100

# Half-lives in days: recent matches count more than old ones
FAST, SLOW = 90, 365


class DecayingAverage:
    """A running ratio in which older observations count less and less."""

    __slots__ = ("half_life", "day", "top", "bottom")

    def __init__(self, half_life):
        self.half_life, self.day, self.top, self.bottom = half_life, None, 0.0, 0.0

    def add(self, day, top, bottom):
        if self.day is not None:
            fade = 0.5 ** ((day - self.day) / self.half_life)
            self.top *= fade
            self.bottom *= fade
        self.day = day
        self.top += top
        self.bottom += bottom

    def value(self, prior, prior_weight=PRIOR_POINTS):
        """The ratio, pulled towards the prior when there is little data."""
        return (self.top + prior * prior_weight) / (self.bottom + prior_weight)


class PlayerHistory:
    """Everything remembered about one player, updated after each of their matches."""

    def __init__(self):
        self.serve_fast, self.serve_slow = DecayingAverage(FAST), DecayingAverage(SLOW)
        self.ret_fast, self.ret_slow = DecayingAverage(FAST), DecayingAverage(SLOW)
        self.serve_adj, self.ret_adj = DecayingAverage(SLOW), DecayingAverage(SLOW)
        self.ace, self.double_fault = DecayingAverage(SLOW), DecayingAverage(SLOW)
        self.first_in, self.bp_saved, self.bp_won = (
            DecayingAverage(SLOW), DecayingAverage(SLOW), DecayingAverage(SLOW))
        self.surface_serve = defaultdict(partial(DecayingAverage, SLOW))
        self.surface_ret = defaultdict(partial(DecayingAverage, SLOW))

        self.recent = deque(maxlen=20)        # (won, expected) at any level
        self.recent_tour = deque(maxlen=20)   # the same, main tour only
        self.schedule = deque(maxlen=40)      # (day, minutes, event)
        self.streak = 0
        self.career = 0
        self.career_tour = 0
        self.surface_career = defaultdict(int)
        self.last_day = None
        self.last_problem_day = None          # last retirement or walkover loss
        self.vs_hand = {"L": [0, 0], "R": [0, 0]}


def _share(values, last_n):
    selected = list(values)[-last_n:]
    if not selected:
        return math.nan
    return sum(item[0] for item in selected) / len(selected)


def read_player(h, court, day, event):
    """Read a player's pre-match features."""
    f = {
        "serve_fast": h.serve_fast.value(SERVE_PRIOR),
        "serve_slow": h.serve_slow.value(SERVE_PRIOR),
        "return_fast": h.ret_fast.value(1 - SERVE_PRIOR),
        "return_slow": h.ret_slow.value(1 - SERVE_PRIOR),
        "serve_adj": h.serve_adj.value(0.0),
        "return_adj": h.ret_adj.value(0.0),
        "ace_rate": h.ace.value(0.07),
        "df_rate": h.double_fault.value(0.04),
        "first_in": h.first_in.value(0.6),
        "bp_saved": h.bp_saved.value(0.6, 20),
        "bp_won": h.bp_won.value(0.4, 20),
        "serve_surface": h.surface_serve[court].value(SERVE_PRIOR),
        "return_surface": h.surface_ret[court].value(1 - SERVE_PRIOR),
        "win_last_5": _share(h.recent, 5),
        "win_last_10": _share(h.recent, 10),
        "win_last_20": _share(h.recent, 20),
        "tour_win_last_20": _share(h.recent_tour, 20),
        "beat_expectation": (
            sum(won - exp for won, exp in h.recent) / len(h.recent) if h.recent else math.nan),
        "streak": h.streak,
        "log_career": math.log1p(h.career),
        "log_tour_career": math.log1p(h.career_tour),
        "log_surface_career": math.log1p(h.surface_career[court]),
    }

    matches_7 = matches_14 = matches_30 = minutes_7 = minutes_14 = minutes_event = 0
    for past_day, past_minutes, past_event in h.schedule:
        gap = day - past_day
        if gap <= 30:
            matches_30 += 1
        if gap <= 14:
            matches_14 += 1
            minutes_14 += past_minutes
        if gap <= 7:
            matches_7 += 1
            minutes_7 += past_minutes
        if past_event == event:
            minutes_event += past_minutes
    rest = day - h.last_day if h.last_day is not None else math.nan
    f.update(
        matches_7=matches_7, matches_14=matches_14, matches_30=matches_30,
        minutes_7=minutes_7, minutes_14=minutes_14, minutes_event=minutes_event,
        log_rest=math.log1p(max(rest, 0)) if rest == rest else math.nan,
        long_layoff=float(rest > 60) if rest == rest else math.nan,
        recent_problem=float(
            h.last_problem_day is not None and day - h.last_problem_day <= 30),
    )
    return f


def update_player(h, opponent_hand, court, day, event, tier, won, expected, minutes):
    h.recent.append((float(won), expected))
    if tier == "main":
        h.recent_tour.append((float(won), expected))
        h.career_tour += 1
    h.schedule.append((day, minutes, event))
    h.streak = h.streak + 1 if won else 0
    h.career += 1
    h.surface_career[court] += 1
    h.last_day = day
    if opponent_hand in h.vs_hand:
        h.vs_hand[opponent_hand][0] += int(won)
        h.vs_hand[opponent_hand][1] += 1


def update_serve_stats(me, opponent, day, court, st, opp_return_before, opp_serve_before):
    """Update a player's serve and return history from one match's statistics."""
    points, won = st["svpt"], st["first_won"] + st["second_won"]
    opp_points, opp_won = st["opp_svpt"], st["opp_first_won"] + st["opp_second_won"]
    ret_won = opp_points - opp_won

    me.serve_fast.add(day, won, points)
    me.serve_slow.add(day, won, points)
    me.ret_fast.add(day, ret_won, opp_points)
    me.ret_slow.add(day, ret_won, opp_points)
    me.surface_serve[court].add(day, won, points)
    me.surface_ret[court].add(day, ret_won, opp_points)

    # Points won above what an average player would win against THIS opponent
    me.serve_adj.add(day, won - points * (1 - opp_return_before), points)
    me.ret_adj.add(day, ret_won - opp_points * (1 - opp_serve_before), opp_points)

    me.ace.add(day, st["ace"], points)
    me.double_fault.add(day, st["df"], points)
    me.first_in.add(day, st["first_in"], points)
    me.bp_saved.add(day, st["bp_saved"], st["bp_faced"])
    me.bp_won.add(day, st["opp_bp_faced"] - st["opp_bp_saved"], st["opp_bp_faced"])


def _stats(row, mine, theirs):
    return {
        "svpt": row[mine + "svpt"], "first_won": row[mine + "1stWon"],
        "second_won": row[mine + "2ndWon"], "ace": row[mine + "ace"], "df": row[mine + "df"],
        "first_in": row[mine + "1stIn"], "bp_saved": row[mine + "bpSaved"],
        "bp_faced": row[mine + "bpFaced"], "opp_svpt": row[theirs + "svpt"],
        "opp_first_won": row[theirs + "1stWon"], "opp_second_won": row[theirs + "2ndWon"],
        "opp_bp_faced": row[theirs + "bpFaced"], "opp_bp_saved": row[theirs + "bpSaved"],
    }


def _log_rank(rank):
    return math.log(rank) if rank == rank and rank > 0 else math.nan


def _log_points(points):
    return math.log1p(points) if points == points and points >= 0 else math.nan


class FeatureState:
    """Everything remembered while walking through the matches, so it can also score a future match."""

    def __init__(self, players, tournament_info):
        self.hand = dict(zip(players["player_id"], players["hand"]))
        self.height = dict(zip(players["player_id"], players["height"]))
        self.ioc = dict(zip(players["player_id"], players["ioc"]))
        self.places = defaultdict(list)
        for row in tournament_info.itertuples(index=False):
            self.places[row.tourney_name].append((row.start_year, row.end_year, row.country, row.indoor))
        self.history = defaultdict(PlayerHistory)
        self.head_to_head = defaultdict(int)
        self.event_aces = defaultdict(partial(DecayingAverage, SLOW))
        self.event_breaks = defaultdict(partial(DecayingAverage, SLOW))

    def place_of(self, name, year):
        for start, end, country, indoor in self.places.get(name, []):
            if start <= year <= end:
                return country, indoor
        return None, math.nan


def _versus_style(h, opponent_hand):
    wins, total = h.vs_hand.get(opponent_hand, (0, 0))
    all_wins = h.vs_hand["L"][0] + h.vs_hand["R"][0]
    all_total = h.vs_hand["L"][1] + h.vs_hand["R"][1]
    overall = (all_wins + 5) / (all_total + 10)
    return (wins + overall * 20) / (total + 20) - overall


def _seed_value(seed):
    return math.log(seed) if seed == seed and seed > 0 else math.log(64)


def match_record(state, row, difference, overall_diff, surface_diff):
    """Read one match's features from the players' histories. Winner here means the first player."""
    winner, loser = row["winner_id"], row["loser_id"]
    w, l = state.history[winner], state.history[loser]
    court, day = row["surface"], row["day"]
    event = (row["tourney_id"], row["source"])

    wf = read_player(w, court, day, event)
    lf = read_player(l, court, day, event)
    record = {
        "tourney_id": row["tourney_id"], "match_num": row["match_num"],
        "year": row["year"], "tourney_level": row["tourney_level"],
        "ended_early": row["ended_early"],
        "d_elo_combined": difference,
        "d_elo_overall": overall_diff,
        "d_elo_surface": surface_diff,
        "d_log_rank": _log_rank(row["loser_rank"]) - _log_rank(row["winner_rank"]),
        "d_log_points": (
            _log_points(row["winner_rank_points"]) - _log_points(row["loser_rank_points"])),
        "d_unranked": float(row["winner_rank"] != row["winner_rank"])
        - float(row["loser_rank"] != row["loser_rank"]),
    }
    for name in wf:
        record["d_" + name] = wf[name] - lf[name]

    # Physical profile and style
    w_hand, l_hand = state.hand.get(winner), state.hand.get(loser)
    record["d_height"] = state.height.get(winner, math.nan) - state.height.get(loser, math.nan)
    record["d_age"] = row["winner_age"] - row["loser_age"]
    record["d_lefty"] = float(w_hand == "L") - float(l_hand == "L")
    record["d_vs_hand"] = _versus_style(w, l_hand) - _versus_style(l, w_hand)

    # Head to head, shrunk towards even when they have rarely met
    wins_w = state.head_to_head[(winner, loser)]
    wins_l = state.head_to_head[(loser, winner)]
    surf_w = state.head_to_head[(winner, loser, court)]
    surf_l = state.head_to_head[(loser, winner, court)]
    record["d_h2h_wins"] = wins_w - wins_l
    # Written as one difference so that swapping the players gives exactly the opposite
    # number. Rounding differences here would reveal which player won (see notebook 05).
    record["d_h2h_share"] = (wins_w - wins_l) / (wins_w + wins_l + 4)
    record["d_h2h_surface"] = surf_w - surf_l

    # Draw information, known before the match
    record["d_seed"] = _seed_value(row["loser_seed"]) - _seed_value(row["winner_seed"])
    for label, code in (("qualifier", "Q"), ("wildcard", "WC"),
                        ("lucky_loser", "LL"), ("protected", "PR")):
        record["d_" + label] = (
            float(row["winner_entry"] == code) - float(row["loser_entry"] == code))

    # Home advantage
    country, indoor = state.place_of(row["tourney_name"], row["year"])
    w_home = float(country is not None and state.ioc.get(winner) == country)
    l_home = float(country is not None and state.ioc.get(loser) == country)
    record["d_home"] = w_home - l_home
    record["one_home_player"] = float(w_home != l_home)

    # Match context: the same from both players' points of view
    record.update(
        best_of_5=float(row["best_of"] == 5),
        draw_size=row["draw_size"],
        round_order=row["round_order"],
        level_slam=float(row["tourney_level"] == "G"),
        level_masters=float(row["tourney_level"] == "M"),
        is_hard=float(court == "Hard"),
        is_clay=float(court == "Clay"),
        is_grass=float(court == "Grass"),
        indoor=indoor,
        court_ace_rate=state.event_aces[row["tourney_name"]].value(0.08, 500),
        court_break_rate=state.event_breaks[row["tourney_name"]].value(0.22, 100),
    )
    return record


def build_features(df, ratings, players, tournament_info, elo_weight, first_year=2000, return_state=False):
    """Return one row per main-tour match played, from the winner's point of view.

    Columns starting with d_ are "winner minus loser" and flip sign for the opposite
    point of view. The other columns describe the match and stay the same.
    With return_state, also return what was remembered, to score future matches.
    """
    state = FeatureState(players, tournament_info)
    hand = state.hand
    history = state.history
    head_to_head = state.head_to_head
    event_aces = state.event_aces
    event_breaks = state.event_breaks

    difference = combined_difference(ratings, elo_weight).to_numpy()
    overall_diff = (ratings["winner_elo"] - ratings["loser_elo"]).to_numpy()
    surface_diff = (ratings["winner_surface_elo"] - ratings["loser_surface_elo"]).to_numpy()

    rows = []
    for i, row in enumerate(df.to_dict("records")):
        winner, loser = row["winner_id"], row["loser_id"]
        w, l = history[winner], history[loser]
        court, day, tier = row["surface"], row["day"], row["tier"]
        event = (row["tourney_id"], row["source"])
        expected = float(win_probability(difference[i], 0))

        if tier == "main" and row["year"] >= first_year and not row["walkover"]:
            rows.append(match_record(state, row, difference[i], overall_diff[i], surface_diff[i]))

        # Everything below uses the result, so it comes after the features are saved
        if row["walkover"]:
            l.last_problem_day = day
            continue

        minutes = row["minutes"] if row["minutes"] == row["minutes"] else DEFAULT_MINUTES
        if row["ended_early"]:
            l.last_problem_day = day

        w_points, l_points = row["w_svpt"], row["l_svpt"]
        has_stats = (w_points == w_points and l_points == l_points
                     and w_points > 0 and l_points > 0)
        if has_stats and all(row[c] == row[c] for c in (
                "w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon", "w_ace", "l_ace", "w_df", "l_df",
                "w_1stIn", "l_1stIn", "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced")):
            # Each player's adjustment uses the opponent's level from before this match
            w_serve_before, w_return_before = w.serve_slow.value(SERVE_PRIOR), w.ret_slow.value(1 - SERVE_PRIOR)
            l_serve_before, l_return_before = l.serve_slow.value(SERVE_PRIOR), l.ret_slow.value(1 - SERVE_PRIOR)
            update_serve_stats(w, l, day, court, _stats(row, "w_", "l_"), l_return_before, l_serve_before)
            update_serve_stats(l, w, day, court, _stats(row, "l_", "w_"), w_return_before, w_serve_before)
            if tier == "main":
                event_aces[row["tourney_name"]].add(day, row["w_ace"] + row["l_ace"], w_points + l_points)
                breaks = (row["w_bpFaced"] - row["w_bpSaved"]) + (row["l_bpFaced"] - row["l_bpSaved"])
                event_breaks[row["tourney_name"]].add(day, breaks, (w_points + l_points) / 6.5)

        update_player(w, hand.get(loser), court, day, event, tier, True, expected, minutes)
        update_player(l, hand.get(winner), court, day, event, tier, False, 1 - expected, minutes)
        head_to_head[(winner, loser)] += 1
        head_to_head[(winner, loser, court)] += 1

    features = pd.DataFrame(rows)
    direction_columns = [c for c in features.columns if c.startswith("d_")]
    features[direction_columns] = features[direction_columns].astype(float).round(8)
    return (features, state) if return_state else features


FEATURE_GROUPS = {
    "Elo": ["d_elo_combined", "d_elo_overall", "d_elo_surface"],
    "Ranking": ["d_log_rank", "d_log_points", "d_unranked"],
    "Serve and return": [
        "d_serve_fast", "d_serve_slow", "d_return_fast", "d_return_slow", "d_serve_adj", "d_return_adj",
        "d_ace_rate", "d_df_rate", "d_first_in", "d_bp_saved", "d_bp_won", "d_serve_surface", "d_return_surface",
    ],
    "Form": ["d_win_last_5", "d_win_last_10", "d_win_last_20", "d_tour_win_last_20", "d_beat_expectation", "d_streak"],
    "Fatigue and fitness": [
        "d_matches_7", "d_matches_14", "d_matches_30", "d_minutes_7", "d_minutes_14", "d_minutes_event",
        "d_log_rest", "d_long_layoff", "d_recent_problem",
    ],
    "Experience": ["d_log_career", "d_log_tour_career", "d_log_surface_career"],
    "Style and age": ["d_height", "d_age", "d_lefty", "d_vs_hand"],
    "Head to head": ["d_h2h_wins", "d_h2h_share", "d_h2h_surface"],
    "Draw": ["d_seed", "d_qualifier", "d_wildcard", "d_lucky_loser", "d_protected"],
    "Home": ["d_home"],
    "Match context": [
        "best_of_5", "draw_size", "round_order", "level_slam", "level_masters", "is_hard", "is_clay",
        "is_grass", "indoor", "court_ace_rate", "court_break_rate",
    ],
}


def both_points_of_view(X):
    """Return the opposite point of view of X: every d_ column changes sign."""
    flipped = X.copy()
    direction_columns = [c for c in X.columns if c.startswith("d_")]
    flipped[direction_columns] = -flipped[direction_columns]
    return flipped
