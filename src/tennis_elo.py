"""Elo ratings across every tour level, with pre-match values saved for each match."""
import math

import numpy as np
import pandas as pd

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
INACTIVE_AFTER_DAYS = 60


def win_probability(rating_a, rating_b):
    """Probability that the player rated rating_a beats the player rated rating_b."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def run_elo(
    df,
    k=32,
    tier_weight=None,
    start_rating=None,
    inactivity=0.0,
    new_player_boost=0.0,
    update_on_early_end=True,
    return_state=False,
):
    """Return pre-match overall and surface Elo for the winner and loser of each match.

    tier_weight:      how much a match at each level moves ratings (main tour = 1)
    start_rating:     a new player's rating, based on the level where they first appear
    inactivity:       share of a rating's distance from 1500 lost after a year away
    new_player_boost: extra K for players with few matches, fading as they play more
    return_state:     also return the final ratings, to score future matches
    """
    tier_weight = tier_weight or {"main": 1.0, "challenger": 1.0, "qualifying": 1.0, "futures": 1.0}
    start_rating = start_rating or {"main": 1500, "challenger": 1500, "qualifying": 1500, "futures": 1500}

    overall, surface, played, last_day = {}, {}, {}, {}
    rows = []

    def prepare(player, tier, day):
        """Create a new player, or apply the inactivity penalty to a returning one."""
        if player not in overall:
            overall[player] = float(start_rating[tier])
            surface[player] = {}
            played[player] = 0
            last_day[player] = day
        elif inactivity > 0:
            gap = day - last_day[player]
            if gap > INACTIVE_AFTER_DAYS:
                keep = 1 - inactivity * min(1.0, (gap - INACTIVE_AFTER_DAYS) / 365)
                overall[player] = 1500 + (overall[player] - 1500) * keep
                for court in surface[player]:
                    surface[player][court] = 1500 + (surface[player][court] - 1500) * keep

    columns = zip(df["winner_id"], df["loser_id"], df["surface"], df["tier"], df["day"],
                  df["walkover"], df["ended_early"])
    for winner, loser, court, tier, day, walkover, ended_early in columns:
        prepare(winner, tier, day)
        prepare(loser, tier, day)
        has_surface = court in SURFACES
        if has_surface:
            # A player's first match on a surface starts from their overall rating
            surface[winner].setdefault(court, overall[winner])
            surface[loser].setdefault(court, overall[loser])

        # Save the ratings BEFORE this match's result is used
        rows.append((
            overall[winner], overall[loser],
            surface[winner][court] if has_surface else math.nan,
            surface[loser][court] if has_surface else math.nan,
        ))

        if walkover or (ended_early and not update_on_early_end):
            continue

        weight = tier_weight[tier]
        k_winner = k * weight * (1 + new_player_boost * math.exp(-played[winner] / 30))
        k_loser = k * weight * (1 + new_player_boost * math.exp(-played[loser] / 30))

        surprise = 1 - win_probability(overall[winner], overall[loser])
        overall[winner] += k_winner * surprise
        overall[loser] -= k_loser * surprise

        if has_surface:
            surprise = 1 - win_probability(surface[winner][court], surface[loser][court])
            surface[winner][court] += k_winner * surprise
            surface[loser][court] -= k_loser * surprise

        played[winner] += 1
        played[loser] += 1
        last_day[winner] = day
        last_day[loser] = day

    frame = pd.DataFrame(
        rows,
        columns=["winner_elo", "loser_elo", "winner_surface_elo", "loser_surface_elo"],
        index=df.index,
    )
    if return_state:
        return frame, {"overall": overall, "surface": surface, "last_day": last_day, "inactivity": inactivity}
    return frame


def ratings_on_day(state, player, court, day):
    """A player's overall and surface Elo on a future day, after any inactivity penalty."""
    gap = day - state["last_day"][player]
    keep = 1.0
    if state["inactivity"] > 0 and gap > INACTIVE_AFTER_DAYS:
        keep = 1 - state["inactivity"] * min(1.0, (gap - INACTIVE_AFTER_DAYS) / 365)
    overall = state["overall"][player]
    # A player's first match on a surface starts from their overall rating
    surface = state["surface"][player].get(court, overall)
    return 1500 + (overall - 1500) * keep, 1500 + (surface - 1500) * keep


def combined_difference(ratings, weight):
    """Winner's rating advantage, from a weighted mix of overall and surface Elo."""
    winner_surface = ratings["winner_surface_elo"].fillna(ratings["winner_elo"])
    loser_surface = ratings["loser_surface_elo"].fillna(ratings["loser_elo"])
    winner_mix = weight * ratings["winner_elo"] + (1 - weight) * winner_surface
    loser_mix = weight * ratings["loser_elo"] + (1 - weight) * loser_surface
    return winner_mix - loser_mix


def log_loss_of_difference(difference):
    """Log loss when the winner's probability comes directly from the rating difference."""
    p = np.clip(1 / (1 + 10 ** (-np.asarray(difference) / 400)), 1e-6, 1 - 1e-6)
    return float(-np.log(p).mean())
