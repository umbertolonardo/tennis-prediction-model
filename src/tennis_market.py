"""Compare prices with the market, and apply the seed rule frozen on men's matches from 2010 to 2018."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Frozen on 2010-2018. Never re-estimated: changing these would invalidate every later check.
SEED_RULE = {"a": 1.0995, "beta": -0.0714, "seed_sd": 1.7923, "threshold": 0.02}
UNSEEDED = 64


def fair_probability(odds_a, odds_b):
    """The probability implied for player A once the bookmaker's margin is removed."""
    return (1 / odds_a) / (1 / odds_a + 1 / odds_b)


def margin(odds_a, odds_b):
    return 1 / odds_a + 1 / odds_b - 1


def edge(probability, odds):
    """Expected profit per unit staked, if the probability is right."""
    return probability * odds - 1


def seed_difference(seed_a, seed_b):
    """Same definition as the model's d_seed: positive when player A has the better seed."""
    def value(seed):
        return math.log(seed) if seed and seed > 0 else math.log(UNSEEDED)
    return value(seed_b) - value(seed_a)


def seed_rule_probability(fair, d_seed):
    """The market's fair probability for a player, adjusted by the frozen rule."""
    fair = np.asarray(fair, dtype=float)
    logit = np.log(fair / (1 - fair))
    z = np.asarray(d_seed, dtype=float) / SEED_RULE["seed_sd"]
    return 1 / (1 + np.exp(-(SEED_RULE["a"] * logit + SEED_RULE["beta"] * z)))


def seed_rule_bets(odds_a, odds_b, seed_a=None, seed_b=None, price_a=None, price_b=None):
    """Return the frozen rule's verdict on each player of one match.

    The probability always comes from Pinnacle's odds. price_a and price_b are the prices you
    can actually get, if they differ from Pinnacle's.
    """
    fair_a = fair_probability(odds_a, odds_b)
    d_seed = seed_difference(seed_a, seed_b)
    edge_a = edge(float(seed_rule_probability(fair_a, d_seed)), price_a or odds_a)
    edge_b = edge(float(seed_rule_probability(1 - fair_a, -d_seed)), price_b or odds_b)
    return {"edge_a": edge_a, "edge_b": edge_b,
            "bet_a": edge_a >= SEED_RULE["threshold"], "bet_b": edge_b >= SEED_RULE["threshold"]}


def fit_on_market(table, columns):
    """Logit on both points of view, with errors clustered by match. No intercept: sides are arbitrary."""
    x = table[columns].fillna(0).to_numpy()
    both = np.vstack([x, -x])
    y = np.r_[np.ones(len(table)), np.zeros(len(table))]
    groups = np.r_[np.arange(len(table)), np.arange(len(table))]
    return sm.Logit(y, both).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})


def both_sides(table, winner_odds, loser_odds):
    """One row per possible bet on a match, leaving out matches that weren't completed."""
    fair = fair_probability(table[winner_odds], table[loser_odds])
    common = {"comment": table["Comment"], "year": table["year"], "level": table["tourney_level"]}
    sides = pd.concat([
        pd.DataFrame({"fair": fair, "odds": table[winner_odds], "won": 1.0, "d_seed": table["d_seed"], **common}),
        pd.DataFrame({"fair": 1 - fair, "odds": table[loser_odds], "won": 0.0, "d_seed": -table["d_seed"], **common}),
    ], ignore_index=True)
    sides = sides[sides["comment"] == "Completed"].copy()
    sides["profit"] = np.where(sides["won"] == 1, sides["odds"] - 1, -1.0)
    sides["rule_edge"] = edge(seed_rule_probability(sides["fair"], sides["d_seed"]), sides["odds"])
    return sides


def benjamini_hochberg(p_values, level=0.05):
    """Which tests pass once we allow for having run many of them."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    thresholds = level * (np.arange(len(p_values)) + 1) / len(p_values)
    below = np.flatnonzero(p_values[order] <= thresholds)
    passes = np.zeros(len(p_values), dtype=bool)
    if below.size:
        passes[order[: below.max() + 1]] = True
    return passes


def load_wta_matches(raw_dir):
    """Women's main-tour matches played, with the names and seed difference the odds linking needs."""
    raw_dir = Path(raw_dir)
    matches = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(raw_dir.glob("wta_matches_*.csv"))], ignore_index=True)
    players = pd.read_csv(raw_dir / "wta_players.csv", low_memory=False)[["player_id", "name_first", "name_last"]]
    matches["start_date"] = pd.to_datetime(matches["tourney_date"].astype(str), format="%Y%m%d")
    matches["year"] = matches["start_date"].dt.year
    walkover = matches["score"].astype(str).str.contains(r"W/O|Walkover|^WO$", regex=True) | matches["score"].isna()
    matches = matches[(matches["tourney_level"] != "D") & ~walkover].copy()
    matches["tourney_name"] = matches["tourney_name"].astype(str).str.strip()
    matches["round"] = matches["round"].astype(str)
    for side in ["winner", "loser"]:
        names = players.rename(columns={"player_id": f"{side}_id", "name_first": f"{side}_first", "name_last": f"{side}_surname"})
        matches = matches.merge(names, on=f"{side}_id", how="left", validate="many_to_one")

    def value(column):
        seed = pd.to_numeric(matches[column], errors="coerce")
        return np.log(seed.where(seed > 0).fillna(UNSEEDED))

    matches["d_seed"] = value("loser_seed") - value("winner_seed")
    return matches.reset_index(drop=True)


def roi_summary(profit, rounds=10000, seed=0):
    """Flat-stake return, with a 95% bootstrap interval."""
    profit = np.asarray(profit, dtype=float)
    rng = np.random.default_rng(seed)
    means = [profit[rng.integers(0, len(profit), len(profit))].mean() for _ in range(rounds)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {"bets": len(profit), "profit_units": round(profit.sum(), 1), "roi_pct": round(100 * profit.mean(), 2),
            "low_95": round(100 * low, 2), "high_95": round(100 * high, 2)}
