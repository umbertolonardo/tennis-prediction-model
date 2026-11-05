"""Event study of retirements, compared with the same player's ordinary losses.

Outcome: won - P(win), with the player's Elo frozen before the window and the opponent's Elo current.
Match k = 0 is a loss in both groups, so it is left out. Baseline: matches -10 to -6.
"""
import numpy as np
import pandas as pd

from tennis_elo import win_probability

WINDOW = 10
BASELINE = (-10, -6)
N_CONTROLS = 5
K = [k for k in range(-WINDOW, WINDOW + 1) if k != 0]
FIRST_YEAR = 2005


def player_match_table(matches, ratings):
    """Return one row per player per played match, in playing order."""
    data = matches.copy()
    data[["winner_elo", "loser_elo"]] = ratings[["winner_elo", "loser_elo"]].to_numpy()
    data["retired"] = data["score"].astype(str).str.contains("RET", regex=False)
    played = data[~data["walkover"]]
    common = ["day", "year", "tier", "ended_early", "tourney_id"]
    long = pd.concat([
        played[common].assign(player=played["winner_id"], won=1.0, own_elo=played["winner_elo"],
                              opp_elo=played["loser_elo"], retired=False, order=played.index,
                              rank=played["winner_rank"]),
        played[common].assign(player=played["loser_id"], won=0.0, own_elo=played["loser_elo"],
                              opp_elo=played["winner_elo"], retired=played["retired"], order=played.index,
                              rank=played["loser_rank"]),
    ])
    return long.sort_values(["player", "order"], kind="stable").reset_index(drop=True)


def walkover_losses_by_player(matches):
    """Days on which each player lost by walkover."""
    losses = matches[matches["walkover"] & matches["score"].notna()]
    return losses.groupby("loser_id")["day"].apply(lambda days: np.sort(days.to_numpy())).to_dict()


def build_windows(long, walkover_days, match_position=True):
    """Return every match around each event, and one row per event.

    Each retirement gets up to N_CONTROLS ordinary losses from the same player and level, nearest in time.
    With match_position, they must also have the same place in a tournament (first match or not).
    Windows containing another retirement or walkover loss are left out, in both groups.
    """
    windows, events = [], []
    for player, table in long.groupby("player", sort=False):
        n = len(table)
        if n < 2 * WINDOW + 1:
            continue
        won = table["won"].to_numpy()
        own = table["own_elo"].to_numpy()
        opp = table["opp_elo"].to_numpy()
        day = table["day"].to_numpy()
        year = table["year"].to_numpy()
        tier = table["tier"].to_numpy()
        rank = table["rank"].to_numpy()
        retired = table["retired"].to_numpy()
        ended_early = table["ended_early"].to_numpy()
        tourney = table["tourney_id"].to_numpy()
        first_match = np.r_[True, tourney[1:] != tourney[:-1]]

        problem = retired.copy()
        if player in walkover_days:
            # A walkover isn't a played match, so it marks the player's next match
            positions = np.searchsorted(day, walkover_days[player], side="left")
            problem[positions[positions < n]] = True
        problem_positions = np.flatnonzero(problem)

        def is_clean(position):
            near = problem_positions[np.abs(problem_positions - position) <= WINDOW]
            return np.all(near == position) if retired[position] else near.size == 0

        in_period = [i for i in range(WINDOW, n) if year[i] >= FIRST_YEAR]
        treated = [i for i in in_period if retired[i] and is_clean(i)]
        if not treated:
            continue
        pool = np.array([i for i in in_period if won[i] == 0 and not ended_early[i] and is_clean(i)], dtype=int)

        def add(event, position, group, weight, event_rank):
            frozen_elo = own[position - WINDOW]
            positions = np.arange(position - WINDOW, min(n - 1, position + WINDOW) + 1)
            outcome = won[positions] - win_probability(frozen_elo, opp[positions])
            live = won[positions] - win_probability(own[positions], opp[positions])
            k = positions - position
            baseline = outcome[(k >= BASELINE[0]) & (k <= BASELINE[1])].mean()
            keep = k != 0
            windows.append(pd.DataFrame({
                "event": event, "player": player, "group": group, "weight": weight, "tier": tier[position],
                "event_rank": event_rank, "k": k[keep], "y": outcome[keep] - baseline, "live": live[keep],
            }))
            next_gap = day[position + 1] - day[position] if position + 1 < n else np.nan
            events.append({"event": event, "group": group, "weight": weight, "tier": tier[position],
                           "year": year[position], "next_gap": next_gap})

        for position in treated:
            same = tier[pool] == tier[position]
            if match_position:
                same = same & (first_match[pool] == first_match[position])
            options = pool[same]
            if options.size == 0:
                continue
            controls = options[np.argsort(np.abs(options - position), kind="stable")[:N_CONTROLS]]
            event = f"{player}_{position}"
            add(event, position, 1, 1.0, rank[position])
            for control in controls:
                add(event, control, 0, 1.0 / len(controls), rank[position])
    return pd.concat(windows, ignore_index=True), pd.DataFrame(events)


def estimate_path(windows, rounds=1000, seed=0):
    """Retirement path minus ordinary-loss path, by match. The bootstrap resamples whole players."""
    players, player_index = np.unique(windows["player"], return_inverse=True)
    k_index = windows["k"].map({k: i for i, k in enumerate(K)}).to_numpy()
    group = windows["group"].to_numpy()
    weight = windows["weight"].to_numpy()
    sums = np.zeros((2, len(players), len(K)))
    weights = np.zeros((2, len(players), len(K)))
    np.add.at(sums, (group, player_index, k_index), weight * windows["y"].to_numpy())
    np.add.at(weights, (group, player_index, k_index), weight)

    def paths(draw):
        draw = draw[:, None]
        retired = (draw * sums[1]).sum(0) / (draw * weights[1]).sum(0)
        ordinary = (draw * sums[0]).sum(0) / (draw * weights[0]).sum(0)
        return retired, ordinary

    retired, ordinary = paths(np.ones(len(players)))
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(players), 1 / len(players))
    draws = np.empty((rounds, len(K)))
    for i in range(rounds):
        retired_draw, ordinary_draw = paths(rng.multinomial(len(players), probabilities))
        draws[i] = retired_draw - ordinary_draw
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    table = pd.DataFrame({
        "k": K, "retired": 100 * retired, "ordinary_loss": 100 * ordinary, "effect": 100 * (retired - ordinary),
        "low_95": 100 * low, "high_95": 100 * high,
    })
    return table, draws


def pooled_effect(table, draws, matches):
    """Average effect across a set of matches, with its 95% interval, in percentage points."""
    positions = [K.index(k) for k in matches]
    low, high = np.percentile(draws[:, positions].mean(1), [2.5, 97.5])
    return {"effect": table["effect"].to_numpy()[positions].mean(), "low_95": 100 * low, "high_95": 100 * high}


def time_away(events, last_year=2020):
    """Time until the next match. Events end in last_year, so everyone has a year in which to return."""
    rows = []
    for group, name in [(1, "Retirement"), (0, "Ordinary loss")]:
        subset = events[(events["group"] == group) & (events["year"] <= last_year)]
        gap, weight = subset["next_gap"], subset["weight"]
        known = subset.dropna(subset=["next_gap"]).sort_values("next_gap")
        middle = np.searchsorted(known["weight"].cumsum(), known["weight"].sum() / 2)
        row = {"event": name, "median_days": known["next_gap"].iloc[middle]}
        for days in [14, 30, 90, 365]:
            row[f"back_in_{days}d_pct"] = 100 * (weight * (gap <= days)).sum() / weight.sum()
        rows.append(row)
    return pd.DataFrame(rows)


def reach_share(windows, events, matches=(1, 5, 10)):
    """Share of events in which the player reached each later match, as a check on attrition."""
    reached = windows[windows["k"].isin(matches)].groupby(["group", "k"])["weight"].sum().unstack()
    share = 100 * reached.div(events.groupby("group")["weight"].sum(), axis=0)
    return share.rename(index={0: "Ordinary loss", 1: "Retirement"})
