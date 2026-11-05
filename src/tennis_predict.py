"""Score a future match with the final model, using a saved snapshot of everything known so far."""
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tennis_data import ROUND_ORDER, load_all_matches
from tennis_domain_features import DOMAIN_GROUPS, build_domain_features, tournament_history
from tennis_elo import combined_difference, ratings_on_day, run_elo, win_probability
from tennis_features import FEATURE_GROUPS, build_features, match_record
from tennis_models import INTERACTIONS, add_interactions, fit_both_views, logistic_model, predict_winner_probability

ELO_SETTINGS = dict(k=28, inactivity=0.3, new_player_boost=1.0)
ELO_WEIGHT = 0.6
FIRST_TRAIN_YEAR = 2005
ACTIVE_DAYS = 3 * 365
# After this many days without new data, recent schedule says nothing about a future match
FRESH_DAYS = 14

ORDER = ["Elo", "Ranking", "Serve and return", "Form", "Fatigue and fitness", "Experience",
         "Style and age", "Head to head", "Draw", "Home"]
FINAL_COLUMNS = (
    [c for group in ORDER for c in FEATURE_GROUPS[group]]
    + list(INTERACTIONS) + DOMAIN_GROUPS["Tournament history"]
)
SCHEDULE_COLUMNS = ["d_matches_7", "d_matches_14", "d_matches_30", "d_minutes_7", "d_minutes_14",
                    "d_minutes_event", "d_log_rest", "d_long_layoff", "d_recent_problem"]
LEVELS = {"Grand Slam": "G", "Masters 1000": "M", "ATP 250 / 500": "A", "Tour Finals": "F"}
ROUNDS = {"First round": "R64", "Round of 32": "R32", "Round of 16": "R16", "Quarterfinal": "QF",
          "Semifinal": "SF", "Final": "F", "Round robin": "RR"}


def build_snapshot(raw_dir, tournament_info_path, out_path):
    """Run the whole pipeline on all data, train the final model, and save the result."""
    matches = load_all_matches(raw_dir)
    players = pd.read_csv(Path(raw_dir) / "atp_players.csv", low_memory=False)
    tournament_info = pd.read_csv(tournament_info_path)

    ratings, elo_state = run_elo(matches, return_state=True, **ELO_SETTINGS)
    features, feature_state = build_features(matches, ratings, players, tournament_info, ELO_WEIGHT, return_state=True)
    domain, domain_players = build_domain_features(matches, ratings, ELO_WEIGHT, return_state=True)

    history = DOMAIN_GROUPS["Tournament history"]
    data = features.merge(domain[["tourney_id", "match_num"] + history], on=["tourney_id", "match_num"], validate="one_to_one")
    data = add_interactions(data)
    train = data[(data["tourney_level"] != "D") & (data["year"] >= FIRST_TRAIN_YEAR)]
    model = fit_both_views(logistic_model(), train[FINAL_COLUMNS])

    last_day = int(matches["day"].max())
    active = {p for p, day in elo_state["last_day"].items() if last_day - day <= ACTIVE_DAYS}
    played = matches[~matches["walkover"]]
    long = pd.concat([
        played[["winner_id", "winner_rank", "winner_rank_points", "day"]].set_axis(["player", "rank", "points", "day"], axis=1),
        played[["loser_id", "loser_rank", "loser_rank_points", "day"]].set_axis(["player", "rank", "points", "day"], axis=1),
    ]).dropna(subset=["rank"]).sort_values("day", kind="stable")
    latest = long.groupby("player").last()

    table = players[players["player_id"].isin(active)].copy()
    table["name"] = (table["name_first"].fillna("") + " " + table["name_last"].fillna("")).str.strip()
    table["rank"] = table["player_id"].map(latest["rank"])
    table["points"] = table["player_id"].map(latest["points"])
    table["dob"] = pd.to_datetime(table["dob"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    table = table.sort_values(["rank", "name"], na_position="last").reset_index(drop=True)
    active = set(table["player_id"])

    # Keep only what future matches can need, so the file stays small
    feature_state.history = _keep(feature_state.history, active)
    feature_state.head_to_head = _keep_pairs(feature_state.head_to_head, active)
    feature_state.hand = {p: feature_state.hand.get(p) for p in active}
    feature_state.height = {p: feature_state.height.get(p, math.nan) for p in active}
    feature_state.ioc = {p: feature_state.ioc.get(p) for p in active}
    elo_state = {
        "overall": {p: v for p, v in elo_state["overall"].items() if p in active},
        "surface": {p: v for p, v in elo_state["surface"].items() if p in active},
        "last_day": {p: v for p, v in elo_state["last_day"].items() if p in active},
        "inactivity": elo_state["inactivity"],
    }
    domain_players = _keep(domain_players, active)

    main = matches[(matches["source"] == "main") & (matches["year"] >= 2015) & (matches["tourney_level"].isin(LEVELS.values()))]
    tournaments = (main.sort_values("day").groupby("tourney_name").agg(
        level=("tourney_level", "last"), surface=("surface", "last"), best_of=("best_of", "max")).reset_index())

    snapshot = {
        "model": model, "columns": FINAL_COLUMNS, "players": table, "tournaments": tournaments,
        "elo_state": elo_state, "feature_state": feature_state, "domain_players": domain_players,
        "last_day": last_day, "last_date": pd.Timestamp("1970-01-01") + pd.Timedelta(days=last_day),
        "train_rows": len(train),
    }
    joblib.dump(snapshot, out_path, compress=3)
    return snapshot, data


def _keep(mapping, active):
    mapping_type = type(mapping)
    result = mapping_type(mapping.default_factory)
    for key, value in mapping.items():
        if key in active:
            result[key] = value
    return result


def _keep_pairs(mapping, active):
    result = type(mapping)(mapping.default_factory)
    for key, value in mapping.items():
        if key[0] in active and key[1] in active:
            result[key] = value
    return result


def predict_match(snapshot, player_a, player_b, surface, tourney_name, level, round_code, best_of, match_date,
                  seed_a=None, seed_b=None, entry_a=None, entry_b=None):
    """Return the win probability of player_a. Player ids must be in the snapshot's player table."""
    if player_a == player_b:
        raise ValueError("Choose two different players.")
    table = snapshot["players"].set_index("player_id")
    match_date = pd.Timestamp(match_date)
    stale = (match_date - snapshot["last_date"]).days > FRESH_DAYS
    # With stale data, measure everything at the end of the data rather than adding a false layoff
    day = snapshot["last_day"] + 1 if stale else (match_date - pd.Timestamp("1970-01-01")).days

    elo = snapshot["elo_state"]
    a_overall, a_surface = ratings_on_day(elo, player_a, surface, day)
    b_overall, b_surface = ratings_on_day(elo, player_b, surface, day)
    ratings = pd.DataFrame([{"winner_elo": a_overall, "loser_elo": b_overall,
                             "winner_surface_elo": a_surface, "loser_surface_elo": b_surface}])
    difference = float(combined_difference(ratings, ELO_WEIGHT).iloc[0])

    def age(player):
        dob = table.at[player, "dob"]
        return (match_date - dob).days / 365.25 if pd.notna(dob) else math.nan

    row = {
        "winner_id": player_a, "loser_id": player_b, "surface": surface, "day": day,
        "tourney_id": "future", "match_num": 0, "source": "future", "tourney_name": tourney_name,
        "year": match_date.year, "tourney_level": level, "ended_early": False,
        "winner_rank": table.at[player_a, "rank"], "loser_rank": table.at[player_b, "rank"],
        "winner_rank_points": table.at[player_a, "points"], "loser_rank_points": table.at[player_b, "points"],
        "winner_age": age(player_a), "loser_age": age(player_b),
        "winner_seed": float(seed_a) if seed_a else math.nan, "loser_seed": float(seed_b) if seed_b else math.nan,
        "winner_entry": entry_a, "loser_entry": entry_b,
        "best_of": best_of, "draw_size": 32, "round_order": ROUND_ORDER.get(round_code, 14),
    }
    record = match_record(snapshot["feature_state"], row, difference, a_overall - b_overall, a_surface - b_surface)

    name = tourney_name.replace("Us Open", "US Open")
    domain = snapshot["domain_players"]
    a_history = tournament_history(domain[player_a], name, match_date.year)
    b_history = tournament_history(domain[player_b], name, match_date.year)
    for key in a_history:
        record["d_" + key] = float(a_history[key]) - float(b_history[key])

    features = pd.DataFrame([record])
    direction = [c for c in features.columns if c.startswith("d_")]
    features[direction] = features[direction].astype(float).round(8)
    if stale:
        features[SCHEDULE_COLUMNS] = np.nan
    features = add_interactions(features)
    probability = float(predict_winner_probability(snapshot["model"], features[snapshot["columns"]])[0])
    return {
        "probability_a": probability,
        "elo_probability_a": float(win_probability(difference, 0)),
        "stale": stale,
        "elo_a": a_overall, "elo_b": b_overall,
        "head_to_head": (snapshot["feature_state"].head_to_head[(player_a, player_b)],
                         snapshot["feature_state"].head_to_head[(player_b, player_a)]),
        "features": features,
    }
