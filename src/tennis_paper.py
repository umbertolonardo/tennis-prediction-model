"""A paper-trading log for the frozen seed rule: no real money, one row per imaginary bet."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["logged_on", "match_date", "tour", "player", "opponent", "seed", "opponent_seed", "bookmaker",
           "odds_taken", "opponent_odds", "edge", "closing_odds", "result"]
RESULTS = ["", "Won", "Lost", "Void"]


def load_log(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(path)
    log["result"] = log["result"].fillna("")
    return log[COLUMNS]


def save_log(log, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log[COLUMNS].to_csv(path, index=False)


def add_bet(log, match_date, tour, player, opponent, seed, opponent_seed, bookmaker, odds_taken, opponent_odds, edge):
    row = {"logged_on": date.today().isoformat(), "match_date": str(match_date), "tour": tour, "player": player,
           "opponent": opponent, "seed": seed, "opponent_seed": opponent_seed, "bookmaker": bookmaker,
           "odds_taken": odds_taken, "opponent_odds": opponent_odds, "edge": round(edge, 4),
           "closing_odds": np.nan, "result": ""}
    return pd.concat([log.astype(object), pd.DataFrame([row]).astype(object)], ignore_index=True)


def summarise(log):
    """Profit counts settled bets. Closing-line value compares the odds taken with the final odds."""
    settled = log[log["result"].isin(["Won", "Lost"])]
    odds = pd.to_numeric(settled["odds_taken"], errors="coerce")
    profit = np.where(settled["result"] == "Won", odds - 1, -1.0)
    closing = pd.to_numeric(log["closing_odds"], errors="coerce")
    taken = pd.to_numeric(log["odds_taken"], errors="coerce")
    has_closing = closing.notna() & (closing > 1)
    clv = taken[has_closing] / closing[has_closing] - 1
    return {
        "logged": len(log), "settled": len(settled), "profit_units": float(profit.sum()),
        "roi_pct": 100 * float(profit.mean()) if len(settled) else None,
        "with_closing": int(has_closing.sum()),
        "clv_pct": 100 * float(clv.mean()) if has_closing.any() else None,
        "beat_close_pct": 100 * float((clv > 0).mean()) if has_closing.any() else None,
    }
