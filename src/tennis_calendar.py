"""Build the 2026 ATP calendar: a summary of each event played, and events still to come from a table written by hand."""
from pathlib import Path

import pandas as pd

FINISHED = "✅ Finished"
LIVE = "🎾 In progress"
UPCOMING = "🗓️ Upcoming"
WAITING = "⏳ Waiting for results"

LEVELS = {"G": "Grand Slam", "M": "Masters 1000", "500": "ATP 500", "250": "ATP 250", "A": "Team event", "F": "ATP Finals"}
ROUND_NAMES = {"R128": "Round of 128", "R64": "Round of 64", "R32": "Round of 32", "R16": "Round of 16", "RR": "Round robin",
               "QF": "Quarterfinals", "SF": "Semifinals", "F": "Final"}
ROUND_ORDER = ["R128", "R64", "R32", "RR", "R16", "QF", "SF", "F"]
TEAM_EVENTS = {"United Cup", "Laver Cup"}  # Their last match doesn't decide a single champion
DAYS_BEFORE_WAITING = 3  # No match for this long, and no final, means the data is behind


def summarise_events(raw_dir):
    """One row per event from TennisMyLife's files. The app reads this summary, never the raw files."""
    tml = Path(raw_dir) / "tml"
    matches = pd.concat([pd.read_csv(tml / name, low_memory=False, dtype={"tourney_level": str})
                         for name in ["2026.csv", "ongoing_tourneys.csv"]], ignore_index=True)
    matches = matches[matches["tourney_level"] != "D"]
    matches = matches.assign(date=pd.to_datetime(matches["tourney_date"].astype(str), format="%Y%m%d"))
    rows = []
    for _, event in matches.groupby("tourney_id"):
        name = event["tourney_name"].iloc[0].strip()
        final = event[event["round"] == "F"]
        played_rounds = [r for r in ROUND_ORDER if (event["round"] == r).any()]
        rows.append({
            "name": name, "level": LEVELS.get(event["tourney_level"].iloc[0], "Other"),
            "surface": event["surface"].iloc[0] + (" (indoor)" if event["indoor"].iloc[0] == "I" else ""),
            "start": event["date"].min(), "last": event["date"].max(), "has_final": len(final) > 0,
            "winner": final["winner_name"].iloc[0] if len(final) and name not in TEAM_EVENTS else None,
            "stage": ROUND_NAMES.get(played_rounds[-1]) if played_rounds else None,
            "played": len(event), "expected": max(int(event["draw_size"].iloc[0]) - 1, len(event)),
        })
    return pd.DataFrame(rows)


def load_events(path):
    return pd.read_csv(path, parse_dates=["start", "last"])


def _played_rows(events, today):
    rows = []
    for event in events.to_dict("records"):
        quiet = (today - event["last"]).days >= DAYS_BEFORE_WAITING
        done = event["has_final"] or (event["name"] in TEAM_EVENTS and quiet)
        rows.append({**event, "end": event["last"] if done else pd.NaT, "city": "",
                     "status": FINISHED if done else WAITING if quiet else LIVE,
                     "progress": 1.0 if done else event["played"] / event["expected"]})
    return rows


def _upcoming_rows(upcoming_path, played_names, today):
    rows = []
    for row in pd.read_csv(upcoming_path, dtype=str).fillna("").to_dict("records"):
        pattern = row["data_pattern"]
        if pattern and any(pd.Series(played_names).str.lower().str.contains(pattern, regex=True)):
            continue  # The results now include this event
        start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
        status = UPCOMING if today < start else WAITING if today > end else LIVE
        rows.append({"name": row["name"], "city": row["city"], "level": row["level"], "surface": row["surface"], "start": start,
                     "end": end, "status": status, "progress": 0.0, "winner": None, "stage": None, "played": 0, "expected": 0})
    return rows


def build_calendar(events, upcoming_path, today):
    """Return one row per event in date order, with its status on this day."""
    today = pd.Timestamp(today)
    played = _played_rows(events, today)
    upcoming = _upcoming_rows(upcoming_path, events["name"].tolist(), today)
    return pd.DataFrame(played + upcoming).sort_values(["start", "name"]).reset_index(drop=True)


def date_label(row):
    if pd.isna(row["end"]):
        return f"From {row['start'].day} {row['start']:%b}"
    return f"{row['start'].day} {row['start']:%b} – {row['end'].day} {row['end']:%b}"


def progress_by_level(calendar):
    """How many events of each main level have finished."""
    rows = []
    for level in ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250"]:
        part = calendar[calendar["level"] == level]
        rows.append({"level": level, "finished": int((part["status"] == FINISHED).sum()), "total": len(part)})
    return rows
