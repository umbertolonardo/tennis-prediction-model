"""Rebuild the calendar summary the app reads, after running src/download_tml.py:

    python app/build_calendar.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tennis_calendar import summarise_events

events = summarise_events(ROOT / "data/raw")
events.to_csv(ROOT / "app/calendar_events.csv", index=False)
print(f"Saved {len(events)} events, with results up to {events['last'].max().date()}.")
