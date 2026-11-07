"""Rebuild the file the app reads, after running src/download_tml.py:

    python app/build_snapshot.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tennis_predict import build_snapshot

snapshot, _ = build_snapshot(ROOT / "data/raw", ROOT / "reference/tournament_info.csv",
                             ROOT / "app/app_snapshot.joblib")
print(f"Saved {len(snapshot['players']):,} players, with data up to {snapshot['last_date'].date()}.")
