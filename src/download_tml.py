"""Download the latest 2026 files from TennisMyLife, for personal and non-commercial use only.

These files must not be shared again, so they stay in data/, which Git ignores. Run from the project folder:
    python src/download_tml.py
A file is only replaced after its download has finished, so a failed download keeps the old one.
"""
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://stats.tennismylife.org/data/"
FILES = ["2026.csv", "2026_challenger.csv", "atp_quali/2026_atp_quali.csv", "ongoing_tourneys.csv", "ATP_Database.csv"]
FOLDER = Path(__file__).resolve().parent.parent / "data/raw/tml"
ATTEMPTS = 3
PAUSE_SECONDS = 5


def download(file):
    request = urllib.request.Request(BASE_URL + file, headers={"User-Agent": "tennis-prediction-project"})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.URLError:
            if attempt == ATTEMPTS:
                raise
            time.sleep(PAUSE_SECONDS)


FOLDER.mkdir(parents=True, exist_ok=True)
for file in FILES:
    content = download(file)
    if not content.startswith((b"tourney_id", b"id,")):
        raise ValueError(f"{file} doesn't look like the expected CSV, so the old copy was kept")
    (FOLDER / Path(file).name).write_bytes(content)
    print(f"Updated {Path(file).name} ({len(content) / 1024:,.0f} KB)")
