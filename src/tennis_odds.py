"""Load bookmaker odds and link them to our matches, using player names and dates."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Team events and the Olympics aren't in the odds files
EVENTS_WITHOUT_ODDS = {"United Cup", "Atp Cup", "Laver Cup", "NextGen Finals", "Next Gen Finals"}

# Our round names, and the names each can have in the odds files
SAME_ROUND = {
    "R128": {"1st Round"}, "R64": {"1st Round", "2nd Round"},
    "R32": {"1st Round", "2nd Round", "3rd Round"}, "R16": {"1st Round", "2nd Round", "3rd Round", "4th Round"},
    "QF": {"Quarterfinals"}, "SF": {"Semifinals"}, "F": {"The Final"}, "RR": {"Round Robin"},
}

BOOKMAKERS = [("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL")]

# A match can be played up to this many days after its tournament's start date
MAX_DAYS_AFTER_START = 21
MAX_DAYS_BEFORE_START = 3


def clean_name(text):
    """Lowercase a name and remove accents and punctuation."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", text.lower()).split()


def odds_name_parts(name):
    """Split 'Ugo Carabelli C.' into its surname words and first initial."""
    words = clean_name(name)
    if not words:
        return (), ""
    # Initials are single letters at the end: "Herbert P.H." ends with p and h
    position = len(words)
    while position > 1 and len(words[position - 1]) == 1:
        position -= 1
    initial = words[position][0] if position < len(words) else ""
    return tuple(words[:position]), initial


def same_player(odds_surname, odds_initial, surname, first_name):
    """True if the name on the odds file fits a player's surname and first name."""
    our_surname = clean_name(surname)
    our_first = clean_name(first_name)
    if odds_initial and our_first and our_first[0][0] != odds_initial:
        return False
    if tuple(our_surname) == odds_surname:
        return True
    # Joined or shortened surnames: "Bautista Agut" and "Agut", or "Mc Donald" and "Mcdonald"
    return (
        "".join(our_surname) == "".join(odds_surname)
        or set(odds_surname) <= set(our_surname)
        or set(our_surname) <= set(odds_surname)
    )


def load_odds(odds_dir, first_year=None, last_year=None):
    """Return the odds files for these years as one table, keeping matches that were actually played.

    Files are named by year, and files up to 2012 are .xls, which needs the xlrd package.
    """
    frames = []
    for path in sorted(Path(odds_dir).iterdir()):
        # Excel leaves a temporary "~$" file next to any file it has open
        if path.suffix not in (".xls", ".xlsx") or path.name.startswith("~$"):
            continue
        year = int(path.stem)
        if (first_year is None or year >= first_year) and (last_year is None or year <= last_year):
            frames.append(pd.read_excel(path))
    odds = pd.concat(frames, ignore_index=True)
    for column in [c for pair in BOOKMAKERS for c in pair] + ["MaxW", "MaxL"]:
        if column not in odds:
            odds[column] = float("nan")
    odds["Date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds[(odds["Comment"] != "Walkover") & odds["Date"].notna()].reset_index(drop=True)
    return clean_odds(odds)


def clean_odds(odds):
    """Remove impossible prices. These checks never look at who won, so they can't favour a strategy."""
    for winner_col, loser_col in BOOKMAKERS:
        winner = pd.to_numeric(odds[winner_col], errors="coerce")
        loser = pd.to_numeric(odds[loser_col], errors="coerce")
        # Odds of 1 or less can't pay out, and a real price always has a margin of 0-25%
        total = 1 / winner.where(winner > 1) + 1 / loser.where(loser > 1)
        valid = total.between(1.0, 1.25)
        odds[winner_col] = winner.where(valid)
        odds[loser_col] = loser.where(valid)

    # The best price is compared across bookmakers, so it can have no margin at all
    for column in ["MaxW", "MaxL"]:
        price = pd.to_numeric(odds[column], errors="coerce")
        odds[column] = price.where(price > 1)
    return odds


def link_odds(matches, odds):
    """Find the odds row for each match; each odds row can only be used once.

    matches needs: start_date, round, tourney_name, winner_surname, winner_first, loser_surname, loser_first.
    The match's winner and loser must both fit, so a match can't be linked the wrong way round.
    """
    parsed = odds[["Date", "Round"]].copy()
    parsed[["w_surname", "w_initial"]] = pd.DataFrame(
        odds["Winner"].map(odds_name_parts).tolist(), index=odds.index)
    parsed[["l_surname", "l_initial"]] = pd.DataFrame(
        odds["Loser"].map(odds_name_parts).tolist(), index=odds.index)

    # Indexing by the surname's last letters would miss spelling variants, so search by date
    used = set()
    found = {}
    for match in matches.sort_values("start_date").itertuples():
        if match.tourney_name in EVENTS_WITHOUT_ODDS or "Olympics" in match.tourney_name:
            continue
        rounds = SAME_ROUND.get(match.round, set())
        low = match.start_date - pd.Timedelta(days=MAX_DAYS_BEFORE_START)
        high = match.start_date + pd.Timedelta(days=MAX_DAYS_AFTER_START)
        window = parsed[(parsed["Date"] >= low) & (parsed["Date"] <= high)]
        options = [
            row.Index for row in window.itertuples()
            if row.Index not in used
            and row.Round in rounds
            and same_player(row.w_surname, row.w_initial, match.winner_surname, match.winner_first)
            and same_player(row.l_surname, row.l_initial, match.loser_surname, match.loser_first)
        ]
        if options:
            # Two players can meet twice in a window, so take the date nearest to the start
            best = min(options, key=lambda i: abs((parsed.at[i, "Date"] - match.start_date).days))
            used.add(best)
            found[match.Index] = best
    return pd.Series(found, name="odds_row")
