"""Convert TennisMyLife's 2026 files to our format, to extend the match data after Sackmann's last matches."""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Sackmann already has these, so TennisMyLife only adds what starts later
LAST_MAIN_DATE = 20260525
LAST_OTHER_DATE = 20260601

LEVELS = {"250": "A", "500": "A", "A": "A", "M": "M", "G": "G", "D": "D", "F": "F", "C": "C"}
TOURNAMENT_NAMES = {"'s-Hertogenbosch": "s Hertogenbosch", "Shanghai": "Shanghai Masters", "US Open": "Us Open"}
# Names the odds files write differently, found by comparing names only
ODDS_FILE_NAMES = {("Adolfo Daniel", "Vallejo"): ("Daniel", "Vallejo"), ("Bu", "Yunchaokete"): ("Yunchaokete", "Bu"),
                   ("Tomas", "Barrios Vera"): ("Marcelo", "Barrios Vera"), ("Juan Alejandro", "Hernandez Serrano"): ("Alejandro", "Hernandez Serrano")}
FIRST_NEW_ID = 90_000_000  # Far above Sackmann's ids, so a new player can't clash with an existing one
MAX_AGE_GAP = 1.5  # Years. Stops a same-name link between two different people
# Same person by name, age and country. I0A0 and I0B0 are both Andre Ilagan; the others use another first or last name.
CHECKED_BY_HAND = {"I0A0": 210318, "MW62": 200125, "P0I5": 212306}


def _clean_name(name):
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join("".join(c for c in text if c.isalpha() or c == " ").split())


def _clean_tourney_id(tourney_id):
    """Turn 2026-407 into Sackmann's 2026-0407. Davis Cup ids are already long, so they stay as they are."""
    year, _, rest = str(tourney_id).partition("-")
    return f"{year}-{rest.zfill(4)}" if rest.isdigit() else str(tourney_id)


def _read(path):
    return pd.read_csv(path, low_memory=False, dtype={"winner_id": str, "loser_id": str, "tourney_level": str})


def _read_tml(raw_dir):
    tml = Path(raw_dir) / "tml"
    main = pd.concat([_read(tml / "2026.csv"), _read(tml / "ongoing_tourneys.csv")], ignore_index=True)
    assert not main.duplicated(["tourney_id", "match_num"]).any(), "a match appears in both main files"
    main["first_date"] = main.groupby("tourney_id")["tourney_date"].transform("min")
    return main, _read(tml / "2026_challenger.csv"), _read(tml / "2026_atp_quali.csv")


def _pairs_from_matches(tml, old):
    """Link ids using matches in both sources. Same players, same round, close dates: certainly the same match."""
    both = tml.merge(old, on=["w", "l", "round"], suffixes=("_t", "_s"))
    gap = (pd.to_datetime(both["tourney_date_t"].astype(str)) - pd.to_datetime(both["tourney_date_s"].astype(str))).dt.days
    both = both[gap.between(-10, 21)]
    return pd.concat([
        both[["winner_id_t", "winner_id_s"]].set_axis(["tml_id", "player_id"], axis=1),
        both[["loser_id_t", "loser_id_s"]].set_axis(["tml_id", "player_id"], axis=1),
    ]).drop_duplicates()


def _pairs_from_opponents(tml, old, known):
    """Link players whose names differ: same event, round and score, against an opponent who is already linked."""
    pairs = []
    for mine, other in [("winner", "loser"), ("loser", "winner")]:
        rows = tml.assign(key=tml[f"{other}_id"].map(known)).dropna(subset=["key"])
        rows = rows[~rows[f"{mine}_id"].isin(known)]
        merged = rows.merge(old, left_on=["event", "round", "score", "key"], right_on=["event", "round", "score", f"{other}_id"], suffixes=("_t", "_s"))
        pairs.append(merged[[f"{mine}_id_t", f"{mine}_id_s"]].set_axis(["tml_id", "player_id"], axis=1))
    return pd.concat(pairs).drop_duplicates()


def _prepare_overlap(raw_dir, main, challenger, qualifying):
    tml = pd.concat([main[main["tourney_date"] <= LAST_OTHER_DATE + 7], challenger[challenger["tourney_date"] <= LAST_OTHER_DATE],
                     qualifying[qualifying["tourney_date"] <= LAST_OTHER_DATE]], ignore_index=True)
    columns = ["tourney_id", "tourney_date", "round", "score", "winner_id", "loser_id", "winner_name", "loser_name"]
    old = pd.concat([pd.read_csv(Path(raw_dir) / f, usecols=columns, low_memory=False)
                     for f in ["atp_matches_2026.csv", "atp_matches_qual_chall_2026.csv"]], ignore_index=True)
    for frame in (tml, old):
        frame["w"] = frame["winner_name"].map(_clean_name)
        frame["l"] = frame["loser_name"].map(_clean_name)
        frame["event"] = frame["tourney_id"].map(_clean_tourney_id)
        frame["score"] = frame["score"].astype(str).str.strip()
    return tml, old


def _name_candidates(players):
    table = players.assign(clean=(players["name_first"].fillna("") + " " + players["name_last"].fillna("")).map(_clean_name))
    return {name: part for name, part in table.groupby("clean")}


def _age_fits(candidate, date, age):
    """A missing birth date or age can't disprove a link, so it doesn't reject one."""
    if pd.isna(candidate["dob"]) or pd.isna(age):
        return True
    born = pd.to_datetime(str(int(candidate["dob"])), format="%Y%m%d", errors="coerce")
    if pd.isna(born):
        return True
    return abs((pd.to_datetime(str(date)) - born).days / 365.25 - age) <= MAX_AGE_GAP


def _link_by_name(tml_id, names, date, age, candidates, taken, recent_matches):
    """Link only when one known person has this name and fits the age.

    Sackmann sometimes lists one person under two ids, with the same birth date. We then use the id with recent matches.
    """
    found = {}
    for name in names:
        for candidate in candidates.get(_clean_name(name), pd.DataFrame()).to_dict("records"):
            if candidate["player_id"] not in taken and _age_fits(candidate, date, age):
                found[candidate["player_id"]] = candidate
    if len({found[p]["dob"] for p in found}) != 1 or any(pd.isna(found[p]["dob"]) for p in found):
        return next(iter(found)) if len(found) == 1 else None
    return max(found, key=lambda p: (recent_matches.get(p, 0), p))


def _appearances(frames):
    """Each TennisMyLife player's name, age and details, from their latest match."""
    columns = {"id": "tml_id", "name": "name", "age": "age", "hand": "hand", "ht": "height", "ioc": "ioc"}
    long = pd.concat([
        frame[[f"{side}_{c}" for c in columns] + ["tourney_date"]].set_axis(list(columns.values()) + ["date"], axis=1)
        for frame in frames for side in ("winner", "loser")
    ])
    return long.sort_values("date", kind="stable").groupby("tml_id").last().reset_index()


def link_players(raw_dir):
    """Return one row per TennisMyLife player, with our id and how the link was made."""
    raw_dir = Path(raw_dir)
    main, challenger, qualifying = _read_tml(raw_dir)
    players = pd.read_csv(raw_dir / "atp_players.csv", low_memory=False)
    tml, old = _prepare_overlap(raw_dir, main, challenger, qualifying)

    # A link is kept only if it's one-to-one, since a person has one id in each source
    pairs = _pairs_from_matches(tml, old)
    for _ in range(3):
        known = dict(zip(pairs["tml_id"], pairs["player_id"]))
        pairs = pd.concat([pairs, _pairs_from_opponents(tml, old, known)]).drop_duplicates()
    unique = (pairs.groupby("tml_id")["player_id"].transform("nunique") == 1) & (pairs.groupby("player_id")["tml_id"].transform("nunique") == 1)
    links = {row.tml_id: (int(row.player_id), "same matches") for row in pairs[unique].itertuples()}

    database = pd.read_csv(raw_dir / "tml/ATP_Database.csv", dtype=str).set_index("id")
    candidates = _name_candidates(players)
    people = _appearances([main, challenger, qualifying])
    taken = {player_id for player_id, _ in links.values()}
    recent = pd.concat([pd.read_csv(path, usecols=["winner_id", "loser_id"], low_memory=False).stack()
                        for path in raw_dir.glob("atp_matches*202[56].csv")]).value_counts().to_dict()
    next_id = FIRST_NEW_ID
    rows = []
    for person in people.sort_values("tml_id").to_dict("records"):
        tml_id = person["tml_id"]
        names = [person["name"]]
        if tml_id in database.index:
            names += [n for n in database.loc[tml_id, ["player", "atpname"]] if isinstance(n, str)]
        if tml_id in links:
            player_id, method = links[tml_id]
        elif tml_id in CHECKED_BY_HAND:
            player_id, method = CHECKED_BY_HAND[tml_id], "checked by hand"
        else:
            player_id = _link_by_name(tml_id, names, person["date"], person["age"], candidates, taken, recent)
            method = "unique name and age"
            if player_id is None:
                player_id, method = next_id, "new player"
                next_id += 1
            else:
                taken.add(player_id)
        rows.append({**person, "player_id": player_id, "method": method})
    return pd.DataFrame(rows)


def load_new_matches(raw_dir, links=None):
    """Return the matches Sackmann doesn't have, with our ids, levels, dates and tournament names."""
    main, challenger, qualifying = _read_tml(raw_dir)
    links = link_players(raw_dir) if links is None else links
    ids = dict(zip(links["tml_id"], links["player_id"]))

    # Sackmann dates every match with the tournament's start, and his pipeline estimates the day from the round
    main = main[main["first_date"] > LAST_MAIN_DATE].assign(tourney_date=lambda d: d["first_date"], source="main")
    starts = dict(zip(main["tourney_id"], main["tourney_date"]))
    challenger = challenger[challenger["tourney_date"] > LAST_OTHER_DATE].assign(source="qual_chall")
    # Rows labelled as a main-draw round in the qualifying file can't be placed, so they're left out
    qualifying = qualifying[qualifying["tourney_id"].isin(starts) & qualifying["round"].str.startswith("Q")].assign(source="qual_chall")
    qualifying["tourney_date"] = qualifying["tourney_id"].map(starts).astype(int)

    new = pd.concat([main, challenger, qualifying], ignore_index=True)
    new["tourney_id"] = new["tourney_id"].map(_clean_tourney_id)
    new["tourney_name"] = new["tourney_name"].str.strip().replace(TOURNAMENT_NAMES)
    is_challenger = new["tourney_level"] == "C"
    new.loc[is_challenger, "tourney_name"] = new.loc[is_challenger, "tourney_name"] + " CH"
    new["tourney_level"] = new["tourney_level"].map(LEVELS)
    assert new["tourney_level"].notna().all(), "unknown tournament level"
    for column in ["winner_seed", "loser_seed"]:
        new[column] = pd.to_numeric(new[column], errors="coerce")
    new["winner_id"] = new["winner_id"].map(ids).astype(int)
    new["loser_id"] = new["loser_id"].map(ids).astype(int)
    assert not new.duplicated(["tourney_id", "source", "match_num"]).any(), "duplicate match"
    return new


def names_for_odds_link(raw_dir, links=None):
    """New main-tour matches with the names and dates that link_odds needs."""
    raw_dir = Path(raw_dir)
    links = link_players(raw_dir) if links is None else links
    new = load_new_matches(raw_dir, links)
    new = new[new["source"] == "main"]
    players = players_with_new(raw_dir, links).drop_duplicates("player_id").set_index("player_id")
    table = pd.DataFrame({
        "tourney_id": new["tourney_id"], "match_num": new["match_num"], "round": new["round"], "tourney_name": new["tourney_name"],
        "start_date": pd.to_datetime(new["tourney_date"].astype(str), format="%Y%m%d"),
        "winner_first": new["winner_id"].map(players["name_first"]), "winner_surname": new["winner_id"].map(players["name_last"]),
        "loser_first": new["loser_id"].map(players["name_first"]), "loser_surname": new["loser_id"].map(players["name_last"]),
    }).reset_index(drop=True)
    for side in ("winner", "loser"):
        for (first, surname), (odds_first, odds_surname) in ODDS_FILE_NAMES.items():
            same = (table[f"{side}_first"] == first) & (table[f"{side}_surname"] == surname)
            table.loc[same, [f"{side}_first", f"{side}_surname"]] = [odds_first, odds_surname]
    return table


def players_with_new(raw_dir, links=None):
    """Our player table, plus the players who first appear in TennisMyLife."""
    raw_dir = Path(raw_dir)
    players = pd.read_csv(raw_dir / "atp_players.csv", low_memory=False)
    links = link_players(raw_dir) if links is None else links
    new = links[links["method"] == "new player"]
    first, _, last = new["name"].str.partition(" ").pipe(lambda parts: (parts[0], parts[1], parts[2]))
    database = pd.read_csv(raw_dir / "tml/ATP_Database.csv", dtype=str).set_index("id")
    born = pd.to_numeric(new["tml_id"].map(database["birthdate"].to_dict()), errors="coerce")
    extra = pd.DataFrame({"player_id": new["player_id"], "name_first": first, "name_last": last, "hand": new["hand"],
                          "dob": born.where(born.between(19000101, 20201231), np.nan), "ioc": new["ioc"], "height": new["height"]})
    return pd.concat([players, extra], ignore_index=True)
