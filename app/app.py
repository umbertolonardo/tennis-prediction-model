"""Tennis match predictor: win probabilities, a comparison with bookmakers, and a practice-bets log."""
import sys
from datetime import date
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]
from tennis_calendar import FINISHED, LIVE, UPCOMING, WAITING, build_calendar, date_label, load_events, progress_by_level
from tennis_market import edge, fair_probability, margin, seed_rule_bets
from tennis_paper import RESULTS, add_bet, load_log, save_log, summarise
from tennis_predict import LEVELS, ROUNDS, predict_match
from ui import PLAYER_COLORS, STATUS, apply_style, calendar_banners, hero, scoreboard, season_list

AUTHOR = "Umberto Lo Nardo"
CALENDAR_EVENTS = ROOT / "app/calendar_events.csv"
UPCOMING_EVENTS = ROOT / "reference/atp_calendar_2026_upcoming.csv"
SNAPSHOT = ROOT / "app/app_snapshot.joblib"
# The data folder is never published, and online everyone would share one log, so this tab only exists on your computer
SHOW_PRACTICE_BETS = (ROOT / "data").exists()
PAPER_LOG = ROOT / "data/paper_trades.csv"
ENTRIES = {"Direct entry": None, "Qualifier": "Q", "Wildcard": "WC", "Lucky loser": "LL", "Protected ranking": "PR"}
SEEDS = list(range(0, 33))
OTHER = "Other tournament"
OLD_RANK_DAYS = 30

st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")


@st.cache_resource
def load_snapshot(file_changed):
    """Reloaded when the file is updated."""
    return joblib.load(SNAPSHOT)


def player_label(table, player, last_day):
    """A rank is only known from a player's last match, so an old one shows its month."""
    row = table.loc[player]
    if pd.isna(row["rank"]):
        return row["name"]
    if last_day - row["rank_day"] > OLD_RANK_DAYS:
        month = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(row["rank_day"]))
        return f"{row['name']} (#{int(row['rank'])} in {month:%b %Y})"
    return f"{row['name']} (#{int(row['rank'])})"


def seed_choice(column, label, key):
    """Return the tournament seed, or 0 for a player without one."""
    choice = column.selectbox(label, SEEDS, key=key, format_func=lambda s: "No seed" if s == 0 else f"Seed {s}")
    return int(choice)


def probability_chart(sources):
    """Stacked bars comparing each source's chance for the two players."""
    rows = []
    for source, chance_a, name_a, name_b in sources:
        rows.append({"source": source, "player": name_a, "chance": chance_a, "start": 0.0, "end": chance_a - 0.003})
        rows.append({"source": source, "player": name_b, "chance": 1 - chance_a, "start": chance_a + 0.003, "end": 1.0})
    data = pd.DataFrame(rows)
    data["middle"] = (data["start"] + data["end"]) / 2
    data["label"] = data["chance"].map("{:.0%}".format)
    order = [source[0] for source in sources]
    names = [sources[0][2], sources[0][3]]
    y = alt.Y("source:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=180, labelFontSize=13, labelColor="#17261f"))
    bars = alt.Chart(data).mark_bar(size=30, cornerRadius=4).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, 1]), axis=None),
        x2="end:Q",
        y=y,
        color=alt.Color("player:N", scale=alt.Scale(domain=names, range=PLAYER_COLORS),
                        legend=alt.Legend(title=None, orient="top", labelFontSize=13, labelColor="#17261f")),
        tooltip=[alt.Tooltip("source:N", title="Who says"), alt.Tooltip("player:N", title="Player"),
                 alt.Tooltip("chance:Q", title="Chance of winning", format=".1%")],
    )
    labels = alt.Chart(data).mark_text(color="white", fontWeight="bold", fontSize=14).encode(x="middle:Q", y=y, text="label:N")
    return (bars + labels).properties(height=46 * len(sources) + 40).configure(font="Barlow").configure_view(stroke=None)


def compare_with_bookmakers(rows, probability_a, name_a, name_b):
    """One row per bookmaker that has both prices."""
    results = []
    for _, row in rows.iterrows():
        odds_a, odds_b = pd.to_numeric(row["Odds A"], errors="coerce"), pd.to_numeric(row["Odds B"], errors="coerce")
        if pd.isna(odds_a) or pd.isna(odds_b):
            continue
        if odds_a <= 1 or odds_b <= 1:
            st.warning(f"{row['Bookmaker']}: odds must be above 1.")
            continue
        results.append({
            "Bookmaker": row["Bookmaker"], "Bookmaker's cut": margin(odds_a, odds_b),
            f"Bookmaker's chance for {name_a}": fair_probability(odds_a, odds_b),
            f"Value on {name_a}": edge(probability_a, odds_a),
            f"Value on {name_b}": edge(1 - probability_a, odds_b),
        })
    return pd.DataFrame(results)


def predict_tab(snapshot):
    table = snapshot["players"].set_index("player_id")
    tournaments = snapshot["tournaments"].set_index("tourney_name")

    setup = st.container(border=True)
    left, right = setup.columns(2)
    label = lambda p: player_label(table, p, snapshot["last_day"])
    player_a = left.selectbox("Player A", table.index, index=0, format_func=label)
    player_b = right.selectbox("Player B", table.index, index=1, format_func=label)
    name_a, name_b = table.at[player_a, "name"], table.at[player_b, "name"]

    with setup.expander("Optional: tournament seeds"):
        st.caption("At a tournament, the best players get a number called a seed: Seed 1 is the top player in the "
                   "draw. You'll find it next to the name in the draw. If you don't know it, leave “No seed”.")
        left, right = st.columns(2)
        seed_a = seed_choice(left, f"{name_a}: seed", "seed_a")
        seed_b = seed_choice(right, f"{name_b}: seed", "seed_b")
        entry_a = ENTRIES[left.selectbox(f"{name_a}: how they entered", list(ENTRIES), key="entry_a")]
        entry_b = ENTRIES[right.selectbox(f"{name_b}: how they entered", list(ENTRIES), key="entry_b")]

    tournament = setup.selectbox("Tournament", [OTHER] + list(tournaments.index))
    known = tournament != OTHER
    level_codes = list(LEVELS.values())
    surfaces = ["Hard", "Clay", "Grass"]
    default_level = level_codes.index(tournaments.at[tournament, "level"]) if known else 2
    default_surface = tournaments.at[tournament, "surface"] if known else "Hard"
    columns = setup.columns(4)
    level = columns[0].selectbox("Level", list(LEVELS), index=default_level, key=f"level_{tournament}")
    surface = columns[1].selectbox(
        "Surface", surfaces, index=surfaces.index(default_surface) if default_surface in surfaces else 0,
        key=f"surface_{tournament}")
    round_name = columns[2].selectbox("Round", list(ROUNDS), index=1)
    best_of = columns[3].selectbox("Best of", [3, 5], index=1 if LEVELS[level] == "G" else 0, key=f"best_{level}")
    match_date = setup.date_input("Match date", value=date.today())

    if player_a == player_b:
        st.info("Choose two different players.")
        return

    result = predict_match(
        snapshot, player_a, player_b, surface, tournament, LEVELS[level], ROUNDS[round_name], best_of, match_date,
        seed_a=seed_a, seed_b=seed_b, entry_a=entry_a, entry_b=entry_b)
    probability_a = result["probability_a"]

    wins_a, wins_b = result["head_to_head"]
    context = " · ".join(([tournament] if known else []) + [level, surface, round_name, f"Best of {best_of}"])
    scoreboard([name_a, name_b], probability_a, context,
               f"Head to head in our data: {wins_a}–{wins_b}. Elo alone gives {name_a} {100 * result['elo_probability_a']:.1f}%.")
    st.caption("Fair odds are the odds with no bookmaker profit, if the model's chance is right.")
    if result["stale"]:
        st.warning(
            f"Our data ends on {snapshot['last_date'].date()}, so this ignores everything since then, including "
            "current form, injuries and ranking changes. Treat this as a rough guide.")

    st.subheader("Against the bookmakers")
    st.write("Type the odds from any betting site. **Odds A** are for "
             f"{name_a} and **Odds B** are for {name_b}.")
    if "bookmakers" not in st.session_state:
        st.session_state.bookmakers = ["Pinnacle", "Bet365", "Other"]
    entered = []
    for i, default in enumerate(st.session_state.bookmakers):
        columns = st.columns([2, 1, 1])
        # The keys include the players, so odds typed for one match never carry over to another
        name = columns[0].text_input("Bookmaker", value=default, key=f"book_{i}")
        odds_a = columns[1].number_input("Odds A", min_value=1.0, value=None, step=0.01, format="%.2f", key=f"a_{i}_{player_a}_{player_b}")
        odds_b = columns[2].number_input("Odds B", min_value=1.0, value=None, step=0.01, format="%.2f", key=f"b_{i}_{player_a}_{player_b}")
        entered.append({"Bookmaker": name, "Odds A": odds_a, "Odds B": odds_b})
    if st.button("Add a bookmaker"):
        st.session_state.bookmakers.append(f"Bookmaker {len(st.session_state.bookmakers) + 1}")
        st.rerun()
    rows = pd.DataFrame(entered).astype({"Odds A": "float64", "Odds B": "float64"})
    comparison = compare_with_bookmakers(rows, probability_a, name_a, name_b)

    chance_column = f"Bookmaker's chance for {name_a}"
    sources = [("Our model", probability_a, name_a, name_b)]
    if not comparison.empty:
        sources += [(row["Bookmaker"], row[chance_column], name_a, name_b) for _, row in comparison.iterrows()]
    st.markdown("**Who is more likely to win?**")
    st.altair_chart(probability_chart(sources), use_container_width=True)
    if comparison.empty:
        return

    percent = {column: "{:+.1%}" for column in comparison.columns if column.startswith("Value")}
    percent.update({"Bookmaker's cut": "{:.1%}", chance_column: "{:.1%}"})
    st.dataframe(comparison.style.format(percent), hide_index=True, use_container_width=True)
    st.caption("**Bookmaker's cut:** the bookmaker's built-in profit. **Value:** how much the model expects you'd gain "
               "or lose per bet, as a percentage of the stake. **A positive value is not a profit:** in our test on "
               "10,375 matches, the model was less accurate than Pinnacle, and betting on its 5%+ values lost 1.22%.")

    pinnacle = rows[rows["Bookmaker"].astype(str).str.lower().str.contains("pinnacle")]
    pinnacle = pinnacle.dropna(subset=["Odds A", "Odds B"])
    if len(pinnacle):
        odds_a, odds_b = float(pinnacle.iloc[0]["Odds A"]), float(pinnacle.iloc[0]["Odds B"])
        if odds_a > 1 and odds_b > 1:
            # The probability comes from Pinnacle, and the bet uses the highest odds entered above
            verdict = seed_rule_bets(odds_a, odds_b, seed_a, seed_b, rows["Odds A"].max(), rows["Odds B"].max())
            backed = name_a if verdict["bet_a"] else name_b if verdict["bet_b"] else None
            if backed:
                st.info(f"**Our test rule:** this would be a pretend bet on {backed}, at the highest odds entered."
                        + (" You can record it in the Practice bets tab." if SHOW_PRACTICE_BETS else ""))
            else:
                st.caption("Our test rule: no pretend bet on this match.")


def paper_tab():
    st.write("Pretend bets on future matches, **with no real money**, to see if our test rule works. Type the "
             "players, their odds at Pinnacle and their seeds: the app tells you if the rule would bet. It also works "
             "for women's matches.")
    log = load_log(PAPER_LOG)

    with st.form("new_bet", clear_on_submit=True):
        columns = st.columns(2)
        player = columns[0].text_input("Player A")
        opponent = columns[1].text_input("Player B")
        seed = seed_choice(columns[0], "Player A seed", "paper_seed_a")
        opponent_seed = seed_choice(columns[1], "Player B seed", "paper_seed_b")
        odds = columns[0].number_input("Player A odds at Pinnacle", 1.0, 100.0, 2.0, 0.01)
        opponent_odds = columns[1].number_input("Player B odds at Pinnacle", 1.0, 100.0, 2.0, 0.01)
        price = columns[0].number_input("Player A highest odds on any site (optional)", min_value=1.0, value=None, step=0.01, format="%.2f")
        opponent_price = columns[1].number_input("Player B highest odds on any site (optional)", min_value=1.0, value=None, step=0.01, format="%.2f")
        tour = columns[0].selectbox("Tour", ["ATP", "WTA"])
        match_date = columns[1].date_input("Match date", value=date.today())
        bookmaker = st.text_input("Site with the highest odds", value="Pinnacle")
        st.caption("Only type higher odds if you could really bet at them right now.")
        submitted = st.form_submit_button("Check the rule and save any bet")

    if submitted:
        if not player or not opponent or odds <= 1 or opponent_odds <= 1:
            st.error("Enter both names, and odds above 1.")
        else:
            taken_a, taken_b = price or odds, opponent_price or opponent_odds
            verdict = seed_rule_bets(odds, opponent_odds, seed, opponent_seed, taken_a, taken_b)
            if verdict["bet_a"]:
                log = add_bet(log, match_date, tour, player, opponent, seed, opponent_seed, bookmaker, taken_a, opponent_odds, verdict["edge_a"])
            if verdict["bet_b"]:
                log = add_bet(log, match_date, tour, opponent, player, opponent_seed, seed, bookmaker, taken_b, odds, verdict["edge_b"])
            if verdict["bet_a"] or verdict["bet_b"]:
                save_log(log, PAPER_LOG)
                st.success("Practice bet saved.")
            else:
                st.info(f"No bet: the values are {verdict['edge_a']:+.1%} and {verdict['edge_b']:+.1%}, and the rule needs +2.0%.")

    summary = summarise(log)
    columns = st.columns(4)
    columns[0].metric("Practice bets", summary["logged"])
    columns[1].metric("Finished", summary["settled"])
    columns[2].metric("Return", f"{summary['roi_pct']:+.1f}%" if summary["roi_pct"] is not None else "–")
    columns[3].metric("Beat final odds by", f"{summary['clv_pct']:+.1f}%" if summary["clv_pct"] is not None else "–")
    if len(log):
        st.dataframe(log, hide_index=True, use_container_width=True)
        st.caption("Add Pinnacle's final odds just before the match starts, and the result afterwards.")
        with st.form("update_bet"):
            columns = st.columns([3, 1, 1])
            row = columns[0].selectbox(
                "Bet", log.index, format_func=lambda i: f"{log.at[i, 'match_date']}: {log.at[i, 'player']} to beat {log.at[i, 'opponent']}")
            closing = columns[1].number_input("Final odds", min_value=1.0, value=None, step=0.01, format="%.2f")
            result = columns[2].selectbox("Result", RESULTS)
            if st.form_submit_button("Save"):
                if closing is not None:
                    log.at[row, "closing_odds"] = closing
                if result:
                    log.at[row, "result"] = result
                save_log(log, PAPER_LOG)
                st.rerun()
    st.caption("Thousands of bets are needed to prove a small return, so early on, look at whether you beat the final "
               "odds rather than at the profit. In the past, this rule at Pinnacle's odds returned +4.40% for men "
               "(2022–2025, not statistically confirmed) and −0.52% for women.")


@st.cache_data
def load_calendar(today, file_changed):
    """Rebuilt when the day changes or the summary is updated."""
    events = load_events(CALENDAR_EVENTS)
    return build_calendar(events, UPCOMING_EVENTS, today), events["last"].max()


def calendar_tab():
    if not CALENDAR_EVENTS.exists():
        st.info("The calendar is missing. Run `python src/download_tml.py`, then `python app/build_calendar.py`.")
        return
    today = pd.Timestamp(date.today())
    calendar, fresh_until = load_calendar(today, CALENDAR_EVENTS.stat().st_mtime)

    finished = int((calendar["status"] == FINISHED).sum())
    st.subheader("2026 season")
    st.progress(finished / len(calendar), text=f"**{finished} of {len(calendar)} events finished**")
    columns = st.columns(4)
    for column, level in zip(columns, progress_by_level(calendar)):
        column.metric(level["level"], f"{level['finished']} of {level['total']}")

    coming = calendar[calendar["status"] == UPCOMING]
    next_event, when = None, None
    if len(coming):
        next_event = coming.iloc[0]
        days = (next_event["start"] - today).days
        when = "tomorrow" if days == 1 else f"in {days} days"
    calendar_banners(calendar[calendar["status"] == LIVE], next_event, when)
    if (calendar["status"] == WAITING).any():
        st.warning(f"Some results are missing, because this calendar was last updated on {fresh_until.day} {fresh_until:%B}.")

    choice = st.radio("Show", ["All events", FINISHED, LIVE, UPCOMING], horizontal=True,
                      format_func=lambda option: STATUS[option][0] if option in STATUS else option)
    shown = calendar if choice == "All events" else calendar[calendar["status"] == choice]
    if shown.empty:
        st.info("No events to show.")
    else:
        season_list(shown)
    st.caption(f"Events on the ATP calendar. Davis Cup ties before the Finals aren't listed. Results up to "
               f"{fresh_until.day} {fresh_until:%B %Y}, from TennisMyLife. Dates of events still to come are written by "
               "hand from the ATP calendar, so they can change. Progress in a live event counts matches played.")


def about_tab(snapshot):
    st.subheader("Picked the winner, on 12,215 unseen matches")
    columns = st.columns(3)
    columns[0].metric("Our model", "66.98%")
    columns[1].metric("Elo rating", "65.87%")
    columns[2].metric("ATP ranking", "63.81%")
    st.markdown(f"""
**Made by {AUTHOR}.**

**What this is:** a statistical model that uses 60 pieces of information about each match, such as player strength,
serve and return, recent form, fatigue and past meetings. It learned from {snapshot['train_rows']:,} main-tour matches
since 2005.

**How good it is:** on 12,215 matches it had never seen (2022 to June 2026), it picked the winner 66.98% of the time.
That beats a simple strength rating called Elo (65.87%) and the ATP ranking (63.81%). On 1,253 newer matches (June to
November 2026) it scored 65.28%. The version you're using was then retrained with those matches too, so it hasn't been
tested on anything newer yet.

**What it can't do:** it doesn't beat the bookmakers. On the same 10,375 matches, Pinnacle was right 67.71% of the time
and the model 66.57%. It also knows nothing about news, injuries or anything after {snapshot['last_date'].date()}.

**Not betting advice.** This is a student project, built to learn. Please don't use it to bet real money.

Match data by Jeff Sackmann (CC BY-NC-SA 4.0). Matches since June 2026 and calendar results by TennisMyLife, for
non-commercial use. Betting odds from tennis-data.co.uk.
""")


apply_style()
if not SNAPSHOT.exists():
    st.error("The model file is missing. Run `python app/build_snapshot.py` first.")
    st.stop()
snapshot = load_snapshot(SNAPSHOT.stat().st_mtime)
last_date = snapshot["last_date"]
hero("Tennis Match Predictor", "ATP men's singles", f"Made by {AUTHOR}. Data up to {last_date.day} {last_date:%B %Y}.")
names = ["Predict a match", "Calendar"] + (["Practice bets"] if SHOW_PRACTICE_BETS else []) + ["About"]
tabs = dict(zip(names, st.tabs(names)))
with tabs["Predict a match"]:
    predict_tab(snapshot)
with tabs["Calendar"]:
    calendar_tab()
if SHOW_PRACTICE_BETS:
    with tabs["Practice bets"]:
        paper_tab()
with tabs["About"]:
    about_tab(snapshot)
