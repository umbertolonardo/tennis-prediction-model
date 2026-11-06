"""The app's look: its stylesheet, plus the pieces of HTML that Streamlit has no ready-made component for."""
from html import escape
from pathlib import Path

import streamlit as st

from tennis_calendar import FINISHED, LIVE, UPCOMING, WAITING, date_label

PLAYER_COLORS = ["#2a78d6", "#c2571a"]
STYLE = Path(__file__).with_name("style.css")
STATUS = {FINISHED: ("Finished", "done"), LIVE: ("In progress", "live"), UPCOMING: ("Upcoming", "upcoming"),
          WAITING: ("Waiting for results", "waiting")}
SURFACES = {"Hard": "hard", "Hard (indoor)": "indoor", "Clay": "clay", "Grass": "grass"}
MAJOR_LEVELS = {"Grand Slam", "ATP Finals"}


def show(html):
    st.markdown(html, unsafe_allow_html=True)


def apply_style():
    show(f"<style>{STYLE.read_text()}</style>")


def hero(title, eyebrow, subtitle):
    show(f'<div class="hero"><div class="hero-eyebrow">{escape(eyebrow)}</div>'
         f'<div class="hero-title" role="heading" aria-level="1">{escape(title)}</div>'
         f'<div class="hero-sub">{escape(subtitle)}</div></div>')


def scoreboard(names, chance_a, context, footer):
    """The model's chances and fair odds for both players."""
    chances = [chance_a, 1 - chance_a]
    rows = ""
    for i in range(2):
        leader = " board-leader" if chances[i] > chances[1 - i] else ""
        rows += (f'<div class="board-row{leader}"><div class="board-name">'
                 f'<span class="board-swatch" style="background:{PLAYER_COLORS[i]}"></span><span>{escape(names[i])}</span></div>'
                 f'<div class="board-odds">{1 / chances[i]:.2f}</div><div class="board-chance">{100 * chances[i]:.1f}%</div></div>')
    show(f'<div class="board"><div class="board-top"><span class="board-title">Model\'s view</span>'
         f'<span class="board-context">{escape(context)}</span></div>'
         f'<div class="board-head"><span></span><span>Fair odds</span><span>Chance</span></div>{rows}'
         f'<div class="board-foot">{escape(footer)}</div></div>')


def _surface(name):
    return f'<span class="surface surface-{SURFACES.get(name, "hard")}">{escape(str(name))}</span>'


def _track(share, text):
    return (f'<div class="track"><div class="track-fill" style="width:{100 * min(share, 1):.0f}%"></div></div>'
            f'<div class="track-text">{escape(text)}</div>')


def _match_progress(event):
    if not event["expected"]:
        return ""
    return _track(event["progress"], f"{event['stage']}: {event['played']} of {event['expected']} matches played")


def calendar_banners(live_events, next_event, when):
    """Cards for the events being played now and the next one to start."""
    cards = ""
    for _, event in live_events.iterrows():
        cards += (f'<div class="banner banner-live"><div class="banner-label">Now playing</div>'
                  f'<div class="banner-name">{escape(event["name"])}</div>'
                  f'<div class="banner-meta">{escape(event["level"])} · {escape(event["surface"])}</div>{_match_progress(event)}</div>')
    if next_event is not None:
        place = f"{next_event['city']} · " if next_event["city"] else ""
        cards += (f'<div class="banner"><div class="banner-label">Next, {escape(when)}</div>'
                  f'<div class="banner-name">{escape(next_event["name"])}</div>'
                  f'<div class="banner-meta">{escape(place)}{escape(date_label(next_event))}</div></div>')
    if cards:
        show(f'<div class="banners">{cards}</div>')


def _event_row(event):
    label, kind = STATUS[event["status"]]
    has_winner = event["status"] == FINISHED and isinstance(event["winner"], str)
    result = f'<small>Winner</small>{escape(event["winner"])}' if has_winner else f'<span class="pill pill-{kind}">{label}</span>'
    live = _match_progress(event) if event["status"] == LIVE else ""
    highlight = " event-live" if event["status"] == LIVE else " event-major" if event["level"] in MAJOR_LEVELS else ""
    return (f'<div class="event{highlight}"><div class="event-date">{escape(date_label(event))}</div>'
            f'<div><div class="event-name">{escape(event["name"])}</div>'
            f'<div class="event-meta"><span>{escape(event["level"])}</span>{_surface(event["surface"])}</div>{live}</div>'
            f'<div class="event-result">{result}</div></div>')


def season_list(events):
    """Events grouped by the month they start in."""
    events = events.sort_values("start")
    html = ""
    for month, group in events.groupby(events["start"].dt.to_period("M")):
        html += f'<div class="month">{month.start_time:%B}</div>'
        html += "".join(_event_row(event) for _, event in group.iterrows())
    show(f'<div class="season">{html}</div>')
