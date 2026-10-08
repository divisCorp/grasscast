"""The plan pipeline as plain functions, so the CLI and the Temporal workflow share it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .features import build_training_table, load_log
from .model import TasteModel
from .planner import activity_durations, candidate_windows, flag_novelty, pick_best, render_ics, render_text, sweet_spot
from .weather import DEFAULT_CACHE, Place, fetch_forecast, fetch_history


@dataclass
class Plan:
    best: pd.DataFrame
    text: str
    ics: str
    n_outings: int


def past_weather(place: Place, log: pd.DataFrame, *, cache_dir: Path = DEFAULT_CACHE, offline: bool = False) -> pd.DataFrame:
    start = log["start_ts"].min().date()
    end = (log["start_ts"] + pd.to_timedelta(log["duration_h"], unit="h")).max().date() + timedelta(days=1)
    return fetch_history(place, start, end, cache_dir=cache_dir, offline=offline)


def score_forecast(table: pd.DataFrame, forecast: pd.DataFrame, *, activity: str | None = None,
                   now: pd.Timestamp | None = None, seed: int = 0) -> pd.DataFrame:
    durations = activity_durations(table)
    if activity:
        if activity not in durations:
            raise ValueError(f"no logged outings for activity '{activity}' (have: {sorted(durations)})")
        durations = {activity: durations[activity]}
    cands = candidate_windows(forecast, durations, now=now)
    if cands.empty:
        return cands
    model = TasteModel(seed=seed).fit(table)
    return flag_novelty(pd.concat([cands, model.predict(cands)], axis=1), table)


def make_plan(log_path: str | Path, place: Place, *, days: int = 7, top: int = 7, activity: str | None = None,
              imperial: bool = True, offline: bool = False, cache_dir: Path = DEFAULT_CACHE,
              now: pd.Timestamp | None = None) -> Plan:
    log = load_log(log_path)
    table = build_training_table(log, past_weather(place, log, cache_dir=cache_dir, offline=offline))
    forecast = fetch_forecast(place, days, cache_dir=cache_dir, offline=offline)
    scored = score_forecast(table, forecast, activity=activity, now=now)
    best = pick_best(scored, top=top) if not scored.empty else scored
    spot = sweet_spot(table)
    return Plan(best=best, text=render_text(best, spot, imperial=imperial, n_outings=len(table)),
                ics=render_ics(best, spot, imperial=imperial), n_outings=len(table))
