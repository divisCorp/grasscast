from datetime import date

import numpy as np
import pandas as pd

from grasscast.planner import (candidate_windows, explain, novelty, pick_best, render_ics, render_text,
                               sweet_spot, verdict)
from tests.conftest import fake_hourly


def _forecast():
    w = pd.DataFrame(fake_hourly(date(2026, 10, 7), date(2026, 10, 9))["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time")


def test_candidates_are_daylight_only_and_in_future():
    c = candidate_windows(_forecast(), {"hike": 2.0, "run": 1.0}, now=pd.Timestamp("2026-10-07 12:00"))
    assert not c.empty
    assert (c["start_ts"] >= pd.Timestamp("2026-10-07 12:00")).all()
    ends = c["start_ts"] + pd.to_timedelta(c["duration_h"], unit="h")
    assert (c["start_ts"].dt.hour >= 7).all() and (ends.dt.hour <= 19).all()
    assert set(c["activity"]) == {"hike", "run"}


def _scored():
    c = candidate_windows(_forecast(), {"hike": 2.0}, now=pd.Timestamp("2026-10-07 00:00"))
    rng = np.random.default_rng(0)
    c["p_great"] = rng.random(len(c))
    c["score"] = 1 + 4 * c["p_great"]
    c["low"], c["high"] = c["score"] - 1, c["score"] + 0.5
    c["untested"] = ""
    return c


def test_pick_best_one_per_day_chronological():
    best = pick_best(_scored())
    assert best["start_ts"].dt.date.is_unique
    assert best["start_ts"].is_monotonic_increasing
    assert best["top_pick"].sum() == 1
    assert best.loc[best["top_pick"], "p_great"].iloc[0] == best["p_great"].max()


def test_untested_windows_are_demoted():
    s = _scored()
    top = s["p_great"].idxmax()
    s.loc[top, "untested"] = "rain risk this high"
    best = pick_best(s)
    assert s.loc[top, "start_ts"] not in set(best.loc[best["top_pick"], "start_ts"])


def test_novelty_and_explain():
    table = pd.DataFrame({"feels_c": [10, 12, 14, 16, 18.0], "wind_kmh": [5, 6, 7, 8, 9.0],
                          "gust_kmh": [10, 11, 12, 13, 14.0], "precip_mm": [0.0] * 5,
                          "rain_chance": [0, 5, 10, 10, 20.0], "cloud": [10, 20, 30, 40, 50.0],
                          "sun_wm2": [400, 500, 600, 700, 800.0], "rating": [4, 5, 4, 5, 4]})
    row = pd.Series({"feels_c": 14, "wind_kmh": 30, "gust_kmh": 12, "precip_mm": 0, "rain_chance": 90,
                     "cloud": 30, "sun_wm2": 600})
    assert novelty(row, table) == ["this windy", "rain risk this high"]
    assert "windier than you like" in explain(row, sweet_spot(table))
    assert explain(pd.Series({**row, "wind_kmh": 7, "rain_chance": 10}), sweet_spot(table)) == "right in your sweet spot"
    assert [verdict(p) for p in (0.9, 0.5, 0.2, 0.05)] == ["GO", "decent", "meh", "skip"]


def test_renderers():
    best = pick_best(_scored())
    spot = {"feels_c": (5.0, 20.0), "wind_kmh": (0, 15.0), "gust_kmh": (0, 30.0), "precip_mm": (0, 0.0),
            "rain_chance": (0, 30.0), "cloud": (0, 60.0), "sun_wm2": (100, 800.0)}
    txt = render_text(best, spot, n_outings=40)
    assert "learned from 40 logged outings" in txt and "Built with PriorLabs-TabPFN" in txt
    ics = render_ics(best, spot, min_p=0.0)
    assert ics.startswith("BEGIN:VCALENDAR") and ics.rstrip().endswith("END:VCALENDAR")
    assert ics.count("BEGIN:VEVENT") == len(best)
    assert "\r\n" in ics
    assert all(len(l.encode()) <= 75 for l in ics.split("\r\n"))
