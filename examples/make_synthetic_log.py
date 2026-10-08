"""Generate a SYNTHETIC outing log for demos and tests.

A made-up person ("Sam" in Salt Lake City) goes hiking, running and biking
from April to early October 2026. The weather is real (Open-Meteo history);
the ratings come from a hidden preference function plus noise. Because we
know the hidden function, we can check whether grasscast actually learns it.

    python examples/make_synthetic_log.py > examples/outings.sample.csv
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasscast.features import window_features  # noqa: E402
from grasscast.weather import Place, fetch_history  # noqa: E402

SLC = Place(40.7608, -111.8910, "America/Denver")
RNG = np.random.default_rng(7)

# activity: (duration_h, ideal feels-like °C, tolerance °C, wind dislike, rain dislike, likes sun)
TASTE = {
    "hike": (2.5, 15.0, 7.0, 0.04, 1.2, 0.0015),
    "run": (1.0, 9.0, 6.0, 0.03, 0.5, -0.0005),
    "bike": (1.5, 19.0, 6.0, 0.08, 1.5, 0.0010),
}


def hidden_rating(act: str, f: dict) -> int:
    _, ideal, tol, wind_k, rain_k, sun_k = TASTE[act]
    comfort = np.exp(-0.5 * ((f["feels_c"] - ideal) / tol) ** 2)
    score = 1.2 + 3.6 * comfort
    score -= wind_k * max(0.0, f["wind_kmh"] - 12)
    score -= rain_k * min(f["precip_mm"], 3)
    score -= 0.8 * rain_k * f["rain_chance"] / 100  # threatening skies take the fun out
    score += sun_k * f["sun_wm2"]
    score += RNG.normal(0, 0.35)
    return int(np.clip(round(score), 1, 5))


def schedule(start: date, end: date):
    for d in pd.date_range(start, end):
        wk = d.dayofweek >= 5
        if wk and RNG.random() < 0.45:
            yield "hike", d + pd.Timedelta(hours=int(RNG.choice([7, 8, 9, 10, 15, 16])))
        if not wk and RNG.random() < 0.22:
            yield "run", d + pd.Timedelta(hours=int(RNG.choice([6, 7, 12, 18, 19])))
        if RNG.random() < 0.08:
            yield "bike", d + pd.Timedelta(hours=int(RNG.choice([8, 10, 13, 17])))


def main() -> None:
    start, end = date(2026, 4, 1), date(2026, 10, 5)
    weather = fetch_history(SLC, start, end)
    rows = []
    for act, ts in schedule(start, end):
        dur = TASTE[act][0]
        f = window_features(weather, ts, dur)
        if f is None or f["daylight"] < 0.99:
            continue
        rows.append((ts.date().isoformat(), ts.strftime("%H:%M"), dur, act, hidden_rating(act, f)))
    print("# SYNTHETIC demo data: real Salt Lake City weather, made-up person, ratings from a hidden formula.")
    print("date,start,duration_h,activity,rating,notes")
    for r in rows:
        print(",".join(map(str, r)) + ",")


if __name__ == "__main__":
    main()
