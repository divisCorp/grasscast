"""Shared fixtures: a fake Open-Meteo so tests never touch the network."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from grasscast.weather import HOURLY_VARS


def fake_hourly(start: date, end: date, seed: int = 0) -> dict:
    """Plausible hourly weather: daily temperature cycle, daylight 07-19, some rain."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    n = len(times)
    hour = times.hour.to_numpy()
    day_idx = (times - times[0]).days.to_numpy()
    base = 12 + 8 * np.sin(day_idx / 9.0)  # weather systems come and go
    temp = base + 7 * np.sin((hour - 9) / 24 * 2 * np.pi)
    wind = np.abs(8 + 6 * np.sin(day_idx / 4.0) + rng.normal(0, 2, n))
    pop = np.clip(50 + 50 * np.sin(day_idx / 5.0 + 1), 0, 100)
    data = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": temp.round(1).tolist(),
        "apparent_temperature": (temp - wind / 6).round(1).tolist(),
        "relative_humidity_2m": np.clip(60 - temp, 10, 100).round().tolist(),
        "precipitation": np.where(pop > 80, 0.5, 0.0).tolist(),
        "cloud_cover": np.clip(pop + rng.normal(0, 10, n), 0, 100).round().tolist(),
        "wind_speed_10m": wind.round(1).tolist(),
        "wind_gusts_10m": (wind * 1.8).round(1).tolist(),
        "shortwave_radiation": np.where((hour >= 7) & (hour < 19), 600 - 4 * pop, 0).round().tolist(),
        "is_day": ((hour >= 7) & (hour < 19)).astype(int).tolist(),
        "precipitation_probability": pop.round().tolist(),
    }
    assert set(HOURLY_VARS) <= set(data)
    return {"hourly": data}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.text = json.dumps(payload)[:200]

    def json(self):
        return self._payload


@pytest.fixture
def fake_open_meteo(monkeypatch):
    """Patch requests.get inside grasscast.weather; records every call."""
    calls = []
    today = date(2026, 10, 7)

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params)))
        if "start_date" in params:
            s = date.fromisoformat(params["start_date"])
            e = date.fromisoformat(params["end_date"])
        else:
            s, e = today, today + timedelta(days=params.get("forecast_days", 7) - 1)
        return FakeResponse(fake_hourly(s, e))

    monkeypatch.setattr("grasscast.weather.requests.get", fake_get)
    return calls


def hidden_taste(feels_c: float, wind: float, pop: float) -> float:
    """The made-up 'person' the tests learn: likes ~14°C, hates wind and rain risk."""
    return 1 + 4 * np.exp(-0.5 * ((feels_c - 14) / 5) ** 2) - 0.08 * max(0, wind - 10) - 1.5 * pop / 100


@pytest.fixture
def sample_log(tmp_path):
    """A 40-outing log whose ratings follow hidden_taste() on the fake weather."""
    from grasscast.features import window_features

    w = pd.DataFrame(fake_hourly(date(2026, 6, 1), date(2026, 9, 30))["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    w = w.set_index("time")
    rng = np.random.default_rng(1)
    rows = ["# test data", "date,start,duration_h,activity,rating,notes"]
    days = pd.date_range("2026-06-01", "2026-09-29")
    for d in rng.choice(days, size=40, replace=False):
        d = pd.Timestamp(d)
        act = "hike" if rng.random() < 0.6 else "run"
        start = d + pd.Timedelta(hours=int(rng.choice([8, 10, 12, 15])))
        dur = 2.0 if act == "hike" else 1.0
        f = window_features(w, start, dur)
        r = int(np.clip(round(hidden_taste(f["feels_c"], f["wind_kmh"], f["rain_chance"])), 1, 5))
        rows.append(f"{start:%Y-%m-%d},{start:%H:%M},{dur},{act},{r},")
    p = tmp_path / "outings.csv"
    p.write_text("\n".join(rows) + "\n")
    return p
