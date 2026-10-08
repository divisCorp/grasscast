"""Hourly weather from Open-Meteo (free, no API key), with an on-disk cache.

Past outings use Open-Meteo's *historical forecast* archive rather than the
reanalysis archive. That way the model is trained on the same kind of numbers
it will see at prediction time (a forecast), not on hindsight-perfect data.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "is_day",
    "precipitation_probability",  # not in the reanalysis fallback; becomes NaN there
]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # fallback

DEFAULT_CACHE = Path(os.environ.get("GRASSCAST_CACHE", ".grasscast/cache"))


class WeatherError(RuntimeError):
    pass


@dataclass(frozen=True)
class Place:
    lat: float
    lon: float
    timezone: str = "auto"


def _cache_path(cache_dir: Path, url: str, params: dict) -> Path:
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:20]
    return cache_dir / f"{key}.json"


def _chaos() -> None:
    """Fault injection for demos/tests: GRASSCAST_CHAOS=0.5 fails ~50% of calls."""
    p = float(os.environ.get("GRASSCAST_CHAOS", "0") or 0)
    if p and random.random() < p:
        raise WeatherError(f"chaos monkey: simulated flaky network (GRASSCAST_CHAOS={p})")


def _get_json(url: str, params: dict, *, cache_dir: Path, offline: bool, max_age_s: float | None) -> dict:
    path = _cache_path(cache_dir, url, params)
    if path.exists():
        fresh = max_age_s is None or (time.time() - path.stat().st_mtime) < max_age_s
        if fresh or offline:
            return json.loads(path.read_text())
    if offline:
        raise WeatherError(f"offline and no cached response for {url}")
    _chaos()
    try:
        r = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:  # network down, DNS, timeout...
        raise WeatherError(str(e)) from e
    if r.status_code != 200:
        raise WeatherError(f"{url} -> HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _to_frame(data: dict) -> pd.DataFrame:
    h = data.get("hourly")
    if not h or "time" not in h:
        raise WeatherError("response has no hourly data")
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def fetch_history(place: Place, start: date, end: date, *, cache_dir: Path = DEFAULT_CACHE,
                  offline: bool = False) -> pd.DataFrame:
    """Hourly weather (as forecast at the time) between two dates, local time."""
    params = {
        "latitude": place.lat,
        "longitude": place.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": place.timezone,
    }
    try:
        # Past days never change, so the cache never expires.
        return _to_frame(_get_json(HISTORY_URL, params, cache_dir=cache_dir, offline=offline, max_age_s=None))
    except WeatherError:
        if offline:
            raise
        # Historical-forecast archive only goes back to ~2022; fall back to reanalysis.
        params["hourly"] = ",".join(v for v in HOURLY_VARS if v != "precipitation_probability")
        return _to_frame(_get_json(ARCHIVE_URL, params, cache_dir=cache_dir, offline=offline, max_age_s=None))


def fetch_forecast(place: Place, days: int = 7, *, cache_dir: Path = DEFAULT_CACHE,
                   offline: bool = False) -> pd.DataFrame:
    """Hourly forecast for the next `days` days. Cached for 1 hour (or forever when offline)."""
    params = {
        "latitude": place.lat,
        "longitude": place.lon,
        "forecast_days": days,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": place.timezone,
    }
    return _to_frame(_get_json(FORECAST_URL, params, cache_dir=cache_dir, offline=offline, max_age_s=3600))
