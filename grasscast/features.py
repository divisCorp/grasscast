"""Turn an outing log + hourly weather into a small feature table."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "start", "duration_h", "activity", "rating"]

# Order matters: this is the column order the model sees.
WEATHER_FEATURES = [
    "temp_c",       # mean air temperature
    "feels_c",      # mean apparent ("feels like") temperature
    "humidity",     # mean relative humidity, %
    "precip_mm",    # total precipitation over the window
    "rain_chance",  # max precipitation probability during the window, %
    "cloud",        # mean cloud cover, %
    "wind_kmh",     # mean wind speed at 10 m
    "gust_kmh",     # max gust at 10 m
    "sun_wm2",      # mean shortwave radiation (how sunny it actually is)
    "daylight",     # fraction of the window in daylight
]
TIME_FEATURES = ["start_hour", "weekend"]
FEATURES = WEATHER_FEATURES + TIME_FEATURES + ["activity"]


class LogError(ValueError):
    pass


def load_log(path: str | Path) -> pd.DataFrame:
    """Read and validate an outing log CSV."""
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LogError(f"{path}: missing column(s) {missing}; expected {REQUIRED_COLUMNS}")
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["activity"] = df["activity"].astype(str).str.strip().str.lower()
    df["start_ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["start"].astype(str), errors="coerce")
    bad = df["start_ts"].isna()
    if bad.any():
        raise LogError(f"{path}: unparseable date/start on row(s) {list(df.index[bad] + 2)}")
    df["duration_h"] = pd.to_numeric(df["duration_h"], errors="raise").clip(lower=0.25)
    df["rating"] = pd.to_numeric(df["rating"], errors="raise")
    if not df["rating"].between(1, 5).all():
        raise LogError(f"{path}: ratings must be between 1 and 5")
    return df.sort_values("start_ts").reset_index(drop=True)


def window_hours(start_ts: pd.Timestamp, duration_h: float) -> pd.DatetimeIndex:
    """The hourly timestamps an outing overlaps (at least one)."""
    first = start_ts.floor("h")
    end = start_ts + pd.Timedelta(hours=float(duration_h))
    n = max(1, math.ceil((end - first) / pd.Timedelta(hours=1)))
    return pd.date_range(first, periods=n, freq="h")


def window_features(weather: pd.DataFrame, start_ts: pd.Timestamp, duration_h: float) -> dict | None:
    """Summarise the weather during one window. None if weather is missing."""
    hours = window_hours(start_ts, duration_h)
    w = weather.reindex(hours)
    if w["temperature_2m"].isna().any():
        return None
    return {
        "temp_c": float(w["temperature_2m"].mean()),
        "feels_c": float(w["apparent_temperature"].mean()),
        "humidity": float(w["relative_humidity_2m"].mean()),
        "precip_mm": float(w["precipitation"].sum()),
        "rain_chance": float(w["precipitation_probability"].max()) if "precipitation_probability" in w else float("nan"),
        "cloud": float(w["cloud_cover"].mean()),
        "wind_kmh": float(w["wind_speed_10m"].mean()),
        "gust_kmh": float(w["wind_gusts_10m"].max()),
        "sun_wm2": float(w["shortwave_radiation"].mean()),
        "daylight": float(w["is_day"].mean()),
        "start_hour": float(start_ts.hour + start_ts.minute / 60),
        "weekend": float(start_ts.dayofweek >= 5),
    }


def build_training_table(log: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per outing: weather features + activity + rating."""
    rows = []
    for r in log.itertuples():
        f = window_features(weather, r.start_ts, r.duration_h)
        if f is None:
            continue
        f.update(activity=r.activity, rating=float(r.rating), start_ts=r.start_ts, duration_h=r.duration_h)
        rows.append(f)
    if not rows:
        raise LogError("no outings overlap the available weather data")
    return pd.DataFrame(rows)


def encode(table: pd.DataFrame, activities: list[str]) -> np.ndarray:
    """Numeric matrix for the model; activity becomes a category code."""
    codes = {a: i for i, a in enumerate(activities)}
    X = table[WEATHER_FEATURES + TIME_FEATURES].to_numpy(dtype=float)
    act = table["activity"].map(codes).fillna(-1).to_numpy(dtype=float)[:, None]
    return np.hstack([X, act])


ACTIVITY_COL = len(WEATHER_FEATURES) + len(TIME_FEATURES)
