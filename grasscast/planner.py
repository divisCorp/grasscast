"""Score every possible outing window in the forecast and pick the best ones."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .features import window_features
from . import ATTRIBUTION

EARLIEST_START = 5   # 05:00
LATEST_START = 20    # 20:00
MIN_DAYLIGHT = 0.99  # the whole window must be in daylight


def activity_durations(table: pd.DataFrame) -> dict[str, float]:
    """Typical (median) outing length per activity, rounded to the half hour."""
    med = table.groupby("activity")["duration_h"].median()
    return {a: max(0.5, round(d * 2) / 2) for a, d in med.items()}


def candidate_windows(forecast: pd.DataFrame, durations: dict[str, float],
                      now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Every (activity, start hour) in the forecast that's fully in daylight."""
    now = now if now is not None else pd.Timestamp.now().floor("h")
    rows = []
    for ts in forecast.index:
        if ts < now or not (EARLIEST_START <= ts.hour <= LATEST_START):
            continue
        for act, dur in durations.items():
            f = window_features(forecast, ts, dur)
            if f is None or f["daylight"] < MIN_DAYLIGHT:
                continue
            f.update(activity=act, start_ts=ts, duration_h=dur)
            rows.append(f)
    return pd.DataFrame(rows)


def sweet_spot(table: pd.DataFrame, min_rating: float = 4.0) -> dict[str, tuple[float, float]]:
    """10th-90th percentile of conditions on your best outings."""
    good = table[table["rating"] >= min_rating]
    if len(good) < 3:
        good = table.nlargest(max(3, len(table) // 3), "rating")
    cols = ["feels_c", "wind_kmh", "gust_kmh", "precip_mm", "rain_chance", "cloud", "sun_wm2"]
    return {c: (float(good[c].quantile(0.1)), float(good[c].quantile(0.9))) for c in cols
            if good[c].notna().any()}


_REASONS = {
    "feels_c": ("colder than you like", "warmer than you like"),
    "wind_kmh": ("", "windier than you like"),
    "gust_kmh": ("", "gustier than you like"),
    "precip_mm": ("", "wetter than you like"),
    "rain_chance": ("", "higher rain risk than you like"),
    "cloud": ("clearer than usual", "cloudier than you like"),
    "sun_wm2": ("darker/greyer than you like", "more sun than you like"),
}


def explain(row: pd.Series, spot: dict[str, tuple[float, float]]) -> str:
    notes = []
    for col, (lo, hi) in spot.items():
        below, above = _REASONS[col]
        span = max(hi - lo, 1e-6)
        if pd.isna(row[col]):
            continue
        if row[col] < lo - 0.1 * span and below:
            notes.append(below)
        elif row[col] > hi + 0.1 * span and above:
            notes.append(above)
    return "; ".join(notes) if notes else "right in your sweet spot"


_NOVELTY_LABELS = {
    "feels_c": ("this cold", "this hot"),
    "wind_kmh": ("", "this windy"),
    "gust_kmh": ("", "gusts this strong"),
    "precip_mm": ("", "this much rain"),
    "rain_chance": ("", "rain risk this high"),
}


def novelty(row: pd.Series, table: pd.DataFrame, min_support: int = 2) -> list[str]:
    """Conditions you've (almost) never logged an outing in. The model is guessing there.

    A value counts as new territory when fewer than `min_support` logged outings
    were at least that extreme.
    """
    out = []
    for col, (low_lbl, high_lbl) in _NOVELTY_LABELS.items():
        v = row.get(col)
        seen = table[col].dropna()
        if pd.isna(v) or seen.empty:
            continue
        if high_lbl and (seen >= v).sum() < min_support and v > seen.median():
            out.append(high_lbl)
        elif low_lbl and (seen <= v).sum() < min_support and v < seen.median():
            out.append(low_lbl)
    return out


def flag_novelty(scored: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    s = scored.copy()
    s["untested"] = [", ".join(novelty(r, table)) for _, r in s.iterrows()]
    return s


def verdict(p_great: float) -> str:
    """Label a window by the chance you'd rate it 4-5 stars."""
    return "GO" if p_great >= 0.6 else "decent" if p_great >= 0.35 else "meh" if p_great >= 0.15 else "skip"


def pick_best(scored: pd.DataFrame, top: int = 7) -> pd.DataFrame:
    """The single best window on each day (any activity), in date order.

    Ranked by P(4-5 stars), ties broken by expected rating. The best day overall
    is flagged with `top_pick`.
    """
    s = scored.copy()
    s["day"] = s["start_ts"].dt.date
    s["rank_key"] = s["p_great"] + 0.01 * s["score"]
    if "untested" in s:
        # Don't put a guess at the top of your week: demote windows outside your history.
        s["rank_key"] -= 0.25 * s["untested"].str.len().gt(0)
    best = s.sort_values("rank_key", ascending=False).groupby("day", sort=False).head(1)
    best = best.sort_values("start_ts").head(top).reset_index(drop=True)
    best["top_pick"] = best["rank_key"] == best["rank_key"].max()
    return best


def _fmt_temp(c: float, imperial: bool) -> str:
    return f"{c * 9 / 5 + 32:.0f}°F" if imperial else f"{c:.0f}°C"


def _fmt_wind(kmh: float, imperial: bool) -> str:
    return f"{kmh / 1.609:.0f} mph" if imperial else f"{kmh:.0f} km/h"


def render_text(best: pd.DataFrame, spot, *, imperial: bool = True, n_outings: int = 0) -> str:
    lines = [f"grasscast: your best windows to get outside (learned from {n_outings} logged outings)", ""]
    if best.empty:
        return "\n".join(lines + ["No daylight windows found in the forecast."])
    lines.append("Best window each day. 'chance' = how likely you'd rate it 4-5 stars; '?' = the model is extrapolating.")
    lines.append("")
    for r in best.itertuples():
        end = r.start_ts + pd.Timedelta(hours=r.duration_h)
        rain = "" if np.isnan(r.rain_chance) else f", {r.rain_chance:.0f}% rain"
        star = "*" if r.top_pick else " "
        label = verdict(r.p_great) + ("?" if getattr(r, "untested", "") else "")
        lines.append(
            f"{star}{label:>7}  {r.start_ts:%a %b %d}  {r.start_ts:%H:%M}-{end:%H:%M}  {r.activity:<5} "
            f"chance {r.p_great:>4.0%}  likely {r.low:.0f}-{r.high:.0f}★   "
            f"feels {_fmt_temp(r.feels_c, imperial)}, wind {_fmt_wind(r.wind_kmh, imperial)}, "
            f"clouds {r.cloud:.0f}%{rain}"
        )
        lines.append(f"          why: {explain(pd.Series(r._asdict()), spot)}")
        if getattr(r, "untested", ""):
            lines.append(f"          ? untested: you've never logged an outing in {r.untested}; treat as a guess")
    lo, hi = spot["feels_c"]
    lines += ["", f"Your sweet spot (from 4-5 star outings): feels {_fmt_temp(lo, imperial)}-{_fmt_temp(hi, imperial)}, "
              f"wind under {_fmt_wind(spot['wind_kmh'][1], imperial)}.",
              f"* = top pick of the week. {ATTRIBUTION}."]
    return "\n".join(lines)


def _ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> str:
    """RFC 5545 line folding: max 75 octets per line, continuation lines start with a space."""
    out, cur = [], b""
    for ch in line:
        b = ch.encode()
        if len(cur) + len(b) > (75 if not out else 74):
            out.append(cur.decode())
            cur = b""
        cur += b
    out.append(cur.decode())
    return "\r\n ".join(out)


def render_ics(best: pd.DataFrame, spot, *, min_p: float = 0.35, imperial: bool = True) -> str:
    """An .ics calendar file so the plan lives in your calendar, not on your screen."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//grasscast//EN", "CALSCALE:GREGORIAN"]
    for i, r in enumerate(best.itertuples()):
        if r.p_great < min_p:  # only put decent-or-better windows in your calendar
            continue
        end = r.start_ts + pd.Timedelta(hours=r.duration_h)
        desc = (f"{r.p_great:.0%} chance you'd rate this 4-5 stars (likely {r.low:.0f}-{r.high:.0f} stars). "
                f"Feels {_fmt_temp(r.feels_c, imperial)}, wind {_fmt_wind(r.wind_kmh, imperial)}, "
                f"clouds {r.cloud:.0f}%. Why: {explain(pd.Series(r._asdict()), spot)}. {ATTRIBUTION}.")
        out += [
            "BEGIN:VEVENT",
            f"UID:grasscast-{r.start_ts:%Y%m%dT%H%M}-{r.activity}-{i}@grasscast",
            f"DTSTAMP:{stamp}",
            f"DTSTART:{r.start_ts:%Y%m%dT%H%M%S}",  # floating local time
            f"DTEND:{end:%Y%m%dT%H%M%S}",
            f"SUMMARY:{_ics_escape(f'{verdict(r.p_great).upper()}{"?" if getattr(r, "untested", "") else ""}: {r.activity} ({r.p_great:.0%} chance of 4-5 stars)')}",
            f"DESCRIPTION:{_ics_escape(desc)}",
            "TRANSP:TRANSPARENT",
            "END:VEVENT",
        ]
    out.append("END:VCALENDAR")
    return "\r\n".join(_fold(l) for l in out) + "\r\n"
