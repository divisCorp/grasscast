from datetime import date

import pandas as pd
import pytest

from grasscast.features import (FEATURES, LogError, build_training_table, encode, load_log, window_features,
                                window_hours)
from tests.conftest import fake_hourly


def _weather():
    w = pd.DataFrame(fake_hourly(date(2026, 7, 1), date(2026, 7, 3))["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time")


def test_window_hours_covers_partial_hours():
    hrs = window_hours(pd.Timestamp("2026-07-01 07:30"), 1.0)
    assert list(hrs.hour) == [7, 8]
    assert len(window_hours(pd.Timestamp("2026-07-01 07:00"), 0.25)) == 1


def test_window_features_values():
    w = _weather()
    f = window_features(w, pd.Timestamp("2026-07-01 08:00"), 2)
    sl = w.loc["2026-07-01 08:00":"2026-07-01 09:00"]
    assert f["temp_c"] == pytest.approx(sl["temperature_2m"].mean())
    assert f["gust_kmh"] == pytest.approx(sl["wind_gusts_10m"].max())
    assert f["rain_chance"] == pytest.approx(sl["precipitation_probability"].max())
    assert f["daylight"] == 1.0 and f["start_hour"] == 8 and f["weekend"] == 0.0
    assert window_features(w, pd.Timestamp("2026-08-01 08:00"), 2) is None  # no weather -> skipped


def test_load_log_validates(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("date,start,duration_h,activity\n2026-07-01,08:00,1,hike\n")
    with pytest.raises(LogError, match="missing column"):
        load_log(p)
    p.write_text("date,start,duration_h,activity,rating\n2026-07-01,08:00,1,hike,9\n")
    with pytest.raises(LogError, match="between 1 and 5"):
        load_log(p)
    p.write_text("# comment\ndate,start,duration_h,activity,rating\n2026-07-02,08:00,1, Hike ,4\n2026-07-01,07:00,2,run,3\n")
    df = load_log(p)
    assert list(df["activity"]) == ["run", "hike"]  # sorted by time, normalised names


def test_training_table_and_encoding(sample_log):
    w = pd.DataFrame(fake_hourly(date(2026, 6, 1), date(2026, 9, 30))["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    t = build_training_table(load_log(sample_log), w.set_index("time"))
    assert len(t) == 40
    X = encode(t, ["hike", "run"])
    assert X.shape == (40, len(FEATURES))
    assert set(X[:, -1]) <= {0.0, 1.0}
