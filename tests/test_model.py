"""Uses the real TabPFN v2 weights (downloaded once, ~100 MB, no account)."""

from datetime import date

import pandas as pd

from grasscast.features import build_training_table, load_log
from grasscast.model import TasteModel
from tests.conftest import fake_hourly


def _table(sample_log):
    w = pd.DataFrame(fake_hourly(date(2026, 6, 1), date(2026, 9, 30))["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return build_training_table(load_log(sample_log), w.set_index("time"))


def test_taste_model_learns_the_hidden_preference(sample_log):
    t = _table(sample_log)
    m = TasteModel().fit(t)
    base = t.iloc[[0]].copy()
    nice = base.assign(feels_c=14.0, temp_c=15.0, wind_kmh=5.0, gust_kmh=9.0, rain_chance=0.0, precip_mm=0.0)
    nasty = base.assign(feels_c=30.0, temp_c=31.0, wind_kmh=25.0, gust_kmh=45.0, rain_chance=95.0)
    p = m.predict(pd.concat([nice, nasty], ignore_index=True))
    assert p.loc[0, "score"] > p.loc[1, "score"] + 1.0
    assert p.loc[0, "p_great"] > p.loc[1, "p_great"]
    assert ((p["low"] <= p["score"] + 1e-6) & (p["score"] <= p["high"] + 1e-6)).all()
    assert p["p_great"].between(0, 1).all()


def test_refuses_tiny_logs(sample_log):
    import pytest

    with pytest.raises(ValueError, match="at least"):
        TasteModel().fit(_table(sample_log).head(5))
