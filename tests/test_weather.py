from datetime import date

import pytest

from grasscast.weather import HISTORY_URL, Place, WeatherError, fetch_forecast, fetch_history

P = Place(40.76, -111.89, "America/Denver")


def test_history_is_cached_forever(fake_open_meteo, tmp_path):
    a = fetch_history(P, date(2026, 7, 1), date(2026, 7, 2), cache_dir=tmp_path)
    b = fetch_history(P, date(2026, 7, 1), date(2026, 7, 2), cache_dir=tmp_path)
    assert len(fake_open_meteo) == 1 and fake_open_meteo[0][0] == HISTORY_URL
    assert len(a) == 48 and a.equals(b)


def test_offline_uses_cache_and_fails_cleanly(fake_open_meteo, tmp_path):
    with pytest.raises(WeatherError, match="offline"):
        fetch_forecast(P, cache_dir=tmp_path, offline=True)
    fetch_forecast(P, cache_dir=tmp_path)
    fc = fetch_forecast(P, cache_dir=tmp_path, offline=True)
    assert len(fc) == 7 * 24 and len(fake_open_meteo) == 1


def test_chaos_mode_raises(fake_open_meteo, tmp_path, monkeypatch):
    monkeypatch.setenv("GRASSCAST_CHAOS", "1")
    with pytest.raises(WeatherError, match="chaos"):
        fetch_forecast(P, cache_dir=tmp_path)
