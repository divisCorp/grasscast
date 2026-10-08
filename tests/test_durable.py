"""Runs the real Temporal workflow against a local dev server (downloaded once, no account)."""

import asyncio
import os

import pytest

pytest.importorskip("temporalio")

from grasscast import durable  # noqa: E402


def test_workflow_retries_flaky_weather_and_writes_plan(fake_open_meteo, sample_log, tmp_path, monkeypatch):
    # Fail the first two weather calls, then succeed: Temporal should retry transparently.
    import grasscast.weather as weather

    real_get = weather.requests.get
    failures = {"n": 0}

    def flaky_get(*a, **kw):
        if failures["n"] < 2:
            failures["n"] += 1
            raise weather.requests.ConnectionError("simulated outage")
        return real_get(*a, **kw)

    monkeypatch.setattr(weather.requests, "get", flaky_get)
    req = durable.PlanRequest(log_path=str(sample_log), lat=40.76, lon=-111.89, timezone="America/Denver",
                              out_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"), days=3)
    res = asyncio.run(durable.start(req, address=None, workflow_id="test-plan"))
    assert failures["n"] == 2
    assert res.n_outings == 40
    assert os.path.exists(res.ics_path) and os.path.exists(res.text_path)
    assert "Built with PriorLabs-TabPFN" in res.text
