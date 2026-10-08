"""Optional: run the plan pipeline as a durable Temporal workflow.

Why bother for a personal tool? Because the useful version of grasscast runs
unattended (say every morning at 6) and talks to a free weather API that
sometimes times out. Temporal gives us, with no extra code:

* automatic retries with backoff for the flaky network calls,
* resume-where-it-left-off if the machine sleeps or the worker crashes
  (finished steps are not re-run; their results live in the workflow history),
* a cron schedule, so the plan just shows up.

Everything runs locally: `--local` starts Temporal's open-source dev server
in-process (no account, no cloud).
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import pandas as pd

    from .features import build_training_table, load_log
    from .pipeline import past_weather, score_forecast
    from .planner import pick_best, render_ics, render_text, sweet_spot
    from .weather import DEFAULT_CACHE, Place, fetch_forecast

TASK_QUEUE = "grasscast"
log = logging.getLogger("grasscast.durable")


@dataclass
class PlanRequest:
    log_path: str
    lat: float
    lon: float
    timezone: str = "auto"
    days: int = 7
    top: int = 7
    imperial: bool = True
    out_dir: str = "."
    cache_dir: str = str(DEFAULT_CACHE)
    score_timeout_s: int = 120  # TabPFN on CPU takes seconds; if a worker dies, retry after this


@dataclass
class PlanResult:
    text: str
    ics_path: str
    text_path: str
    n_outings: int
    n_windows: int


def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso")


def _df_from_json(s: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(s), orient="split")
    for col in ("start_ts",):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _step(fn):
    """Log each attempt of an activity, and why it failed, in one readable line."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        attempt = activity.info().attempt
        log.info("%s: start (attempt %d)", fn.__name__, attempt)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            log.warning("%s: attempt %d failed: %s -> Temporal will retry", fn.__name__, attempt, e)
            raise
        log.info("%s: done", fn.__name__)
        return result

    return wrapper


# ---- activities: each one is a retryable, independently-recorded step ----

@activity.defn
@_step
def build_taste_table(req: PlanRequest) -> str:
    """Read the outing log and join it with the weather during each outing."""
    place = Place(req.lat, req.lon, req.timezone)
    outings = load_log(req.log_path)
    weather = past_weather(place, outings, cache_dir=Path(req.cache_dir))
    return _df_to_json(build_training_table(outings, weather))


@activity.defn
@_step
def get_forecast(req: PlanRequest) -> str:
    fc = fetch_forecast(Place(req.lat, req.lon, req.timezone), req.days, cache_dir=Path(req.cache_dir))
    fc = fc.reset_index()
    return _df_to_json(fc)


@activity.defn
@_step
def score_windows(req: PlanRequest, table_json: str, forecast_json: str) -> str:
    """Run TabPFN over every candidate window. CPU-only, a few seconds."""
    table = _df_from_json(table_json)
    fc = _df_from_json(forecast_json)
    fc["time"] = pd.to_datetime(fc["time"])
    fc = fc.set_index("time")
    scored = score_forecast(table, fc)
    best = pick_best(scored, top=req.top) if not scored.empty else scored
    return _df_to_json(best)


@activity.defn
@_step
def write_plan(req: PlanRequest, table_json: str, best_json: str) -> PlanResult:
    table = _df_from_json(table_json)
    best = _df_from_json(best_json)
    spot = sweet_spot(table)
    text = render_text(best, spot, imperial=req.imperial, n_outings=len(table))
    out = Path(req.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grasscast-plan.txt").write_text(text + "\n")
    (out / "grasscast-plan.ics").write_text(render_ics(best, spot, imperial=req.imperial))
    return PlanResult(text=text, ics_path=str(out / "grasscast-plan.ics"),
                      text_path=str(out / "grasscast-plan.txt"), n_outings=len(table), n_windows=len(best))


# ---- the workflow: deterministic glue, survives crashes ----

# Weather calls: keep retrying with backoff (capped at 2 min between tries) for up to
# 30 minutes. For a 6 AM unattended run, "eventually" is fine; giving up is not.
NETWORK_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), backoff_coefficient=2.0,
                            maximum_interval=timedelta(minutes=2), maximum_attempts=0)
NETWORK_BUDGET = timedelta(minutes=30)
LOCAL_RETRY = RetryPolicy(maximum_attempts=3)


@workflow.defn
class GrassPlanWorkflow:
    @workflow.run
    async def run(self, req: PlanRequest) -> PlanResult:
        table = await workflow.execute_activity(
            build_taste_table, req, start_to_close_timeout=timedelta(minutes=2),
            schedule_to_close_timeout=NETWORK_BUDGET, retry_policy=NETWORK_RETRY)
        forecast = await workflow.execute_activity(
            get_forecast, req, start_to_close_timeout=timedelta(minutes=1),
            schedule_to_close_timeout=NETWORK_BUDGET, retry_policy=NETWORK_RETRY)
        best = await workflow.execute_activity(
            score_windows, args=[req, table, forecast], start_to_close_timeout=timedelta(seconds=req.score_timeout_s),
            retry_policy=LOCAL_RETRY)
        return await workflow.execute_activity(
            write_plan, args=[req, table, best], start_to_close_timeout=timedelta(minutes=1),
            retry_policy=LOCAL_RETRY)


ACTIVITIES = [build_taste_table, get_forecast, score_windows, write_plan]


def make_worker(client):
    from temporalio.worker import Worker

    return Worker(client, task_queue=TASK_QUEUE, workflows=[GrassPlanWorkflow], activities=ACTIVITIES,
                  activity_executor=ThreadPoolExecutor(4))


async def run_worker(address: str) -> None:
    from temporalio.client import Client

    client = await Client.connect(address)
    log.info("worker connected to %s, polling task queue '%s' (Ctrl-C to stop)", address, TASK_QUEUE)
    await make_worker(client).run()


async def start(req: PlanRequest, *, address: str | None, workflow_id: str, cron: str | None = None,
                wait: bool = True) -> PlanResult | None:
    """Start (or attach to) a plan workflow. With address=None, run a local dev server in-process."""
    from temporalio.client import Client
    from temporalio.common import WorkflowIDReusePolicy

    env = None
    if address is None:
        from temporalio.testing import WorkflowEnvironment

        log.info("starting a local Temporal dev server (open source, no account needed)...")
        env = await WorkflowEnvironment.start_local()
        client = env.client
    else:
        client = await Client.connect(address)
    try:
        kwargs = dict(id=workflow_id, task_queue=TASK_QUEUE,
                      id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE)
        if cron:
            kwargs["cron_schedule"] = cron
        if env is not None:
            async with make_worker(client):
                handle = await client.start_workflow(GrassPlanWorkflow.run, req, **kwargs)
                log.info("started workflow %s", handle.id)
                return await handle.result()
        handle = await client.start_workflow(GrassPlanWorkflow.run, req, **kwargs)
        log.info("started workflow %s on %s", handle.id, address)
        return await handle.result() if wait and not cron else None
    finally:
        if env is not None:
            await env.shutdown()


def run(req: PlanRequest, **kw) -> PlanResult | None:
    return asyncio.run(start(req, **kw))
