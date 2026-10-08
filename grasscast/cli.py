"""Command line: `grasscast plan | log | eval | durable`."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from . import ATTRIBUTION, __version__


def _place_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--log", default=os.environ.get("GRASSCAST_LOG", "outings.csv"), help="outing log CSV")
    p.add_argument("--lat", type=float, default=_env_float("GRASSCAST_LAT"), help="latitude (or GRASSCAST_LAT)")
    p.add_argument("--lon", type=float, default=_env_float("GRASSCAST_LON"), help="longitude (or GRASSCAST_LON)")
    p.add_argument("--tz", default=os.environ.get("GRASSCAST_TZ", "auto"), help="IANA timezone, default: auto")
    p.add_argument("--offline", action="store_true", help="use cached weather only (no network)")
    p.add_argument("--metric", action="store_true", help="show °C and km/h instead of °F and mph")


def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v else None


def _place(args):
    from .weather import Place

    if args.lat is None or args.lon is None:
        sys.exit("grasscast: set --lat/--lon (or GRASSCAST_LAT/GRASSCAST_LON)")
    return Place(args.lat, args.lon, args.tz)


def cmd_plan(args) -> None:
    from .pipeline import make_plan

    plan = make_plan(args.log, _place(args), days=args.days, top=args.top, activity=args.activity,
                     imperial=not args.metric, offline=args.offline)
    print(plan.text)
    if args.ics:
        Path(args.ics).write_text(plan.ics)
        print(f"\nWrote {len(plan.best)} windows to {args.ics} (import it into any calendar app).")


def cmd_log(args) -> None:
    """Append one outing. Meant to take 5 seconds when you get home."""
    when = datetime.fromisoformat(args.start) if args.start else datetime.now().replace(second=0, microsecond=0)
    path = Path(args.log)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "start", "duration_h", "activity", "rating", "notes"])
        w.writerow([when.date().isoformat(), when.strftime("%H:%M"), args.hours, args.activity.lower(),
                    args.rating, args.notes or ""])
    print(f"logged: {args.activity} on {when:%a %b %d %H:%M}, {args.hours}h, {args.rating}/5 -> {path}")


def cmd_eval(args) -> None:
    import pandas as pd

    from .evaluate import cross_validate, learning_curve
    from .features import build_training_table, load_log
    from .pipeline import past_weather

    log = load_log(args.log)
    table = build_training_table(log, past_weather(_place(args), log, offline=args.offline))
    print(f"{len(table)} outings, activities: {', '.join(sorted(table['activity'].unique()))}\n")
    pd.set_option("display.width", 120)
    print("5-fold cross-validation, repeated twice (lower MAE is better):")
    print(cross_validate(table).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if not args.no_curve and len(table) >= 40:
        print("\nHeld-out MAE as the log grows (how many outings until it knows you?):")
        print(learning_curve(table).to_string())
    print(f"\n{ATTRIBUTION}.")


def cmd_durable(args) -> None:
    from . import durable

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("temporalio").setLevel(logging.ERROR)  # our own one-line step logs are enough
    if args.worker:
        import asyncio

        asyncio.run(durable.run_worker(args.address))
        return
    place = _place(args)
    req = durable.PlanRequest(log_path=str(Path(args.log).resolve()), lat=place.lat, lon=place.lon,
                              timezone=place.timezone, days=args.days, top=args.top,
                              imperial=not args.metric, out_dir=str(Path(args.out).resolve()),
                              score_timeout_s=args.score_timeout)
    from temporalio.client import WorkflowFailureError

    try:
        res = durable.run(req, address=None if args.local else args.address, workflow_id=args.workflow_id,
                          cron=args.cron, wait=not args.no_wait)
    except WorkflowFailureError as e:
        sys.exit(f"grasscast: workflow failed: {e.cause or e}")
    if res:
        print(res.text)
        print(f"\nWrote {res.text_path} and {res.ics_path}")
    elif args.cron:
        print(f"Scheduled workflow '{args.workflow_id}' with cron '{args.cron}'. Keep a worker running.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="grasscast", description=f"When should I go outside? {ATTRIBUTION}.")
    ap.add_argument("--version", action="version", version=f"grasscast {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="rank the best outdoor windows in the next few days")
    _place_args(p)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--top", type=int, default=7)
    p.add_argument("--activity", help="only plan this activity (e.g. hike, run)")
    p.add_argument("--ics", help="also write an .ics calendar file")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("log", help="append an outing to your log")
    p.add_argument("activity")
    p.add_argument("rating", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--start", help="ISO start time, e.g. 2026-10-07T07:30 (default: now)")
    p.add_argument("--notes")
    p.add_argument("--log", default=os.environ.get("GRASSCAST_LOG", "outings.csv"))
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("eval", help="check how well the model predicts your ratings")
    _place_args(p)
    p.add_argument("--no-curve", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("durable", help="run the plan as a durable Temporal workflow")
    _place_args(p)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--top", type=int, default=7)
    p.add_argument("--out", default=".", help="where to write grasscast-plan.txt/.ics")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="spin up a throwaway local Temporal dev server")
    g.add_argument("--worker", action="store_true", help="run a worker against --address")
    p.add_argument("--address", default="localhost:7233")
    p.add_argument("--workflow-id", default="grasscast-plan")
    p.add_argument("--cron", help="e.g. '0 6 * * *' to refresh the plan every morning at 6")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--score-timeout", type=int, default=120, help="seconds before a stuck scoring step is retried")
    p.set_defaults(func=cmd_durable)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
