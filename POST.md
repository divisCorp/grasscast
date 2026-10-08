---
title: 'grasscast: a "should I go outside?" forecast that learns your taste from a few dozen outings (TabPFN + Temporal)'
published: false
tags: devchallenge, hf26challenge, ai, python
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

## What I Built

Weather apps tell you it'll be 64°F with 10 mph wind on Saturday. They can't tell you whether *you* will enjoy a hike in that, or whether Thursday morning would be the better bet.

**grasscast** is a small command-line tool that answers *"when should I go outside this week?"* for one person. You keep a tiny log of your outings, one line each: date, start time, hours, activity, and how much you enjoyed it on a 1–5 scale. grasscast joins each outing with the weather you had, learns your taste with **TabPFN** (an open-weight tabular foundation model), scores every daylight window in the next 7 days, and gives back:

- the **best window for each day**, with the chance you'd rate it 4–5 stars,
- a one-line **"why"** that compares it with the conditions of your best outings,
- an **`.ics` file** so the plan goes into your calendar and you can close the laptop.

The screen time is the point. Logging an outing is one command when you get home (`grasscast log hike 5 --hours 2.5`), and the plan is something you glance at once a week, or once a morning if you let it run on a schedule. Everything else happens outside.

It's for anyone who has a vague sense of "I like hiking when it's cool and calm" and would rather have a tool learn the details than check four weather apps.

## Demo

![grasscast demo: logging an outing, planning the week, and a durable run surviving a flaky network](https://github.com/divisCorp/grasscast/raw/main/demo/demo.gif)

*The demo above is a real terminal recording (asciinema → GIF) of the commands below, run against the bundled demo log.*

> **Honesty note on the demo data:** I don't have a real outing log yet, so the bundled `examples/outings.sample.csv` is **synthetic**. It uses real Salt Lake City weather from April to October 2026 and a made-up person, whose ratings come from a hidden formula plus noise. That turned out to be useful: because the true taste is known, I can check whether the model actually learns it (see *How I Built It*).

Here's the weekly plan it produced on Oct 7 (lightly trimmed):

```
$ grasscast plan --ics plan.ics
grasscast: your best windows to get outside (learned from 63 logged outings)

Best window each day. 'chance' = how likely you'd rate it 4-5 stars; '?' = the model is extrapolating.

      GO  Wed Oct 07  17:00-18:30  bike  chance  67%  likely 2-5★   feels 75°F, wind 7 mph, clouds 0%, 1% rain
          why: warmer than you like
*     GO  Thu Oct 08  10:00-11:30  bike  chance  95%  likely 4-5★   feels 67°F, wind 3 mph, clouds 1%, 1% rain
          why: right in your sweet spot
      GO  Fri Oct 09  09:00-10:30  bike  chance  91%  likely 4-5★   feels 62°F, wind 7 mph, clouds 38%, 2% rain
          why: higher rain risk than you like
     GO?  Sun Oct 11  12:00-13:30  bike  chance  91%  likely 4-5★   feels 68°F, wind 13 mph, clouds 8%, 81% rain
          why: windier than you like; gustier than you like; higher rain risk than you like
          ? untested: you've never logged an outing in gusts this strong, rain risk this high; treat as a guess
     ...
Your sweet spot (from 4-5 star outings): feels 55°F-73°F, wind under 9 mph.
* = top pick of the week. Built with PriorLabs-TabPFN.

Wrote 7 windows to plan.ics (import it into any calendar app).
```

Look at Sunday. The model gives a bike ride in an 81%-rain forecast a 91% chance of being great, which is obviously wrong. It's wrong for an honest reason: the synthetic person almost never went out in rain (Salt Lake City summers are dry; only **one** of 63 outings had a rain chance above 40%), so the model has no evidence either way. Instead of pretending, grasscast detects that the window sits outside anything in your log, marks it `GO?`, and ranks it below the windows it actually has evidence for. Thursday is the top pick, not Sunday. Log a couple of drizzly walks and the model learns the rest.

## Code

{% github https://github.com/divisCorp/grasscast %}

Python, MIT licensed, about 1,000 lines plus 17 tests. Quick start:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e ".[dev]"
export GRASSCAST_LAT=40.7608 GRASSCAST_LON=-111.8910 GRASSCAST_TZ=America/Denver
grasscast plan --log examples/outings.sample.csv --ics plan.ics
```

## How I Built It

**The open pieces:** [TabPFN](https://github.com/PriorLabs/TabPFN) v2 (open weights, local CPU inference) for the model, [Temporal](https://github.com/temporalio/temporal) (open-source server plus Python SDK) for durable execution, and [Open-Meteo](https://open-meteo.com) open weather data (free, no API key).

### 1. Why TabPFN is the right model for a personal log

A personal outing log is tiny: 20, 40, maybe 100 rows. That's the regime where classic ML struggles and where TabPFN is built to shine. TabPFN is a transformer pretrained on millions of synthetic tabular problems. At `fit()` time it doesn't run gradient descent. It reads your whole table *as context* and predicts in a single forward pass. So there's no training loop, no hyperparameter search, and nothing to overfit with a grid search on 40 rows.

The part I like most is that TabPFN's regressor returns a **full predictive distribution**, not just a number. grasscast uses three things from it:

```python
out = reg.predict(X, output_type="full")
score   = out["mean"]                                    # expected rating
low, hi = out["quantiles"][0], out["quantiles"][-1]      # 80% band
p_great = 1 - out["criterion"].cdf(out["logits"], 3.5)   # P(you rate it 4-5 stars)  (simplified)
```

Windows are ranked by `p_great`. "95% chance this is a great outing" is something a person can act on in a way that "predicted 4.3" isn't.

Setup is one line. Fitting on 63 outings and scoring all ~200 candidate windows of the week takes about 5 seconds on a plain CPU (no GPU):

```python
TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu",
                                           categorical_features_indices=[ACTIVITY_COL])
```

(I pinned the v2 weights because they download without an account, so anyone can clone and run this.)

### 2. Features: train on forecasts, not hindsight

For each outing, grasscast summarizes the hourly weather during it: feels-like temperature, humidity, rain amount, max rain chance, cloud, sunshine, mean wind, max gust, plus start hour, weekend, and activity (as a categorical feature, so one model learns that you like biking warmer than running).

One detail matters a lot. Past weather comes from Open-Meteo's **historical *forecast*** archive: what the forecast said at the time, not the reanalysis of what actually happened. The model is trained on the same kind of number it sees at prediction time, so it learns "I enjoy days *forecast* like this", which is the question you're actually asking.

### 3. Does it actually learn? (yes, and faster than the baselines)

Since the synthetic person's real taste is known, I could measure it. `grasscast eval` runs repeated 5-fold cross-validation against simple baselines, plus a learning curve on held-out outings:

```
5-fold cross-validation, repeated twice (lower MAE is better):
                    model  MAE (stars)  rank corr
       TabPFN (grasscast)         0.59       0.87
            random forest         0.73       0.82
         ridge regression         0.89       0.76
     average per activity         1.13       0.30
always guess your average         1.31      -0.20

Held-out MAE as the log grows (trimmed; full table in demo/eval.txt):
outings  TabPFN  avg/activity  random forest
10         1.06          1.27           1.15
20         0.95          1.29           1.07
40         0.66          1.16           0.78
```

TabPFN is the most accurate at every log size, and its lead over the random forest is largest when the log is smallest (10–20 outings), which is exactly where a personal tool spends its first month. The `eval` command works on your own log too, so you can see how well it knows *you*.

### 4. Making it durable with Temporal

The useful version of this tool runs by itself at 6 AM. Over a week of mornings, a free weather API will time out at some point and a laptop will go to sleep mid-run. So the same pipeline also ships as a **Temporal workflow** with four activities:

```
build_taste_table  ->  get_forecast  ->  score_windows (TabPFN)  ->  write_plan (.txt + .ics)
```

- Weather calls retry with exponential backoff (capped at 2 minutes) for up to 30 minutes.
- Each completed step is recorded in the workflow history, so a crashed worker resumes at the next step without re-fetching anything.
- `--cron "0 6 * * *"` refreshes the calendar file every morning.
- `grasscast durable --local` spins up Temporal's open-source dev server in-process. No account, no cloud.

To show it working, there's a chaos switch that makes a percentage of weather calls fail:

```
$ GRASSCAST_CHAOS=0.75 grasscast durable --local      # log prefixes trimmed
15:19:19 build_taste_table: start (attempt 1)
15:19:19 build_taste_table: attempt 1 failed: chaos monkey: simulated flaky network -> Temporal will retry
15:19:20 build_taste_table: start (attempt 2)
   ... attempts 2-4 also fail; Temporal backs off 1s, 2s, 4s, 8s between tries ...
15:19:34 build_taste_table: start (attempt 5)
15:19:34 build_taste_table: done
15:19:35 get_forecast: attempt 1 failed ... attempt 2 failed ...
15:19:39 get_forecast: done
15:19:44 score_windows: done
15:19:44 write_plan: done
```

And a nastier test (`demo/crash_resume.sh`): start a worker, **`kill -9` it in the middle of the TabPFN step**, then start a fresh worker:

```
[worker1] build_taste_table: done
[worker1] get_forecast: done
[worker1] score_windows: start (attempt 1)
== worker #1 killed with SIGKILL during score_windows
[worker2] score_windows: start (attempt 2)    <- picks up exactly here; no weather re-fetch
[worker2] score_windows: done
[worker2] write_plan: done
```

The new worker never re-runs the two weather steps. Their results come from Temporal's history.

### 5. Tests

17 pytest tests cover feature extraction, the planner (daylight-only windows, one pick per day, untested windows demoted, valid RFC 5545 calendar output), the weather cache and offline mode, the real TabPFN model learning a hidden preference, an end-to-end plan, and the Temporal workflow surviving injected network failures on a real local server. The tests fake the weather API, so they run without a network.

## Why Does Open Innovation Matter?

**Your outing log is sensitive data.** It's a record of when you leave home, for how long, and where you like to go. That's exactly the data I don't want to upload to a third-party API so it can tell me "go Thursday." With open weights, it never leaves my machine. The only thing sent anywhere is a latitude and longitude to a free weather service. I verified the whole plan also runs inside a network namespace with **no network at all** once the forecast is cached (`demo/offline-run.txt`). It works at a trailhead with one bar of signal, or none.

**It costs nothing to run, forever.** No API key, no per-call billing, no account. TabPFN v2 runs on a plain CPU in seconds, the Temporal dev server is a local binary, and Open-Meteo is free. A tool you're supposed to forget about for a week can't come with a monthly bill or an API key that expires.

**Open models give you the whole distribution, not a sentence.** Because I run TabPFN myself, I get its full predictive distribution and can compute "probability of a 4–5 star outing" exactly. A closed chat API would give me a confident-sounding paragraph. I'd have no calibrated probability and no way to tell when it's extrapolating. The `untested` flag depends on being able to inspect both the data and the model's uncertainty.

**I can swap and inspect every layer.** The model version is one line (v2 today, a newer TabPFN tomorrow), the weather source is a function, and the durability layer is a self-hosted open-source server rather than someone's SaaS. If a piece disappears, I replace that piece without rewriting the tool.

## My Agent Session

Full disclosure: grasscast was built with an AI coding agent, which the challenge rules allow. I didn't export the session to DevRelay. In place of the session, the repo includes everything needed to check the work: `demo/record_demo.sh` regenerates the demo GIF from real runs, `demo/crash_resume.sh` reproduces the kill-the-worker test, and the captured outputs (`demo/eval.txt`, `demo/durable-chaos.txt`, `demo/crash-resume.txt`, `demo/offline-run.txt`) are unedited.

## Prize Categories

- **Best Use of TabPFN**: TabPFN is the model at the center of grasscast. In-context regression on a tiny personal log, using the full predictive distribution for P(4–5★) ranking and uncertainty bands, benchmarked against baselines with a learning curve.
- **Best Use of Temporal**: the plan pipeline runs as a durable Temporal workflow with retried weather activities, crash-resume (demonstrated with `kill -9`), and a cron schedule, all on the local open-source dev server.

*Built with PriorLabs-TabPFN · Weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)*
