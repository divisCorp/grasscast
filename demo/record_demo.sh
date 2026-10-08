#!/usr/bin/env bash
# Records demo/demo.gif from REAL command runs (asciinema + agg).
# Usage: asciinema rec -c demo/record_demo.sh demo/demo.cast && agg demo/demo.cast demo/demo.gif
cd "$(dirname "$0")/.."
export GRASSCAST_LAT=40.7608 GRASSCAST_LON=-111.8910 GRASSCAST_TZ=America/Denver
export GRASSCAST_LOG=examples/outings.sample.csv
type_cmd() {
  printf '\033[1;32m$\033[0m '
  local s="$1"; for ((i=0; i<${#s}; i++)); do printf '%s' "${s:$i:1}"; sleep 0.03; done; echo; sleep 0.4
}
say() { printf '\033[1;36m# %s\033[0m\n' "$1"; sleep 1.2; }
run() { type_cmd "$1"; bash -c "$1" 2>&1 | grep -v -e 'HF Hub' -e 'Cluster Id' -e '^Temporal ' -e '^$'; sleep 2.5; }

clear
say "My outing log: date, start, hours, activity, how much I enjoyed it (1-5)"
run "head -6 examples/outings.sample.csv"
say "Got back from a walk? Logging it takes one line:"
run "grasscast log walk 4 --hours 1 --start 2026-10-06T17:30 --log /tmp/demo-log.csv"
say "When should I go out this week? TabPFN learns my taste from 63 outings, locally:"
run "grasscast plan --ics plan.ics"
say "Nightly run as a durable Temporal workflow, with a simulated flaky network:"
run "GRASSCAST_CACHE=\$(mktemp -d) GRASSCAST_CHAOS=0.6 grasscast durable --local --out /tmp/gc 2>&1 | grep -E 'attempt|done|started' | cut -c1-118"
sleep 2
