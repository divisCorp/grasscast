#!/usr/bin/env bash
# Durability demo: kill the worker in the middle of the TabPFN step, start a new
# one, and watch the workflow finish WITHOUT re-fetching any weather.
#
# Needs the Temporal CLI (`temporal`), e.g. `brew install temporal` or
# https://docs.temporal.io/cli. Everything runs on localhost.
set -euo pipefail
cd "$(dirname "$0")/.."
TEMPORAL=${TEMPORAL:-temporal}
export GRASSCAST_LAT=${GRASSCAST_LAT:-40.7608} GRASSCAST_LON=${GRASSCAST_LON:--111.8910} GRASSCAST_TZ=${GRASSCAST_TZ:-America/Denver}
export GRASSCAST_CACHE=$(mktemp -d)   # empty cache, so the weather steps really hit the network
LOG=examples/outings.sample.csv
OUT=$(mktemp -d)

"$TEMPORAL" server start-dev --headless --port 7233 --log-level error >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 4

echo "== worker #1 starts"
python -m grasscast durable --worker > "$OUT/worker1.log" 2>&1 &
W1=$!
python -m grasscast durable --log "$LOG" --out "$OUT" --no-wait --score-timeout 15 --workflow-id crash-demo

# wait until the TabPFN step has started, then pull the plug
until grep -q "score_windows: start" "$OUT/worker1.log"; do sleep 0.2; done
kill -9 $W1
echo "== worker #1 killed with SIGKILL during score_windows"
sed 's/^/   [worker1] /' "$OUT/worker1.log" | grep -v "HF Hub"

echo "== worker #2 starts (the workflow resumes on its own)"
python -m grasscast durable --worker > "$OUT/worker2.log" 2>&1 &
W2=$!
"$TEMPORAL" workflow result --workflow-id crash-demo --address localhost:7233 >/dev/null 2>&1 || true
until grep -q "write_plan: done" "$OUT/worker2.log"; do sleep 0.5; done
kill $W2 2>/dev/null || true
sed 's/^/   [worker2] /' "$OUT/worker2.log" | grep -v "HF Hub"

echo "== activity history (from Temporal)"
"$TEMPORAL" workflow show --workflow-id crash-demo --address localhost:7233 2>/dev/null \
  | grep -E "ActivityTaskScheduled|ActivityTaskStarted|ActivityTaskCompleted|ActivityTaskTimedOut|WorkflowExecutionCompleted" \
  | sed 's/^/   /' || true
echo
cat "$OUT/grasscast-plan.txt"
