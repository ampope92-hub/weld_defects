#!/bin/bash

# Verifier entrypoint. This runs in the verifier container, not the agent's — it sees
# only the paths declared in `artifacts`. All tooling is baked into tests/Dockerfile;
# nothing is installed or fetched here.

set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# This shell's own process group. Each run below is started under `setsid` so it gets a group
# of its own that can be torn down as a unit, but /proc can be read in the window before
# setsid(2) has taken effect, and in that window it still reports THIS group. Killing that
# group kills the verifier itself: the log stops mid-script, pytest never runs, no ctrf.json
# is written and the reward stays 0. Three rounds were lost to exactly that before the cause
# was found, so no group is ever signalled unless it is demonstrably not our own.

# Fold stderr into stdout. A previous round ended with both crate runs reported as successful
# and then no pytest output at all, which is unreadable when only stdout is captured: whatever
# ended the script said so on a stream nobody kept.
exec 2>&1

# Root's PATH, set here rather than inherited, holds nothing the build user can write to, and
# the interpreter that runs the tests is called by its full path below besides, so no file
# agent code drops anywhere can stand in for a command this script runs as root. The build
# user's cargo calls are given the toolchain directory explicitly.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# The graded predictions must come out of the agent's own crate, so discard anything on
# disk and rebuild from the collected sources. The dependency tree is already compiled
# into this image, so only the agent's own sources recompile. A build or run failure is
# not fatal here — the tests report the missing file.
#
# That rebuild executes code the agent wrote, so it runs as the unprivileged `runner`. As
# root it could read the answer key and echo it back, or rewrite test_outputs.py before
# pytest reads it below. Artifacts land owned by root, so hand the crate over first and pin
# the shipped export read-only. /logs is left alone — Harbor owns it and the reward is
# written there last, after every agent process is dead. /tests is closed off below; the
# graded labels are read from the root-only copy baked into this image (see tests/Dockerfile).
chown -R runner:runner /app/predictor
chmod -R a-w /app/data

# Everything below that touches agent-authored code runs through this helper: the build user,
# with supplementary groups cleared and no-new-privileges set, so an inherited group or a
# setuid binary cannot widen what that code can reach.
as_runner() {
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner "$@"
}

# The shipped tests/ directory also lands here at runtime, so the verifier's own code and a
# plaintext copy of the answer key sit under /tests just as agent-authored code is about to
# run. Close the whole directory to the build user before anything untrusted starts: root
# still reads it, `runner` cannot even traverse it. If it is still reachable the run is
# abandoned with reward 0 rather than graded — a reachable /tests is one the agent can read
# the answers out of, or whose test file it can rewrite before pytest reads it. The graded
# labels are read from the root-only copy at /opt/golden either way.
chmod 700 /tests 2>/dev/null || true
if as_runner bash -c 'test -r /tests/test_outputs.py || test -r /tests/golden/test_labels.csv' \
     2>/dev/null; then
  echo "FATAL: /tests is still reachable by the unprivileged build user; the verifier's code"
  echo "and answer key are exposed to agent-authored code, so this run cannot be graded."
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi
echo "verifier files closed to the unprivileged build user"

rm -f /app/predictions.csv
# Harbor uploads the agent's sources with whatever mtime they carried in its container,
# which can predate this image's own layers. Cargo decides freshness by mtime, so an older
# source silently counts as "already built" — stamp the collected sources to now so the
# rebuild below is unconditional. Belt and braces with the `cargo clean -p` in tests/Dockerfile.
find /app/predictor/src -type f -exec touch {} +

# Run the crate in its own session, so every process it spawns shares one process group that
# can be torn down as a unit. A bare `runuser` leaves a double-forked child reparented to init
# and still alive during grading, where it could rewrite reward.txt after the reward is
# computed. [verifier].timeout_sec is the only bound on the build: a wall-clock wrapper
# here can fail a slow but correct one.
setsid \
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
  env PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin bash -c 'cd /app/predictor && cargo run --release --offline' &
BUILD_PID=$!
wait "$BUILD_PID"
echo "predictor exit status: $?"

# Reap run 1 before anything else starts. A double-forked child left alive from this run
# would otherwise still be around during the holdout run below, where it could write
# /app/predictions.csv and pass off run 1's numbers as the holdout's.
# `setsid` makes the launched process a session leader, and a session leader's process group
# id is its own pid, so the group is addressed directly as -"$PID". The previous version read
# the group id back out of /proc instead, which races setsid(2): in the window before it takes
# effect /proc still reports the child in THIS shell's group, and the kill then took down the
# verifier itself -- silently, with no pytest output and the reward left at 0.
kill -TERM -- -"$BUILD_PID" 2>/dev/null || true
kill -KILL -- -"$BUILD_PID" 2>/dev/null || true
as_runner bash -c 'kill -9 -1' 2>/dev/null || true

# ---------------------------------------------------------------------------------------
# Second run, on a batch the agent has never seen, from a different welding cell.
#
# Everything graded so far comes from the batch that ships in the agent's own image, so a
# program could fit once and carry the fitted rule in its source without reading train.csv at
# run time. The same binary is therefore run again against a batch from a second cell whose
# settings place every finding in a different region of the process envelope, baked into this
# image at /opt/holdout with its own training file and never present in any container the
# agent can reach. A rule fitted to the shipped cell costs about 48 per weld there against the
# 22.0 bar; a program that fits from the training file it is handed costs about 16.
#
# The agent's own export is put back byte-for-byte afterwards, so test_shipped_data_unmodified
# still checks exactly what the agent left behind.
# Only a regular file is staged. A symlink left at /app/predictions.csv pointing into
# /opt/golden would be followed by pytest, which reads as root, so anything that is not a
# plain file is left behind and named in the log. Written by runner, so the move keeps
# runner's ownership: hand it to root before any more agent code runs, or a later run could
# rewrite an earlier run's graded output.
stage() {
  if [ -L "$1" ] || [ ! -f "$1" ]; then
    echo "$1 is missing or not a regular file, not staged"
    rm -f "$1"
    return 0
  fi
  mv "$1" "$2"
  chown root:root "$2"
  chmod 600 "$2"
}
stage /app/predictions.csv /opt/run_quote.csv

cp -a /app/data /opt/agent_data
cp -f /opt/holdout/train.csv /app/data/train.csv
cp -f /opt/holdout/test_features.csv /app/data/test_features.csv
chmod -R a-w /app/data
rm -f /app/predictions.csv

setsid \
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
  env PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin bash -c 'cd /app/predictor && cargo run --release --offline' &
HOLDOUT_PID=$!
wait "$HOLDOUT_PID"
echo "predictor exit status on holdout batch: $?"
kill -TERM -- -"$HOLDOUT_PID" 2>/dev/null || true
kill -KILL -- -"$HOLDOUT_PID" 2>/dev/null || true
as_runner bash -c 'kill -9 -1' 2>/dev/null || true

stage /app/predictions.csv /opt/run_holdout.csv

# Restore the agent's export exactly as collected.
rm -f /app/data/train.csv /app/data/test_features.csv
cp -a /opt/agent_data/. /app/data/
chmod -R a-w /app/data

# -P keeps the working directory (/app, per the Dockerfile's WORKDIR) off sys.path. The
# crate runs as `runner` and writes into /app, so without this it can drop an /app/pytest.py
# holding sys.exit(0); `python -m pytest` would import that no-op, exit 0, and bypass the
# whole suite for reward 1. The test file and cache are addressed absolutely either way.
/usr/local/bin/python -P -m pytest -o cache_dir=/tmp/pytest_cache \
  --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

# Harbor grades from reward.txt, never from this script's exit status.
exit 0
