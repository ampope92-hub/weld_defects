#!/usr/bin/env bash
set -euo pipefail

# Reference model for the weld defect bar.
#
# The findings are steeply imbalanced — 70% of the training welds are clean and
# cracks are 2% — and every classifier here fits for overall accuracy by default.
# On this data that means calling almost everything `none` or `porosity`: about
# 90% right, and it misses most of the cracks. Graded on the mean per-type catch
# rate, where a crack counts as much as a clean weld, that lands at 0.41 and the
# bar is 0.47. More capacity does not help; the objective is the problem.
#
# smartcore has no sample-weight argument on LogisticRegression, so the fix is to
# rebalance the training set itself: draw the same number of rows for each of the
# five findings, with replacement for the rare ones, and fit on that. Every class
# then carries equal weight in the loss. Scores ~0.60.

# The model itself is solution/main.rs, next to this script — Rust is not a shell
# language and 118 lines of it inside a heredoc is neither reviewable nor
# compilable on its own. Drop it in and let cargo build it.
install -m 644 "$(dirname "$0")/main.rs" /app/predictor/src/main.rs

cd /app/predictor
cargo run --release --offline
