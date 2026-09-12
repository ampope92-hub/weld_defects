#!/usr/bin/env bash
set -euo pipefail

# Reference model for the weld consequence-cost bar.
#
# Three things have to be right, and only the first is obvious.
#
# 1. The findings are steeply imbalanced — 70% of the training welds are clean and cracks are
#    2% — so a classifier fitted for accuracy calls almost everything `none` or `porosity`.
#
# 2. The grade is not accuracy and not a catch rate: it is what the calls cost on the contracts
#    table, averaged per weld. That table is steeply asymmetric — shipping a cracked weld costs
#    900 against 40 for pulling a sound one — so the right call is the finding with the lowest
#    EXPECTED cost, not the most likely finding. Taking the argmax of the posterior scores 26.4
#    against a 20.0 bar even though it is right most of the time.
#
# 3. The two interact, and getting that wrong is worse than ignoring costs. The cost rule needs
#    posteriors carrying the population's own class priors. Resampling to fix the imbalance
#    shifts those priors, so applying the table to rebalanced posteriors scores 30.3 — worse
#    than rebalancing and taking the argmax (21.1), and far worse than the 18.2 here. This
#    reference simply does not rebalance; dividing the training priors back out afterwards
#    works equally well.
#
# smartcore exposes no predict_proba and its predict() takes the argmax, so the posteriors are
# rebuilt from model.coefficients() and model.intercept() and passed through a softmax.


# The model itself is solution/main.rs, next to this script — Rust is not a shell heredoc.
install -m 644 "$(dirname "$0")/main.rs" /app/predictor/src/main.rs

cd /app/predictor
cargo run --release --offline
