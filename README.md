# Weld Defect Triage (Cost-Sensitive Classification in Rust)

A robotic welding shop wants to flag joints worth ultrasonic inspection directly from process data. The agent gets 9,000 inspected welds (`train.csv`, with a five-way `defect` label: none, porosity, undercut, lack_of_fusion, crack), a held-back batch of process features (`test_features.csv`), a data dictionary, and the contracts desk's consequence-cost table. It must fill in a Rust Cargo project (`smartcore` and `csv` pre-wired, offline, no new crates) so that `cargo run --release` reads the data, fits a model, and writes `/app/predictions.csv` with one `weld_id,defect` call per weld. The grade is the mean consequence cost per weld, not accuracy.

## What makes it hard

- **The objective is expected cost, not accuracy.** The cost table is steeply asymmetric: shipping a cracked weld costs 900, pulling a sound one costs 40. The right call is the finding with the lowest expected cost under the posterior, which is often not the most likely one. An argmax classifier is right most of the time and scores 26.4 per weld against a 20.0 bar. The instruction points at the table but never states the decision rule.
- **Class imbalance interacts with the cost rule.** 70% of welds are clean and 2% are cracks. The cost rule needs posteriors that carry the population's priors, and rebalancing - the standard fix for imbalance - shifts exactly those priors. Cost table on rebalanced posteriors: 30.3; rebalance then argmax: 21.1; natural class mix plus cost table: 18.2. The two correct routes are to fit on the natural mix, or rebalance and divide the training priors back out.
- **The library hides the posteriors.** smartcore 0.3.2 has no `predict_proba` and `predict()` returns the argmax, so class probabilities must be rebuilt from `coefficients()` and `intercept()` through a softmax.
- **Leaky columns.** `ut_scan_minutes` and `repair_pass_count` are written after a finding is signed off and carry nothing on the held-back batch; the data dictionary says where each column is written, so they can be ruled out by reading it.

Across five generator seeds, the two correct routes score 17.65-19.48 and 17.63-19.09; the strongest wrong route tops out at 20.90, plain argmax at 29.16, and calling everything a crack at 33.49. The bar sits at 20.0.

## How it's graded

The verifier runs in a separate, offline container. It collects only the crate sources, `Cargo.toml`/`Cargo.lock` and the export, drops them onto a pre-compiled dependency skeleton, and rebuilds and runs the crate itself, so a hand-written predictions file never reaches the grader. The cost table used for grading lives in `test_outputs.py`, not the agent's editable copy. Eleven checks:

- **Shipped batch:** export unaltered (sha256); exact two-column schema; one call per weld; every call a known finding; at least three distinct findings called; mean consequence cost strictly under 20.0 per weld.
- **Holdout re-run:** `/app/data` is swapped for a batch and training file from a different welding cell (process settings in a different region of the envelope), and the same binary runs again. The same structural checks apply, plus a cost strictly under 22.0. A model fitted on the shipped cell and carried across costs 46.7-49.0 there, while a refit costs 15.7-17.1, so only a program that fits from its input passes.

Anti-cheat: agent code is built and run via `setpriv` as an unprivileged user (no supplementary groups, no-new-privs); root's `PATH` is set explicitly and pytest is started by full interpreter path with `-P`; the answer key is root-only in the verifier image and `/tests` is closed (run scored 0 if reachable); outputs are staged only if they are regular files and read with `O_NOFOLLOW`; all build-user processes are killed before grading.

## Repository layout

- `instruction.md` - the task prompt given to the agent.
- `task.toml` - metadata, difficulty/solution/verification write-ups, timeouts and network policy.
- `environment/` - agent image: `Dockerfile`, the starter Cargo crate in `predictor/`, and the export, consequence table and data dictionary in `data/`.
- `solution/` - reference `main.rs` (multinomial logistic regression on the natural class mix, softmax posteriors, minimum-expected-cost decision), `solve.sh`, the generators `generate_data.py` and `generate_holdout.py`, and `calibrate_seeds.py`, which reproduces the route table.
- `tests/` - verifier image, `test.sh`, `test_outputs.py`, the compiled crate skeleton, golden labels and the second-cell holdout.

## Running it

Requires Docker and Harbor (`uv tool install harbor` or `pip install harbor`). From the repo root:

```
harbor run -p . -a oracle
```

runs the reference solution through the verifier (should score reward 1). To try an agent instead:

```
harbor run -p . -a <agent> -m <model>
```

for example `-a claude-code`.

---

Task design, data generators, reference solution and verifier by Audrey Pope.
