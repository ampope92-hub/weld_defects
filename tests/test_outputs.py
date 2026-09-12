"""Verifier for the weld defect classifier.

test.sh deletes any predictions.csv the agent left behind, rebuilds the Cargo project at
/app/predictor and runs it, so the file graded here is always the output of a clean release
build. It runs the program twice, once on the shipped batch and once on a batch from a
different welding cell that the agent has never seen, with that cell's own training file, and
captures each output to a root-only path. A rule fitted to the shipped cell and carried in the
source misplaces every finding on the second cell and fails its bar; a program that fits from
the training file it is handed does not.

The score is the mean CONSEQUENCE COST per weld, and lower is better. Each call is charged
against the contracts table: what the plant pays when a weld that really had one finding is
called another. That table is steeply asymmetric — shipping a cracked weld costs 900, pulling a
sound one costs 40 — so accuracy and cost come apart, and the cheapest call for a weld is not in
general the most likely one. A classifier taking the argmax of its posterior is right most of
the time and scores 26.4 against the 20.0 bar.

The table lives in this file rather than being read from /app/data. The agent is shipped a copy
as domain evidence, but a verifier grading against the agent's copy would be grading against
numbers the agent can edit. Both come from cost_matrix() in solution/calibrate_seeds.py, which
emits the shipped CSV with `--emit-costs`, so the duplication has one source.

Only pytest is installed, so this is computed from the standard library.
"""

import csv
import hashlib
import os
import pathlib

import pytest

# test.sh runs the agent's program twice and moves each output aside root-only: first on the
# graded batch that shipped in the agent's image, then on a holdout batch it has never seen.
PRED_PATH = pathlib.Path("/opt/run_quote.csv")
HOLDOUT_PRED_PATH = pathlib.Path("/opt/run_holdout.csv")
HOLDOUT_LABELS_PATH = pathlib.Path("/opt/holdout/test_labels.csv")
# Read from the root-only copy baked into the verifier image, not the /tests mount: the
# agent's crate is rebuilt and run in this container, and only this path is closed to it.
LABELS_PATH = pathlib.Path("/opt/golden/test_labels.csv")
TEST_FEATURES = pathlib.Path("/app/data/test_features.csv")
# The bar sits above every route that ignores the class imbalance or falls for the provenance
# trap, and below every honest partly-rebalanced fit. Calibrated across five generator seeds in
# solution/calibrate_seeds.py: the two correct routes score 17.65-19.48 and 17.63-19.09, the
# strongest wrong route (rebalance, then take the argmax) ceilings at 20.90, plain argmax at
# 29.16, and calling every weld a crack at 33.49.
COST_BAR = 20.0            # mean consequence cost per weld; LOWER is better
# Bar for the holdout run, and slacker than the graded batch's because cost is lower-is-better:
# that run exists to show the program refits from whatever it is handed, not to grade the model
# twice, and a batch the author never calibrated against should not fail a good solution on
# ordinary seed variation. Every correct route costs at most 19.48 across the calibration seeds,
# while a program re-emitting calls fitted to the other batch lands around the 33 a
# batch-independent guess is worth, so this sits between them with room on both sides.
# instruction.md names this figure, so there is no graded threshold the agent was not told about.
HOLDOUT_COST_BAR = 22.0
FINDINGS = {"none", "porosity", "undercut", "lack_of_fusion", "crack"}

# frozen at task-build time; the shipped export must not be altered
TEST_FEATURES_SHA = "b8f389b0a9162bb738e878279f4b40bb48ad931b21df30cd60c64568f0542fba"
TRAIN_SHA = "8096285d66827bfeec8c6ce90faf6d00fc3efe96db0edcea821129481a7c5e03"


def _read_csv(path):
    # The graded files are agent output read by root, so the open itself refuses to follow a
    # symlink; test.sh already declines to stage one, and this is the second lock on that door.
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, [row for row in reader if row]


@pytest.fixture(scope="session")
def preds():
    assert PRED_PATH.exists(), (
        "/app/predictions.csv was not written — `cargo run --release` in "
        "/app/predictor must produce it"
    )
    return _read_csv(PRED_PATH)


@pytest.fixture(scope="session")
def labels():
    _, rows = _read_csv(LABELS_PATH)
    return {row[0]: row[1] for row in rows}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The grading contract, held here rather than read from /app/data: the agent is shipped a copy
# of this table as domain evidence, and a verifier that read the agent's copy would be graded
# against numbers the agent could edit. Rows are the finding the weld actually had, columns the
# finding it was called. Kept in sync with environment/data/consequence_costs.csv.
_ORDER = ["none", "porosity", "undercut", "lack_of_fusion", "crack"]
_COST = {
    "none":           {"none": 0.0, "porosity": 40.0, "undercut": 40.0, "lack_of_fusion": 40.0, "crack": 40.0},
    "porosity":       {"none": 25.0, "porosity": 0.0, "undercut": 6.25, "lack_of_fusion": 6.25, "crack": 6.25},
    "undercut":       {"none": 60.0, "porosity": 15.0, "undercut": 0.0, "lack_of_fusion": 15.0, "crack": 15.0},
    "lack_of_fusion": {"none": 350.0, "porosity": 87.5, "undercut": 87.5, "lack_of_fusion": 0.0, "crack": 87.5},
    "crack":          {"none": 900.0, "porosity": 225.0, "undercut": 225.0, "lack_of_fusion": 225.0, "crack": 0.0},
}


def _mean_cost(called, truth):
    """Mean consequence cost per weld: what the batch's calls add up to on the contract table.

    Lower is better. A model can be right most of the time and still be expensive, because the
    table is steeply asymmetric — shipping a cracked weld costs 900 against 40 for pulling a
    sound one.
    """
    assert called, "no calls to score"
    return sum(_COST[t][c] for c, t in zip(called, truth)) / len(called)


def test_shipped_data_unmodified():
    """The shipped train.csv and test_features.csv are byte-for-byte as issued."""
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    assert _sha(pathlib.Path("/app/data/train.csv")) == TRAIN_SHA, "train.csv was altered"


def test_schema(preds):
    """predictions.csv carries exactly the two columns weld_id and defect."""
    header, rows = preds
    assert header == ["weld_id", "defect"], (
        f"predictions.csv must have exactly columns weld_id,defect, got {header}"
    )
    wide = [i for i, row in enumerate(rows, start=2) if len(row) != 2]
    assert not wide, (
        f"{len(wide)} row(s) do not carry exactly two fields (e.g. line {wide[0]}: "
        f"{rows[wide[0] - 2]}) — the file must be weld_id,defect and nothing else"
    )


def test_covers_every_weld_exactly_once(preds, labels):
    """Every held-back weld has exactly one row — no duplicates, omissions or unknown ids."""
    _, rows = preds
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate weld_id entries"
    missing = set(labels) - set(ids)
    extra = set(ids) - set(labels)
    assert not missing, f"{len(missing)} held-back welds missing (e.g. {sorted(missing)[:3]})"
    assert not extra, f"{len(extra)} unknown weld_ids (e.g. {sorted(extra)[:3]})"


def test_calls_are_known_findings(preds):
    """Every call is one of the five findings the inspection log records."""
    _, rows = preds
    bad = sorted({row[1] for row in rows} - FINDINGS)
    assert not bad, (
        f"defect must be one of {sorted(FINDINGS)}, got unknown value(s): {bad[:5]}"
    )


def test_calls_not_single_class(preds):
    """The classifier calls at least three distinct findings rather than collapsing."""
    _, rows = preds
    distinct = {row[1] for row in rows}
    assert len(distinct) >= 3, (
        f"the model called only {sorted(distinct)} — that is not a defect classifier"
    )


def test_cost_bar(preds, labels):
    """The batch's calls come in under the required mean consequence cost of 20.0 per weld."""
    _, rows = preds
    called = [row[1] for row in rows]
    truth = [labels[row[0]] for row in rows]
    score = _mean_cost(called, truth)
    assert score < COST_BAR, (
        f"mean consequence cost is {score:.2f} per weld, not under the required {COST_BAR:.1f} — "
        "the calls are accurate-but-expensive: the table is asymmetric and the costly misses "
        "are not being bought off"
    )


@pytest.fixture(scope="session")
def holdout_preds():
    """The same program's output for the holdout batch."""
    assert HOLDOUT_PRED_PATH.exists(), (
        "the predictor wrote no /app/predictions.csv when it was run against the holdout "
        "batch — it must read and fit from the CSVs in /app/data at run time"
    )
    return _read_csv(HOLDOUT_PRED_PATH)


@pytest.fixture(scope="session")
def holdout_labels():
    _, rows = _read_csv(HOLDOUT_LABELS_PATH)
    return {row[0]: row[1] for row in rows}


def test_holdout_schema(holdout_preds):
    """The holdout run writes the same two-column file the graded run has to write."""
    header, rows = holdout_preds
    assert header == ["weld_id", "defect"], (
        f"the holdout run's predictions.csv must have exactly columns weld_id,defect, "
        f"got {header}"
    )
    wide = [i for i, row in enumerate(rows, start=2) if len(row) != 2]
    assert not wide, (
        f"{len(wide)} row(s) of the holdout run do not carry exactly two fields (e.g. line "
        f"{wide[0]}: {rows[wide[0] - 2]})"
    )


def test_holdout_calls_are_known_findings(holdout_preds):
    """Every call in the holdout run is one of the five findings."""
    _, rows = holdout_preds
    bad = sorted({row[1] for row in rows} - FINDINGS)
    assert not bad, (
        f"defect must be one of {sorted(FINDINGS)}, got unknown value(s) on the holdout "
        f"batch: {bad[:5]}"
    )


def test_holdout_calls_not_single_class(holdout_preds):
    """The holdout run calls at least three distinct findings, as the graded run must."""
    _, rows = holdout_preds
    distinct = {row[1] for row in rows}
    assert len(distinct) >= 3, (
        f"on the holdout batch the model called only {sorted(distinct)} — that is not a "
        "defect classifier"
    )


def test_holdout_covers_every_weld(holdout_preds, holdout_labels):
    """Run on an unseen batch, the program calls every weld in it exactly once."""
    _, rows = holdout_preds
    called = [row[0] for row in rows]
    assert len(called) == len(set(called)), "the holdout run repeated a weld_id"
    assert set(called) == set(holdout_labels), (
        "the holdout run did not call exactly the welds in that batch — the program is not "
        "reading weld_ids from the test_features.csv it was given"
    )


def test_holdout_cost(holdout_preds, holdout_labels):
    """The mean consequence cost holds up on a batch the agent never saw."""
    _, rows = holdout_preds
    called = [row[1] for row in rows]
    truth = [holdout_labels[row[0]] for row in rows]
    score = _mean_cost(called, truth)
    assert score < HOLDOUT_COST_BAR, (
        f"mean consequence cost on the holdout batch is {score:.2f}, not under "
        f"{HOLDOUT_COST_BAR:.1f}. The same program was graded on the shipped batch, so either it "
        "is re-emitting calls fitted to that batch rather than fitting the CSVs it was given, or "
        "the design does not carry over to a fresh draw"
    )
