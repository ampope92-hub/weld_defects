"""Verifier for the weld defect classifier.

test.sh deletes any predictions.csv the agent left behind and rebuilds and runs the
Cargo project at /app/predictor, so the file graded here is always the output of a
clean release build. We check that the file exists, its schema, that it covers every
held-back weld exactly once, that every call is one of the five recorded findings, that
the calls are not all one class, that the shipped export was not edited, and that the
mean per-defect-type catch rate against /tests/golden/test_labels.csv (never present in
the agent environment) reaches the 0.47 stated in the task.

The score is macro-averaged recall: for each of the five findings, the share of welds
of that type the model called correctly, then averaged with every type counting the
same. Only pytest is installed, so this is computed from the standard library rather
than pandas/scikit-learn.
"""

import csv
import hashlib
import pathlib

import pytest

PRED_PATH = pathlib.Path("/app/predictions.csv")
LABELS_PATH = pathlib.Path("/tests/golden/test_labels.csv")
TEST_FEATURES = pathlib.Path("/app/data/test_features.csv")
MACRO_RECALL_BAR = 0.47
FINDINGS = {"none", "porosity", "undercut", "lack_of_fusion", "crack"}

# frozen at task-build time; the shipped export must not be altered
TEST_FEATURES_SHA = "31c023ae48f5e6ca85674cb088681390666198bce382e9b47ec3354cee3c8233"
TRAIN_SHA = "fa314f02f1e636e18d6f2086bcae1f6a81b73046b507ccb6a507e56629903ce1"


def _read_csv(path):
    with path.open(newline="") as fh:
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


def _macro_recall(called, truth):
    """Mean over findings of the share of that finding's welds called correctly."""
    per_class = []
    for finding in sorted(FINDINGS):
        total = sum(1 for t in truth if t == finding)
        if not total:
            continue
        hit = sum(1 for c, t in zip(called, truth) if t == finding and c == finding)
        per_class.append(hit / total)
    assert per_class, "no findings present in the held-back batch"
    return sum(per_class) / len(per_class)


def test_shipped_data_unmodified():
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    assert _sha(pathlib.Path("/app/data/train.csv")) == TRAIN_SHA, "train.csv was altered"


def test_schema(preds):
    header, _ = preds
    assert header == ["weld_id", "defect"], (
        f"predictions.csv must have exactly columns weld_id,defect, got {header}"
    )


def test_covers_every_weld_exactly_once(preds, labels):
    _, rows = preds
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate weld_id entries"
    missing = set(labels) - set(ids)
    extra = set(ids) - set(labels)
    assert not missing, f"{len(missing)} held-back welds missing (e.g. {sorted(missing)[:3]})"
    assert not extra, f"{len(extra)} unknown weld_ids (e.g. {sorted(extra)[:3]})"


def test_calls_are_known_findings(preds):
    _, rows = preds
    bad = sorted({row[1] for row in rows} - FINDINGS)
    assert not bad, (
        f"defect must be one of {sorted(FINDINGS)}, got unknown value(s): {bad[:5]}"
    )


def test_calls_not_single_class(preds):
    _, rows = preds
    distinct = {row[1] for row in rows}
    assert len(distinct) >= 3, (
        f"the model called only {sorted(distinct)} — that is not a defect classifier"
    )


def test_macro_recall_bar(preds, labels):
    _, rows = preds
    called = [row[1] for row in rows]
    truth = [labels[row[0]] for row in rows]
    score = _macro_recall(called, truth)
    assert score >= MACRO_RECALL_BAR, (
        f"mean per-defect catch rate is {score:.4f}, below the required "
        f"{MACRO_RECALL_BAR:.2f} — the rare defects are being missed"
    )
