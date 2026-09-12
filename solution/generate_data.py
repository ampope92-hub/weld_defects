"""Weld defect classification generator (cost-sensitive decisions + class imbalance).

Design:
  * Five defect classes with a steep prior: the common class is ~70% of rows and the
    two rare classes are ~2% each. Fitting for accuracy, which is what every default
    does, buys almost nothing by getting the rare classes right — so the default model
    predicts the common classes almost everywhere.
  * The graded metric is the mean CONSEQUENCE COST per weld, charged against the
    contracts table shipped in environment/data/consequence_costs.csv and emitted from
    cost_matrix() in calibrate_seeds.py. It is steeply asymmetric — shipping a cracked
    weld costs 900, pulling a sound one costs 40 — so the cheapest call is not in
    general the most likely one, and taking the argmax of a posterior scores 26.4
    against a 20.0 bar.
  * The two interact, which is the load-bearing difficulty. The cost rule needs
    posteriors carrying the population's own class priors, and rebalancing the training
    set — the standard fix for the imbalance above — shifts exactly those priors.
    Applying the table to rebalanced posteriors scores 30.3, worse than rebalancing and
    taking the argmax (21.1) and far worse than not rebalancing at all (18.2).
  * The rare classes are genuinely separable — they have their own signal — so this is
    not a hopeless-minority problem, it is a decision-rule problem.
  * ut_scan_minutes and repair_pass_count are drawn last: as-inspected figures on the
    signed-off cohort, scheduler bookings on the held-back batch. A measured round
    showed both frontier models detect and exclude them, so they are a hazard a careful
    agent avoids rather than the difficulty.
"""
import pathlib

import numpy as np
import pandas as pd

# The shipped instance; solution/generate_holdout.py refuses this seed.
SHIPPED_SEED = 0
# Row shuffle applied to the held-back batch, shared with the holdout batch.
SHUFFLE_SEED = 55

CLASSES = ["none", "porosity", "undercut", "lack_of_fusion", "crack"]
PRIOR = np.array([0.70, 0.16, 0.08, 0.04, 0.02])


SPREAD = 1.8


# Where each finding sits in the process envelope. Process A is the cell the shipped batch
# comes from. Process B is a different cell with its own settings, which is what the held-back
# batch is drawn from: a model fitted once to the shipped batch and carried across places every
# finding in the wrong region there, while a model refitted to that batch's own training file
# does not. That is what proves at grading time that the program reads train.csv.
PROCESSES = {
    "A": {
        "prior": PRIOR,
        "centres": {
            0: {"heat": 1.95, "speed": 42.0, "gas": 18.0, "gap": 0.9, "gap_sd": 0.25},
            1: {"heat": 2.55, "speed": 38.0, "gas": 11.5, "gap": 1.0, "gap_sd": 0.30},
            2: {"heat": 2.70, "speed": 52.0, "gas": 17.0, "gap": 0.8, "gap_sd": 0.22},
            3: {"heat": 1.35, "speed": 49.0, "gas": 17.5, "gap": 1.6, "gap_sd": 0.35},
            4: {"heat": 3.15, "speed": 35.0, "gas": 16.5, "gap": 1.3, "gap_sd": 0.40},
        },
    },
    "B": {
        "prior": np.array([0.66, 0.14, 0.11, 0.06, 0.03]),
        "centres": {
            0: {"heat": 2.75, "speed": 50.0, "gas": 12.5, "gap": 1.4, "gap_sd": 0.25},
            1: {"heat": 1.60, "speed": 56.0, "gas": 19.0, "gap": 0.7, "gap_sd": 0.30},
            2: {"heat": 1.45, "speed": 36.0, "gas": 13.0, "gap": 1.7, "gap_sd": 0.22},
            3: {"heat": 3.20, "speed": 41.0, "gas": 11.0, "gap": 0.6, "gap_sd": 0.35},
            4: {"heat": 2.20, "speed": 62.0, "gas": 20.5, "gap": 1.1, "gap_sd": 0.40},
        },
    },
}


def make_welds(seed=0, n_train=9000, n_test=4000, process="A"):
    rng = np.random.default_rng(seed)
    proc = PROCESSES[process]

    def block(n):
        y = rng.choice(len(CLASSES), n, p=proc["prior"])
        # each class sits in its own region of the process envelope
        centres = proc["centres"]
        heat = np.array([rng.normal(centres[c]["heat"], 0.38 * SPREAD) for c in y])
        speed = np.array([rng.normal(centres[c]["speed"], 6.0 * SPREAD) for c in y])
        gas = np.array([rng.normal(centres[c]["gas"], 2.2 * SPREAD) for c in y])
        gap = np.array([rng.normal(centres[c]["gap"], centres[c]["gap_sd"] * SPREAD) for c in y])
        return pd.DataFrame({
            "heat_input_kj_mm": heat.clip(0.4, 5.0).round(3),
            "travel_speed_mm_s": speed.clip(15, 80).round(2),
            "gas_flow_l_min": gas.clip(4, 28).round(2),
            "root_gap_mm": gap.clip(0.1, 3.2).round(3),
            "plate_thk_mm": rng.choice([6, 8, 10, 12, 16], n),
            "preheat_c": rng.normal(85, 30, n).clip(15, 220).round(0),
            "wire_lot": rng.choice([f"L{i:02d}" for i in range(12)], n),
            "defect": [CLASSES[c] for c in y],
        }), y

    tr, ytr = block(n_train)
    te, _yte = block(n_test)

    # ---- provenance columns -------------------------------------------------------------
    # Both fields are carried on the weld's UT work order. The signed-off cohort is exported
    # after its work orders have been reconciled against the inspection reports, so there the
    # two fields are the as-inspected figures — consequences of the finding. The held-back
    # batch is pulled off the scheduling system in the same export run, BEFORE reconciliation,
    # so there the two fields are the figures the scheduler booked against the job queue.
    #
    # That is the trap, and the whole point is that it leaves no mark on the held-back batch.
    # The booked figures are drawn from the same mix as the reconciled ones — the schedule is
    # built from the historical finding rate — so both columns have the same range, spread and
    # marginal distribution in both cohorts. A `.describe()` comparison, a drift/PSI screen or
    # a train-vs-test histogram sees nothing; the columns simply do not describe THAT weld's
    # finding, and the agent has no labels on the held-back batch with which to notice.
    #
    # Draw order is deliberate: the two tr columns and the te noise term are drawn in the same
    # sequence and shape as the previous export, so train.csv stays bit-identical and only
    # test_features.csv moves. `scheduled` is drawn last, independent of the batch's real
    # findings, and is what the te columns are composed from.
    SCAN_EXTRA = {0: 0.0, 1: 2.4, 2: 3.1, 3: 5.2, 4: 6.8}
    REPAIR_P = {0: 0.02, 1: 0.55, 2: 0.45, 3: 0.85, 4: 0.95}

    tr["ut_scan_minutes"] = np.round(
        6.0 + np.array([SCAN_EXTRA[c] for c in ytr]) + rng.normal(0, 0.9, len(tr)), 2
    ).clip(3.0, 30.0)
    te_scan_noise = rng.normal(0, 0.9, len(te))
    tr["repair_pass_count"] = np.array(
        [rng.binomial(2, REPAIR_P[c]) for c in ytr], dtype=int
    )

    # The finding the scheduler had booked against each held-back weld, drawn from the same
    # rate as the cohort but independent of what that weld actually turned out to have.
    scheduled = rng.choice(len(CLASSES), len(te), p=proc["prior"])
    te["ut_scan_minutes"] = np.round(
        6.0 + np.array([SCAN_EXTRA[c] for c in scheduled]) + te_scan_noise, 2
    ).clip(3.0, 30.0)
    te["repair_pass_count"] = np.array(
        [rng.binomial(2, REPAIR_P[c]) for c in scheduled], dtype=int
    )

    # Keep the label last so the two provenance columns sit among the ordinary features
    # rather than trailing the label, which would flag them as an afterthought.
    tr = tr[[c for c in tr.columns if c != "defect"] + ["defect"]]
    te = te[[c for c in te.columns if c != "defect"] + ["defect"]]
    te.insert(0, "weld_id", [f"W{i:05d}" for i in range(len(te))])
    return tr, te


if __name__ == "__main__":
    import hashlib

    # The shipped instance is seed 0, with the held-back batch shuffled so row order
    # carries no signal about the finding. This rewrites environment/data/{train.csv,
    # test_features.csv} and tests/golden/test_labels.csv byte for byte.
    root = pathlib.Path(__file__).resolve().parent.parent
    tr, te = make_welds(SHIPPED_SEED)
    te = te.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)

    (root / "environment" / "data").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "golden").mkdir(parents=True, exist_ok=True)
    tr.to_csv(root / "environment" / "data" / "train.csv", index=False)
    te.drop(columns=["defect"]).to_csv(
        root / "environment" / "data" / "test_features.csv", index=False
    )
    te[["weld_id", "defect"]].to_csv(root / "tests" / "golden" / "test_labels.csv", index=False)

    print(tr.defect.value_counts(normalize=True).round(4).to_string())
    print(f"\ntrain {len(tr)}, test {len(te)}")
    for rel in ("environment/data/train.csv", "environment/data/test_features.csv",
                "tests/golden/test_labels.csv"):
        b = (root / rel).read_bytes()
        print(f"  {rel:38s} {len(b):>9d} bytes  sha256 "
              f"{hashlib.sha256(b).hexdigest()}")
