"""Generates the hidden holdout batch the verifier re-runs the agent's program against.

A different process from the shipped instance (process B in solution/generate_data.py, a
second cell with its own settings) and a different seed. The shipped instance is process A,
seed 0; this must never be 0.

Why it exists: the graded batch ships inside the agent's image, so a program could fit once,
paste the resulting calls into the Rust source as constants, and pass without reading the CSVs
at run time. The verifier swaps /app/data for this batch and runs the same binary again.
weld_ids are W00000.. in both batches, so a program emitting stored calls returns the right
ids labelled for the wrong welds, and its mean consequence cost rises to roughly what a call
that does not depend on the batch is worth. An honest program refits and scores normally.

Bar for that run is a mean consequence cost of 22.0 — deliberately slacker than the graded
batch's 20.0, because it exists to show the program refits rather than to grade it twice.

Writes tests/holdout/{train.csv,test_features.csv,test_labels.csv}. The features go into the
verifier image at /opt/holdout and are copied into /app/data for the second run; the labels
stay root-only and never enter a container the agent can reach.

Usage: python solution/generate_holdout.py <seed>
"""
import hashlib
import pathlib
import sys

from generate_data import SHIPPED_SEED, SHUFFLE_SEED, make_welds

seed = int(sys.argv[1])
if seed == SHIPPED_SEED:
    sys.exit(f"seed {seed} is the shipped instance — the holdout must differ")

out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "holdout"
out.mkdir(parents=True, exist_ok=True)

tr, te = make_welds(seed, process="B")
# Same shuffle the shipped batch got, so row order carries no signal about the label.
te = te.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)

tr.to_csv(out / "train.csv", index=False)
te.drop(columns=["defect"]).to_csv(out / "test_features.csv", index=False)
te[["weld_id", "defect"]].to_csv(out / "test_labels.csv", index=False)

print(f"holdout seed {seed} -> {out}")
print("  class mix (train):")
print(tr.defect.value_counts(normalize=True).round(4).to_string())
for f in ("train.csv", "test_features.csv", "test_labels.csv"):
    b = (out / f).read_bytes()
    print(f"  {f:20s} {len(b):>8d} bytes  sha256 {hashlib.sha256(b).hexdigest()}")
