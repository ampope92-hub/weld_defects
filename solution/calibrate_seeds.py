"""Reproduces the route table difficulty_explanation quotes, across five generator seeds.

Each row is a decision rule applied to the same fitted posteriors, scored on the shipped
consequence table. The bar (20.0) sits between the worst correct route and the best wrong one.

    ./.venv-calib/bin/python solution/calibrate_seeds.py
"""
from collections import Counter

import numpy as np
from generate_data import CLASSES, make_welds
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HONEST = ["heat_input_kj_mm", "travel_speed_mm_s", "gas_flow_l_min", "root_gap_mm",
          "plate_thk_mm", "preheat_c"]
MISS = {"none": 0.0, "porosity": 25.0, "undercut": 60.0, "lack_of_fusion": 350.0, "crack": 900.0}
FALSE = 40.0
BAR = 20.0


def cost_matrix():
    """costs[actual][called], matching environment/data/consequence_costs.csv."""
    C = np.zeros((len(CLASSES), len(CLASSES)))
    for i, t in enumerate(CLASSES):
        for j, c in enumerate(CLASSES):
            if t == c:
                C[i, j] = 0.0
            elif c == "none":
                C[i, j] = MISS[t]
            elif t == "none":
                C[i, j] = FALSE
            else:
                C[i, j] = 0.25 * MISS[t]
    return C


def mean_cost(calls, truth, C):
    ti = {c: i for i, c in enumerate(CLASSES)}
    return np.mean([C[ti[t], ti[c]] for c, t in zip(calls, truth)])


def cheapest(P, cls, C):
    """The call with the lowest expected cost under each row's posterior."""
    j = [cls.index(c) for c in CLASSES]
    return [CLASSES[int(np.argmin(P[i][j] @ C))] for i in range(len(P))]


def balanced_index(y, seed):
    return np.concatenate([
        np.random.default_rng(seed).choice(
            np.where(y == c)[0], int(len(y) / len(np.unique(y))), replace=True)
        for c in np.unique(y)])


def write_cost_csv(path):
    """Emit environment/data/consequence_costs.csv from cost_matrix().

    Single source of truth. The table is duplicated by necessity — the verifier cannot grade
    against the agent's copy, which the agent can edit — so it is generated here rather than
    hand-maintained. Two decimals: 0.25 * 25.0 = 6.25 does not survive one.
    """
    with open(path, "w", newline="") as fh:
        fh.write("actual," + ",".join(CLASSES) + "\n")
        C = cost_matrix()
        fh.writelines(t + "," + ",".join(f"{C[i, j]:.2f}" for j in range(len(CLASSES))) + "\n" for i, t in enumerate(CLASSES))


if __name__ == "__main__":
    import sys
    if "--emit-costs" in sys.argv:
        import pathlib as _p
        out = _p.Path(__file__).resolve().parent.parent / "environment/data/consequence_costs.csv"
        write_cost_csv(out)
        print(f"wrote {out}")
        raise SystemExit(0)
    C = cost_matrix()
    hdr = ("seed", "plain+cost", "rebal+fix+cost", "argmax", "rebal+argmax",
           "rebal+cost(nofix)", "all-crack")
    print(f"{hdr[0]:>4} {hdr[1]:>11} {hdr[2]:>15} {hdr[3]:>8} {hdr[4]:>13} {hdr[5]:>18} {hdr[6]:>10}")
    worst_correct, best_wrong = 0.0, 1e9
    for s in range(5):
        tr, te = make_welds(s)
        Xtr, ytr, Xte, yte = tr[HONEST], tr.defect, te[HONEST], te.defect
        plain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
        idx = balanced_index(ytr, s)
        reb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(
            Xtr.iloc[idx], ytr.iloc[idx])
        P_plain, P_reb = plain.predict_proba(Xte), reb.predict_proba(Xte)
        prior = np.array([(ytr == c).mean() for c in list(reb.classes_)])
        P_fix = P_reb * prior / (1.0 / len(CLASSES))
        P_fix /= P_fix.sum(1, keepdims=True)

        correct = [mean_cost(cheapest(P_plain, list(plain.classes_), C), yte, C),
                   mean_cost(cheapest(P_fix, list(reb.classes_), C), yte, C)]
        wrong = [mean_cost(plain.predict(Xte), yte, C),
                 mean_cost(reb.predict(Xte), yte, C),
                 mean_cost(cheapest(P_reb, list(reb.classes_), C), yte, C),
                 mean_cost(["crack"] * len(yte), yte, C)]
        worst_correct = max(worst_correct, *correct)
        best_wrong = min(best_wrong, *wrong)
        print(f"{s:4d} {correct[0]:11.2f} {correct[1]:15.2f} {wrong[0]:8.2f} "
              f"{wrong[1]:13.2f} {wrong[2]:18.2f} {wrong[3]:10.2f}")
        if s == 0:
            print("      calls made by plain+cost:",
                  dict(Counter(cheapest(P_plain, list(plain.classes_), C))))
    print()
    print(f"worst correct route over seeds {worst_correct:.2f}")
    print(f"best wrong route over seeds    {best_wrong:.2f}")
    print(f"bar {BAR:.1f} sits between them "
          f"({BAR - worst_correct:+.2f} / {best_wrong - BAR:+.2f} clear)")
