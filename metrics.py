"""MIG + DCI on saved probe embeddings (runs/emb_*.npz) vs ground-truth
factors (regime, u, phase-sin/cos). Reviewer #4: disentanglement metrics
beside our block matrix.

Usage: python3 metrics.py runs/emb_nepa_g1_d1_s0_*.npz ...
Prints per-file MIG and DCI-disentanglement; aggregates mean over files.
"""
import glob
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def factors(d):
    ph = np.arctan2(d["phase"][:, 0], d["phase"][:, 1])
    return {"regime": (d["regime"], True), "u": (d["u"], False), "phase": (ph, False)}


def mig(Z, fac):
    # per-dim MI with each factor; MIG = mean over factors of (top1-top2)/H
    gaps = []
    for y, disc in fac.values():
        mi = (mutual_info_classif if disc else mutual_info_regression)(
            Z, y, random_state=0)
        if disc:
            _, c = np.unique(y, return_counts=True)
            H = -(c / c.sum() * np.log(c / c.sum())).sum()
        else:
            yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, 21)[1:-1]))
            _, c = np.unique(yb, return_counts=True)
            H = -(c / c.sum() * np.log(c / c.sum())).sum()
        s = np.sort(mi)[::-1]
        gaps.append((s[0] - s[1]) / H)
    return float(np.mean(gaps))


def dci(Z, fac):
    # importance matrix R (dims x factors) from small random forests
    R = []
    for y, disc in fac.values():
        m = (RandomForestClassifier if disc else RandomForestRegressor)(
            n_estimators=40, max_depth=8, random_state=0, n_jobs=8).fit(Z, y)
        R.append(m.feature_importances_)
    R = np.stack(R, 1) + 1e-11                    # (D, F)
    P = R / R.sum(1, keepdims=True)               # per-dim distribution over factors
    H = -(P * np.log(P) / np.log(R.shape[1])).sum(1)
    rho = R.sum(1) / R.sum()
    return float((rho * (1 - H)).sum())           # DCI disentanglement


def main(paths):
    ms, ds = [], []
    for p in paths:
        d = np.load(p)
        Z = d["Z"]
        fac = factors(d)
        m, c = mig(Z, fac), dci(Z, fac)
        ms.append(m); ds.append(c)
        print(f"{p}: MIG {m:.3f}  DCI-dis {c:.3f}", flush=True)
    print(f"MEAN ({len(ms)} files): MIG {np.mean(ms):.3f}  DCI-dis {np.mean(ds):.3f}")


if __name__ == "__main__":
    main([f for a in sys.argv[1:] for f in sorted(glob.glob(a))])
