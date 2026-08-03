"""Does HAPT's activity F1 actually read a SLOW factor, or just the end of
the window? Two probes on an existing checkpoint (no retraining):

(a) anchor position: probe z_slow at positions near window START vs END.
    If F1 is flat across position, no long-horizon integration is needed —
    the label (end-point sample) is readable locally.
(b) contamination: transition-free windows (one activity across the span)
    vs contaminated windows. If contaminated scores as well as clean, the
    number is not evidence of slow-factor separation.

Same subject-group split as training (test subjects held out).
"""
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from hapt_train import Encoder, L, D_SLOW, DEV, load

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs_hapt/model_hapt_nepa_g1_d1_s0_vf0.pt"


def main():
    Wall, act, accmag, subj = load()
    meta = np.load("data/hapt_meta.npz")
    assert np.array_equal(meta["act"], act), "meta/data misalignment"
    contaminated = meta["spans_transition"]
    ck = torch.load(CKPT, map_location=DEV)
    test_u = set(ck["test_u"])
    is_test = np.array([u in test_u for u in subj])
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)

    enc = Encoder().to(DEV); enc.load_state_dict(ck["enc"]); enc.eval()
    Wt = torch.from_numpy(Wall).to(DEV)
    # embeddings at every position, slice z_slow block
    with torch.no_grad():
        allZ = torch.cat([enc(Wt[i:i + 256]) for i in range(0, len(Wt), 256)]).cpu().numpy()
    zslow = allZ[:, :, :D_SLOW]                              # (N, L, D_SLOW)

    def f1_at(pos_idx, mask=None):
        B = zslow[:, pos_idx, :]
        trm = tr if mask is None else tr[mask[tr]]
        tem = te if mask is None else te[mask[te]]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(B[trm], act[trm])
        return float(f1_score(act[tem], clf.predict(B[tem]), average="macro")), len(tem)

    print(f"checkpoint: {CKPT}")
    print(f"contaminated fraction: {contaminated.mean():.3f} "
          f"(test {contaminated[te].mean():.3f})\n")

    # (a) anchor position sweep (fraction of window length)
    print("(a) z_slow activity F1 vs anchor position (all test windows):")
    for frac in (0.12, 0.25, 0.5, 0.75, 1.0):
        pos = min(int(frac * L), L - 1)
        f1, n = f1_at(pos)
        print(f"    pos {frac:4.0%} (patch {pos:3d}/{L}): F1 {f1:.3f}  (n={n})")

    # (b) clean vs contaminated subset, at the END position (training's readout)
    print("\n(b) z_slow activity F1 by window purity (end-position readout):")
    clean = ~contaminated
    f1c, nc = f1_at(L - 1, mask=clean)
    f1x, nx = f1_at(L - 1, mask=contaminated)
    f1a, na = f1_at(L - 1)
    print(f"    clean (single-activity span): F1 {f1c:.3f}  (n={nc})")
    print(f"    contaminated (straddles):     F1 {f1x:.3f}  (n={nx})")
    print(f"    all:                          F1 {f1a:.3f}  (n={na})")


if __name__ == "__main__":
    main()
