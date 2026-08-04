"""Anomaly attribution via horizon-split residuals (the motivating payoff).

Inject typed anomalies into a fresh eval series:
  point:      3-step additive spikes (instantaneous glitch)
  contextual: 400-step +8% frequency shift with NO regime switch (drift from
              the current slow state; invisible to short-window statistics)

Attribution rule (one-directional by design, cf. reviewer #9): a point anomaly
contaminates the target embedding, so it can spike BOTH residuals; a
contextual anomaly should spike only the LONG-horizon residual, which the gate
forces to depend on z_slow alone. We therefore report (a) detection AUROC per
type on its own residual and (b) point-vs-contextual attribution AUROC from
the long/short residual ratio. Long residual uses a median over anchors
(Delta in {64,128}) for target-contamination robustness.

Usage: python3 anomaly.py tag=nepa_g1_d1_s0_tau16_ds16_lam4 [seed=55]
Needs runs/model_<tag>.pt (train.py saves it). Writes runs/anom_<tag>.json.
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datagen import generate
from train import Encoder, Predictor, OFFSETS, P, L, D_Z, W, DEV

N_EVAL = 400_000
TAU = 16.0
D_SLOW = 16
SHORT, LONGS = 1, [64, 128]     # horizons (patches) for the two residuals


def inject(seed=55, ctx_mult=1.08):
    rng = np.random.default_rng(seed)
    fm = np.ones(N_EVAL, np.float32)
    y = np.zeros(N_EVAL, np.int8)                     # 0 normal 1 point 2 contextual
    for _ in range(60):                               # contextual spans
        t0 = rng.integers(10_000, N_EVAL - 10_000)
        fm[t0:t0 + 400] = ctx_mult
        y[t0:t0 + 400] = 2
    x, s, u, phi = generate(N_EVAL, seed=seed, freq_mult=fm)
    for _ in range(300):                              # point spikes
        t0 = rng.integers(10_000, N_EVAL - 10_000)
        if y[t0:t0 + 3].any():
            continue
        x[t0:t0 + 3] += 6.0 * rng.choice([-1, 1])
        y[t0:t0 + 3] = 1
    return x, y


def main(args):
    tag = args.get("tag", "nepa_g1_d1_s0_tau16_ds16_lam4")
    gated = "_g1_" in tag
    ctx_mult = float(args.get("ctx", 1.08))
    x, y = inject(int(args.get("seed", 55)), ctx_mult)
    ck = torch.load(f"runs/model_{tag}.pt", map_location=DEV)
    enc, tgt, pred = Encoder().to(DEV), Encoder().to(DEV), Predictor(D_Z).to(DEV)
    enc.load_state_dict(ck["enc"]); tgt.load_state_dict(ck["tgt"])
    pred.load_state_dict(ck["pred"])
    enc.eval(); tgt.eval(); pred.eval()
    gvals = torch.sigmoid((TAU - torch.tensor(OFFSETS, dtype=torch.float32)) / W).to(DEV)
    oi = {d: i for i, d in enumerate(OFFSETS)}

    def predict_from(z, t_anchor, delta):
        za = z[:, t_anchor]
        g = gvals[oi[delta]] if gated else 1.0
        za_in = torch.cat([za[:, :D_SLOW], za[:, D_SLOW:] * g], -1)
        didx = torch.full((len(za),), oi[delta], device=DEV, dtype=torch.long)
        return pred(za_in, didx)

    # disjoint windows; score patches in the causal-context-rich tail
    starts = np.arange(0, len(x) - L * P, L * P)
    eval_ts = range(160, L)
    rs, rl, yl = [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack(
                [x[s0:s0 + L * P].reshape(L, P) for s0 in bs])).to(DEV)
            z, zbar = enc(xb), tgt(xb)
            for t in eval_ts:
                r_s = ((predict_from(z, t - SHORT, SHORT) - zbar[:, t]) ** 2).mean(-1)
                # long residual on SLOW dims only: fast target dims are
                # unpredictable at Delta>>tau and would swamp the drift signal
                r_ls = torch.stack(
                    [((predict_from(z, t - d, d)[:, :D_SLOW]
                       - zbar[:, t, :D_SLOW]) ** 2).mean(-1) for d in LONGS])
                rs.append(r_s.cpu().numpy())
                rl.append(r_ls.median(0).values.cpu().numpy())
                # patch label: any anomalous step inside the patch
                lab = np.stack([y[s0 + t * P:s0 + (t + 1) * P].max() for s0 in bs])
                yl.append(lab)
    rs, rl, yl = map(np.concatenate, (rs, rl, yl))

    res = {"tag": tag, "ctx_mult": ctx_mult, "n": int(len(yl)),
           "n_point": int((yl == 1).sum()), "n_ctx": int((yl == 2).sum())}
    norm = yl == 0
    res["point_det_auroc_short"] = float(roc_auc_score(yl[norm | (yl == 1)] == 1,
                                                       rs[norm | (yl == 1)]))
    res["ctx_det_auroc_long"] = float(roc_auc_score(yl[norm | (yl == 2)] == 2,
                                                    rl[norm | (yl == 2)]))
    anom = yl > 0
    res["attrib_auroc_ratio"] = float(roc_auc_score(
        yl[anom] == 2, np.log(rl[anom] + 1e-9) - np.log(rs[anom] + 1e-9)))
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs/anom_c{int(ctx_mult*100)}_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main(dict(a.split("=") for a in sys.argv[1:]))
