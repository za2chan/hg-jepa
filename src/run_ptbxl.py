"""PTB-XL v2 (static slow factor = NORM diagnosis). Reuses train_real() for
training; the slow factor is a GLOBAL per-record label, so the natural readout
is the LAST position (full-record causal summary), not multi-position (which is
for a VARYING slow factor like HAPT activity). Shift robustness uses ECG-
realistic perturbations (noise, amplitude, baseline wander).
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from model import D_Z, D_SLOW
from train_real import train_real, rankme, DEV

SL = {"z_slow": slice(0, D_SLOW), "z_fast": slice(D_SLOW, D_Z), "z_full": slice(0, D_Z)}


@torch.no_grad()
def _enc_last(enc, Wc):
    return torch.cat([enc(Wc[i:i + 128])[:, -1] for i in range(0, len(Wc), 128)]).cpu().numpy()


def probe_static(res):
    enc, Wt, lab, fast, tr, te = (res["enc"], res["Wt"], res["lab"], res["fast"],
                                  res["tr"], res["te"])
    Zl = _enc_last(enc, Wt)                          # (n, D_Z) full-record summary
    y = lab[:, -1]; fz = fast[:, -1]                 # per-record NORM / fast proxy
    out = {}
    for b, sl in SL.items():
        B = Zl[:, sl]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(B[tr], y[tr])
        f1 = f1_score(y[te], clf.predict(B[te]), average="macro")
        leak = make_pipeline(StandardScaler(), Ridge()).fit(B[tr], fz[tr]).score(B[te], fz[te])
        out[b] = dict(slow_kept_f1=float(f1), leak_r2=float(leak),
                      rankme=float(rankme(B[te])))
    return out


@torch.no_grad()
def shift_static(res, strengths=(0.0, 0.25, 0.5, 1.0, 2.0), kind="noise", seed=0):
    enc, Wt, lab, tr, te = res["enc"], res["Wt"], res["lab"], res["tr"], res["te"]
    y = lab[:, -1]
    g = torch.Generator(device=DEV).manual_seed(seed)
    Lw = Wt.shape[1]
    t = torch.linspace(0, 1, Lw * (Wt.shape[2] // 1), device=DEV)  # sample-time (approx)
    Zc = _enc_last(enc, Wt)
    Wte = Wt[te]
    out = dict(strengths=list(strengths), kind=kind)
    for b in ("z_slow", "z_full"):
        sl = SL[b]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Zc[tr][:, sl], y[tr])
        f1s = []
        for s in strengths:
            if kind == "noise":
                Wp = Wte + s * torch.randn(Wte.shape, generator=g, device=DEV)
            elif kind == "scale":
                Wp = Wte * (1.0 + s)
            else:                                    # baseline wander: low-freq drift added
                ph = torch.linspace(0, 2 * np.pi, Wte.shape[1], device=DEV)[None, :, None]
                Wp = Wte + s * torch.sin(ph)         # ~1 cycle across the 10 s record
            f1s.append(float(f1_score(y[te], clf.predict(_enc_last(enc, Wp)[:, sl]), average="macro")))
        out[b] = f1s
    return out


if __name__ == "__main__":
    os.makedirs("../runs_v2", exist_ok=True)
    print("=== PTB-XL v2 (static NORM; HGLP-Reg, bounded target, gate+xcov) ===")
    res = train_real("../data/ptbxl_v2.npz", n_ax=1, tau=16.0, w=8, dmin=8, dmax=48,
                     min_context=8, seed=0)
    pr = probe_static(res)
    for b, r in pr.items():
        print(f"  {b:7s} NORM-F1 {r['slow_kept_f1']:.3f} | leak(ecg R2) {r['leak_r2']:.3f} "
              f"| RankMe {r['rankme']:.1f}")
    print("--- shift robustness (fit clean, eval perturbed) ---")
    sh = {k: shift_static(res, kind=k) for k in ("noise", "scale", "wander")}
    for k, s in sh.items():
        rz = s["z_slow"][-1] / s["z_slow"][0]; rf = s["z_full"][-1] / s["z_full"][0]
        print(f"  {k:7s} z_slow {[round(v,3) for v in s['z_slow']]} (ret {rz:.0%}) | "
              f"z_full {[round(v,3) for v in s['z_full']]} (ret {rf:.0%})")
    json.dump(dict(probe=pr, shift=sh), open("../runs_v2/ptbxl_v2_seed0.json", "w"), indent=2)
    print("saved runs_v2/ptbxl_v2_seed0.json")
