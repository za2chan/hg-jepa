"""Real-data property verification for tac.py (no ground truth -> verify
properties, not correctness; docs/tau.md records the findings).

(1) Physical-units sanity: derived tau in seconds vs known fast dynamics.
(2) Per-group stability: tau estimated per subject/patient/bearing;
    median/IQR/min/max. Label-free -> deployment-surviving diagnostic.
Writes runs/tac_real.json.
"""
import json

import numpy as np

from tac import tau_from_tac

# dataset -> (patch samples, sampling Hz, known fast timescale, note)
SPECS = {
    "HAPT":   (4, 50.0, "gait cycle ~1 s"),
    "PTB-XL": (10, 100.0, "beat interval ~0.8 s"),
    "XJTU":   (4, 25600.0, "shaft rotation ~25-29 ms (35-40 Hz)"),
}


def load():
    d = np.load("data/hapt.npz")
    W = d["W"].reshape(len(d["W"]), 128, 4, 3)         # (N, L, patch, xyz)
    sig = np.linalg.norm(W, axis=-1).reshape(len(W), -1)  # accel magnitude
    yield "HAPT", sig, d["subj"]
    d = np.load("data/ptbxl.npz")
    yield "PTB-XL", d["W"].reshape(len(d["W"]), -1), d["pid"]
    d = np.load("data/xjtu_raw.npz")
    bs = sorted({k.split("__")[0] for k in d.files})
    W = np.concatenate([d[f"{b}__W"] for b in bs]).reshape(-1, 512 * 4)
    g = np.concatenate([[b] * len(d[f"{b}__W"]) for b in bs])
    yield "XJTU", W, g


def main():
    out = {}
    for name, sig, groups in load():
        patch, hz, known = SPECS[name]
        sec_per_patch = patch / hz
        r = tau_from_tac(sig, patch=patch)
        taus = {}
        for g in np.unique(groups):
            Wg = sig[groups == g]
            if len(Wg) < 3:
                continue
            taus[str(g)] = tau_from_tac(Wg, patch=patch)["tau"]
        tv = np.array(list(taus.values()))
        out[name] = {
            "per_series_patches": r["per_series"],
            "argmin_series": r["argmin_series"],
            "chosen_patches": r["chosen"],
            "tau_patches": r["tau"], "eps": r["eps"], "c": r["c"],
            "tac_slow_patches": r["tac_slow"], "argmax_series": r["argmax_series"],
            "sec_per_patch": sec_per_patch,
            "tau_seconds": r["tau"] * sec_per_patch,
            "known_fast_dynamics": known,
            "per_group_tau_patches": {
                "n_groups": len(tv), "median": float(np.median(tv)),
                "iqr": [float(np.percentile(tv, 25)), float(np.percentile(tv, 75))],
                "min": float(tv.min()), "max": float(tv.max())},
        }
        print(f"{name}: tau {r['tau']:.2f} patches = {r['tau']*sec_per_patch*1000:.1f} ms "
              f"(argmin {r['argmin_series']}, chosen {r['chosen']:.2f}p) | known: {known} | "
              f"groups n={len(tv)} med {np.median(tv):.2f} "
              f"IQR [{np.percentile(tv,25):.2f},{np.percentile(tv,75):.2f}] "
              f"range [{tv.min():.2f},{tv.max():.2f}]")
    json.dump(out, open("runs/tac_real.json", "w"), indent=2)


if __name__ == "__main__":
    main()
