"""Aggregate the 3-seed hard-regime runs in runs_hard/ into mean/std stats.

Writes hard_3seed_stats.json: {variant: {metric: [mean, std]}}. This is the
source of the README results table; figures.py recomputes the same stats
internally for plotting.
"""
import glob
import json

import numpy as np

VARIANTS = {
    "NEPA gate+dcor (ours)": "nepa_g1_d1",
    "NEPA gate only": "nepa_g1_d0",
    "NEPA nogate": "nepa_g0_d0",
    "NEPA dcor-only": "nepa_g0_d1",
    "AR + gate": "ar_g1_d0",
    "AR no gate": "ar_g0_d0",
}


def main():
    stats = {}
    for name, pre in VARIANTS.items():
        files = glob.glob(f"runs_hard/{pre}_s*_tau16_ds16_lam4.json")
        if not files:
            continue
        rs = [json.load(open(f)) for f in files]
        keys = [k for k in rs[0] if k != "tag"]
        stats[name] = {k: [float(np.mean([r[k] for r in rs])),
                           float(np.std([r[k] for r in rs]))] for k in keys}
    json.dump(stats, open("hard_3seed_stats.json", "w"), indent=2)
    print("aggregated", len(stats), "variants -> hard_3seed_stats.json")


if __name__ == "__main__":
    main()
