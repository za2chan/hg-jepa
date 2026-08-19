"""SEP split into its three terms, with per-seed spread, for every setting.

The paper reports SEP as one number. That hides which term a method actually
loses on, which is exactly what a reader needs to diagnose a loss against SFA:
the same SEP can come from failing inclusion, failing allocation, or failing
exclusion, and those imply different fixes. Everything needed is already stored --
rotation_*.json and twosided_*.json both keep per-seed slow/fast scores per block
-- so this recomputes and never retrains.

Also emits the two alternative aggregations (sum and harmonic mean) so a reader
can check that the ranking is not an artefact of taking a product.

python3 src/exp/term_breakdown.py -> runs_v2/term_breakdown.json (+ markdown to stdout)
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "hglp"))
RUNS = ROOT / "runs_v2"

# rotation files: (label, filename stem). Sleep-EDF uses the patch-10 config A,
# the only one whose tau comes from the rule rather than by hand.
ROT = [
    ("Synthetic/Reg",  "rotation_synth_l1+ema_gap0.03"),
    ("Synthetic/NCE",  "rotation_synth_nce+ema_gap0.03"),
    ("PTB-XL/Reg",     "rotation_ptbxl_l1+ema"),
    ("PTB-XL/NCE",     "rotation_ptbxl_nce+ema"),
    ("HAPT/Reg",       "rotation_hapt_l1+ema"),
    ("HAPT/NCE",       "rotation_hapt_nce+ema"),
    ("Sleep-EDF/Reg",  "rotation_sleepedf_l1+ema_p10A"),
    ("Sleep-EDF/NCE",  "rotation_sleepedf_nce+ema_p10A"),
]
LAMS = [0, 1, 4, 16, 64]
METHODS = ["SFA", "ICA-slow", "PCA", "random"]


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def terms_per_seed(d, key):
    """(inclusion, allocation, exclusion) arrays over seeds, from stored values.

    Stored layout is [mean, sd, seed0, seed1, ...]; the per-seed tail is what makes
    a paired comparison possible, so the spread here is the real seed spread and
    not a propagated error bar."""
    a, b = d[key]["slow_half"], d[key].get("fast_half")
    if b is None:
        return None
    incl = np.array([clip01(v) for v in a["slow"][2:]])
    excl = np.array([clip01(1.0 - v) for v in a["fast"][2:]])
    alloc = np.array([clip01(v) for v in b["fast"][2:]])
    return incl, alloc, excl


def aggregates(incl, alloc, excl):
    prod = incl * alloc * excl
    ssum = (incl + alloc + excl) / 3.0
    eps = 1e-9
    harm = 3.0 / (1.0 / (incl + eps) + 1.0 / (alloc + eps) + 1.0 / (excl + eps))
    return prod, ssum, harm


def ms(x):
    return f"{x.mean():.3f}±{x.std(ddof=1) if len(x) > 1 else 0.0:.3f}"


def main():
    out = {}
    for label, stem in ROT:
        p = RUNS / f"{stem}.json"
        if not p.exists():
            print(f"[skip] {label}: {p.name} missing")
            continue
        d = json.loads(p.read_text())
        rows = {}
        for lam in LAMS:
            t = terms_per_seed(d, f"gate@lam{lam:g}")
            if t:
                rows[f"ours@lam{lam:g}"] = t
        for m in METHODS:
            if m in d:
                t = terms_per_seed(d, m)
                if t:
                    rows[m] = t
        if "ungated(block)" in d:
            t = terms_per_seed(d, "ungated(block)")
            if t:
                rows["mechanism-free"] = t

        out[label] = {}
        print(f"\n### {label}")
        print(f"| method | inclusion | allocation | exclusion | SEP (product) | mean | harmonic |")
        print(f"|---|---|---|---|---|---|---|")
        for name, (i, a, e) in rows.items():
            prod, ssum, harm = aggregates(i, a, e)
            out[label][name] = dict(
                inclusion=[float(i.mean()), float(i.std(ddof=1) if len(i) > 1 else 0)],
                allocation=[float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0)],
                exclusion=[float(e.mean()), float(e.std(ddof=1) if len(e) > 1 else 0)],
                sep=[float(prod.mean()), float(prod.std(ddof=1) if len(prod) > 1 else 0)],
                mean_agg=float(ssum.mean()), harmonic_agg=float(harm.mean()),
                inclusion_per_seed=i.tolist(), allocation_per_seed=a.tolist(),
                exclusion_per_seed=e.tolist(), sep_per_seed=prod.tolist())
            print(f"| {name} | {ms(i)} | {ms(a)} | {ms(e)} | {ms(prod)} | "
                  f"{ssum.mean():.3f} | {harm.mean():.3f} |")

        # where does our best lose to SFA? paired by seed, so the comparison is not
        # swamped by the across-seed variance
        if "SFA" in rows:
            best = max((k for k in rows if k.startswith("ours@")),
                       key=lambda k: aggregates(*rows[k])[0].mean())
            bi, ba, be = rows[best]
            si, sa, se = rows["SFA"]
            out[label]["_diagnosis"] = dict(
                best_ours=best,
                d_inclusion=float((bi - si).mean()), d_allocation=float((ba - sa).mean()),
                d_exclusion=float((be - se).mean()),
                d_sep=float((aggregates(bi, ba, be)[0] - aggregates(si, sa, se)[0]).mean()))
            g = out[label]["_diagnosis"]
            print(f"\n  {best} minus SFA (paired): "
                  f"incl {g['d_inclusion']:+.3f}  alloc {g['d_allocation']:+.3f}  "
                  f"excl {g['d_exclusion']:+.3f}  -> SEP {g['d_sep']:+.3f}")

    (RUNS / "term_breakdown.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> runs_v2/term_breakdown.json")


if __name__ == "__main__":
    main()
