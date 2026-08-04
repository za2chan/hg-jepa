"""Domain robustness (HANDOFF step 4): does z_slow lose less under subject shift?

Three DISJOINT window sets per partition:
  A_fit   24 subjects, EARLY windows — the ONLY windows the encoder sees, and
          the only windows the probe is fit on.
  A_test  same 24 subjects, LATE windows — in-domain reference (seen subjects,
          unseen data).
  B        6 held-out subjects, all windows — out of domain.

Metric: domain drop Δ = F1(A_test) − F1(B), per block. Claim: Δ_slow < Δ_full.
CAVEAT (measured, not hypothetical): HAPT recordings follow an ordered activity
protocol, so A_test's LATE windows are ~87% walking with zero STANDING support
while all-of-B covers the whole protocol. A_test is therefore both a time shift
and a label shift, and macro-F1 is averaged over a different class set on each
side — Δ can come out negative for that reason alone. `delta_late` repeats the
comparison against B's late windows only (time-matched, subject shift isolated);
`n_cls` records the class support of each eval set. Read both.
B is only 6 subjects, so the OOD estimate is noisy: B rotates over disjoint
subject folds and we report error bars over partitions as well as over seeds.

Usage: python3 domain_shift.py [n_partitions] [n_seeds] [steps]
Writes runs_v2/domain_shift_hapt.json (…_smoke.json for short runs).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np

from probes import C_MIN, encode_all, multi_position
from train_real import train_real

ROOT = pathlib.Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/hapt_v2.npz"
HAPT = dict(n_ax=3, tau=40.0, w=12, dmin=12, dmax=128, min_context=16)
# Both stems are carried (CLAUDE.md §2, amended 2026-08-04). CLI: argv[4].
STEM = "nce+ema"
BLOCKS = ("z_slow", "z_full")
N_B, FIT_FRAC, STEPS = 6, 0.7, 2500


def split(subj, part, n_b=N_B, fit_frac=FIT_FRAC, seed=0):
    """A/B subject partition (rotation `part`) + time split inside A.

    HAPT windows are cut with stride L/2, so window i overlaps i±1: the window
    AT the fit/test boundary shares samples with both sides and is dropped.
    Global index order is time order within a recording (prep/hapt.py), so
    idx[:k] is early and idx[k+1:] is late for that subject.
    Also returns B's LATE windows: HAPT's activity protocol is ordered, so
    A_test and all-of-B differ in label mix as well as in subject; the
    time-matched B slice is the diagnostic that separates the two.
    """
    subs = np.random.default_rng(seed).permutation(np.unique(subj))
    assert (part + 1) * n_b <= len(subs), "B folds must stay disjoint"
    b_sub = subs[part * n_b:(part + 1) * n_b]
    a_sub = np.setdiff1d(subs, b_sub)
    cut = lambda s: int(np.sum(subj == s) * fit_frac)
    early = lambda ss: np.sort(np.concatenate([np.flatnonzero(subj == s)[:cut(s)] for s in ss]))
    late = lambda ss: np.sort(np.concatenate([np.flatnonzero(subj == s)[cut(s) + 1:] for s in ss]))
    fit, test, ood = early(a_sub), late(a_sub), np.flatnonzero(np.isin(subj, b_sub))
    return fit, test, ood, late(b_sub), a_sub, b_sub


def check(W, subj, fit, test, ood, ood_late, a_sub, b_sub):
    """The three asserts the experiment's validity rests on."""
    # (c) subject sets disjoint and exhaustive
    assert not set(a_sub) & set(b_sub) and len(a_sub) + len(b_sub) == len(np.unique(subj))
    # (b) B subjects appear nowhere in training/probe-fitting
    assert not set(subj[fit]) & set(b_sub) and not set(subj[test]) & set(b_sub)
    assert set(subj[ood]) == set(b_sub) and not len(np.setdiff1d(ood_late, ood))
    # index disjointness
    for x, y in ((fit, test), (fit, ood), (test, ood)):
        assert not set(x.tolist()) & set(y.tolist())
    # (a) no SAMPLE overlap between A_fit and A_test. Stride = L/2 means every
    # window is exactly two half-blocks on a fixed grid, so two windows overlap
    # iff they share a half-block -> compare half-block byte hashes.
    half = lambda ix: {hash(W[i, :128].tobytes()) for i in ix} | \
                      {hash(W[i, 128:].tobytes()) for i in ix}
    shared = half(fit) & half(test)
    assert not shared, f"{len(shared)} overlapping half-blocks between A_fit and A_test"
    return dict(n_fit=len(fit), n_test=len(test), n_ood=len(ood),
                n_ood_late=len(ood_late), a_sub=a_sub.tolist(), b_sub=b_sub.tolist())


def run(part, seed, steps):
    d = np.load(NPZ)
    fit, test, ood, ood_late, a_sub, b_sub = split(d["subj"], part)
    meta = check(d["W"], d["subj"], fit, test, ood, ood_late, a_sub, b_sub)
    lk, te = STEM.split("+")
    res = train_real(str(NPZ), seed=seed, steps=steps, train_idx=fit,
                     loss_kind=lk, target_enc=te, log_every=10 ** 9, **HAPT)
    assert np.array_equal(res["tr"], fit), "encoder trained on something other than A_fit"
    Z = encode_all(res["enc"], res["Wt"])
    # macro-F1 averages over the classes PRESENT in each eval set, so the sets are
    # only comparable if their class support matches — record it, do not assume it.
    n_cls = lambda ix: int(np.isin(np.arange(1, 7), res["lab"][ix][:, C_MIN:]).sum())
    out = dict(part=part, seed=seed, n_cls=[n_cls(test), n_cls(ood), n_cls(ood_late)], **meta)
    for b in BLOCKS:
        # one probe fit (same train rows + seed -> same subsample) scored on each set
        p_ = {k: multi_position(Z, res["lab"], res["fast"], fit, ix, block=b, c_min=C_MIN)
              for k, ix in (("in", test), ("ood", ood), ("ood_late", ood_late))}
        f1 = {k: v["slow_kept_f1"] for k, v in p_.items()}
        out[b] = dict(in_f1=f1["in"], ood_f1=f1["ood"], delta=f1["in"] - f1["ood"],
                      ood_late_f1=f1["ood_late"], delta_late=f1["in"] - f1["ood_late"],
                      in_leak=p_["in"]["leak_r2"], ood_leak=p_["ood"]["leak_r2"],
                      chance_f1=p_["in"]["chance_f1"], rankme=p_["in"]["rankme"])
    return out


def summarize(cells, n_part):
    ms = lambda v: (float(np.mean(v)), float(np.std(v)))
    agg = {}
    for b in BLOCKS:
        agg[b] = {k: ms([c[b][k] for c in cells])
                  for k in ("in_f1", "ood_f1", "delta", "ood_late_f1", "delta_late")}
        for k in ("delta", "delta_late"):                   # error bar over partitions
            agg[b][k + "_over_partitions"] = ms(
                [np.mean([c[b][k] for c in cells if c["part"] == p]) for p in range(n_part)])
    for k in ("delta", "delta_late"):
        paired = [c["z_full"][k] - c["z_slow"][k] for c in cells]
        agg["claim_" + k] = dict(full_minus_slow=ms(paired), n_runs=len(paired),
                                 n_supporting=int(np.sum(np.array(paired) > 0)))
    return agg


if __name__ == "__main__":
    n_part = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seeds = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    if len(sys.argv) > 4:
        globals()['STEM'] = sys.argv[4]
    cells = []
    for p in range(n_part):
        for s in seeds:
            print(f"=== partition {p} seed {s} ({STEM}, {steps} steps) ===", flush=True)
            cells.append(run(p, s, steps))
            c = cells[-1]
            print(f"  n: fit {c['n_fit']} / A_test {c['n_test']} / B {c['n_ood']} "
                  f"/ B_late {c['n_ood_late']} | classes present {c['n_cls']} "
                  f"| B subjects {c['b_sub']}", flush=True)
    agg = summarize(cells, n_part)
    out = dict(config=dict(stem=STEM, steps=steps, seeds=seeds, n_partitions=n_part,
                           n_b=N_B, fit_frac=FIT_FRAC, c_min=C_MIN, **HAPT),
               cells=cells, agg=agg)
    path = ROOT / ("runs_v2/domain_shift_hapt.json" if steps == STEPS
                   else "runs_v2/domain_shift_hapt_smoke.json")
    json.dump(out, open(path, "w"), indent=2)

    print("\n=== domain drop  Δ = F1(A_test) − F1(B) | late = time-matched B ===")
    print(f"{'part':>4} {'seed':>4} | {'sl_in':>6} {'sl_ood':>6} {'Δ_slow':>7} {'Δlate':>6} |"
          f" {'fu_in':>6} {'fu_ood':>6} {'Δ_full':>7} {'Δlate':>6}")
    for c in cells:
        s, f = c["z_slow"], c["z_full"]
        print(f"{c['part']:>4} {c['seed']:>4} | {s['in_f1']:6.3f} {s['ood_f1']:6.3f} "
              f"{s['delta']:+7.3f} {s['delta_late']:+6.3f} | {f['in_f1']:6.3f} "
              f"{f['ood_f1']:6.3f} {f['delta']:+7.3f} {f['delta_late']:+6.3f}")
    for b in BLOCKS:
        a = agg[b]
        print(f"{b:7s} in {a['in_f1'][0]:.3f}±{a['in_f1'][1]:.3f}  "
              f"ood {a['ood_f1'][0]:.3f}±{a['ood_f1'][1]:.3f}  "
              f"Δ {a['delta'][0]:+.3f}±{a['delta'][1]:.3f} (runs) "
              f"±{a['delta_over_partitions'][1]:.3f} (partitions)  |  "
              f"Δlate {a['delta_late'][0]:+.3f}±{a['delta_late'][1]:.3f}")
    for k in ("delta", "delta_late"):
        cl = agg["claim_" + k]
        print(f"claim Δ_slow < Δ_full ({k}): Δ_full−Δ_slow = {cl['full_minus_slow'][0]:+.3f}"
              f"±{cl['full_minus_slow'][1]:.3f}, holds in "
              f"{cl['n_supporting']}/{cl['n_runs']} runs")
    print("saved", path)
