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
# z_slow vs z_full OF THE SAME GATED MODEL only answers "which readout of our
# model is more robust", not "is our model more robust" — the ungated control is
# the baseline a method claim needs (same issue the gate x xcov ablation fixed).
# z_fast is carried so the shift can be attributed: if the fast structure shifts
# as much as the slow one, excluding it cannot buy robustness.
BLOCKS = ("z_slow", "z_fast", "z_full")
UNGATED_BLOCKS = ("z_slow", "z_full")          # 'z_slow' here = an arbitrary first-16
N_B, FIT_FRAC, STEPS = 6, 0.7, 2500
N_BLOCKS = 10          # contiguous blocks per subject for SPLIT_MODE="block"
SPLIT_MODE = "block"   # "block" (default) | "time" (the original, kept for comparison)


def _blocks(idx, fit_frac, rng, n_blocks):
    """Cut one subject's time-ordered window indices into n_blocks contiguous
    blocks and assign whole blocks to fit/test at random.

    Why not a plain random split: windows are cut with stride L/2, so window i
    shares half its samples with i±1 and a per-window random split leaks
    immediately. Why not the time split it replaces: HAPT is recorded in a fixed
    activity-protocol ORDER, so "first 70% vs last 30%" makes the two sides
    differ in label mix as well as in time -- the late third is 85% dynamic and
    contains no STANDING. Block assignment samples both sides from across the
    whole protocol, so the label mixes match in expectation and the only thing
    left between A_test and B is the subject.

    The FIRST window of every block is dropped: it is the only one that can
    overlap the last window of the preceding block, which may be on the other
    side of the split. (The half-block hash assert in check() verifies this.)
    """
    out = ([], [])
    for j, blk in enumerate(np.array_split(idx, n_blocks)):
        if len(blk) < 2:
            continue
        out[int(rng.random() >= fit_frac)].append(blk[1:])   # drop the seam window
    return [np.concatenate(v) if v else np.empty(0, int) for v in out]


def split(subj, part, n_b=N_B, fit_frac=FIT_FRAC, seed=0, mode=None,
          n_blocks=N_BLOCKS):
    """A/B subject partition (rotation `part`) + a within-A split of windows.

    mode="block" (default): whole contiguous blocks assigned at random, so A_fit
      and A_test share the activity mix -- see _blocks().
    mode="time": the original first-70%/last-30% cut, kept so the two designs can
      be compared rather than silently swapped.

    Global index order is time order within a recording (prep/hapt.py).
    Also returns B's LATE windows: under mode="time" that is the diagnostic that
    separates the label shift from the subject shift; under mode="block" it is
    only reported for continuity.
    """
    mode = mode or SPLIT_MODE
    subs = np.random.default_rng(seed).permutation(np.unique(subj))
    assert (part + 1) * n_b <= len(subs), "B folds must stay disjoint"
    b_sub = subs[part * n_b:(part + 1) * n_b]
    a_sub = np.setdiff1d(subs, b_sub)
    cut = lambda s: int(np.sum(subj == s) * fit_frac)
    late = lambda ss: np.sort(np.concatenate([np.flatnonzero(subj == s)[cut(s) + 1:] for s in ss]))
    if mode == "block":
        rng = np.random.default_rng(1000 + part)     # same split for every seed
        pairs = [_blocks(np.flatnonzero(subj == s), fit_frac, rng, n_blocks) for s in a_sub]
        fit = np.sort(np.concatenate([p[0] for p in pairs]))
        test = np.sort(np.concatenate([p[1] for p in pairs]))
    elif mode == "time":
        early = lambda ss: np.sort(np.concatenate(
            [np.flatnonzero(subj == s)[:cut(s)] for s in ss]))
        fit, test = early(a_sub), late(a_sub)
    else:
        raise ValueError(mode)
    ood = np.flatnonzero(np.isin(subj, b_sub))
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


N_DRAW = 4         # subject-matched draws averaged into delta_matched
# Only the blocks that appear in a claim get the matched treatment: it costs
# n_draw x 2 extra probe fits per block, and z_fast is never a claim's subject.
MATCHED_BLOCKS = ("z_slow", "z_full")


def _matched(Z, r, fit, test, ood, subj, block, n_draw=N_DRAW, seed=0):
    """Subject- and size-matched in-domain vs out-of-domain score.

    Each draw takes |B| of A's subjects and their A_test windows, then an equal
    number of B windows. Averaging over draws removes the eval-set breadth effect
    that makes the raw drop negative without saying anything about domain shift.
    """
    rng = np.random.default_rng(seed)
    a_subs, b_subs = np.unique(subj[test]), np.unique(subj[ood])
    f1 = lambda ix: multi_position(Z, r["lab"], r["fast"], fit, ix,
                                   block=block, c_min=C_MIN)["slow_kept_f1"]
    ins, oods = [], []
    for _ in range(n_draw):
        pick = rng.choice(a_subs, len(b_subs), replace=False)
        ia = test[np.isin(subj[test], pick)]
        ib = rng.choice(ood, min(len(ia), len(ood)), replace=False)
        ins.append(f1(ia)); oods.append(f1(ib))
    return dict(in_=float(np.mean(ins)), ood=float(np.mean(oods)))


def run(part, seed, steps):
    d = np.load(NPZ)
    fit, test, ood, ood_late, a_sub, b_sub = split(d["subj"], part)
    meta = check(d["W"], d["subj"], fit, test, ood, ood_late, a_sub, b_sub)
    lk, te = STEM.split("+")
    kw = dict(seed=seed, steps=steps, train_idx=fit, loss_kind=lk, target_enc=te,
              log_every=10 ** 9, **HAPT)
    res = train_real(str(NPZ), **kw)
    assert np.array_equal(res["tr"], fit), "encoder trained on something other than A_fit"
    ung = train_real(str(NPZ), gate=False, xcov=False, **kw)     # baseline for the claim
    # macro-F1 averages over the classes PRESENT in each eval set, so the sets are
    # only comparable if their class support matches — record it, do not assume it.
    n_cls = lambda ix: int(np.isin(np.arange(1, 7), res["lab"][ix][:, C_MIN:]).sum())

    def dist(ix):
        v = res["lab"][ix][:, C_MIN:]
        v = v[(v >= 1) & (v <= 6)]
        return np.bincount(v, minlength=7)[1:7] / max(len(v), 1)

    # The point of the block split is that A_test and B should now have the SAME
    # activity mix, so half the L1 distance between their label distributions (=
    # total variation) is the number that says whether it worked. Under the time
    # split it was large by construction: the late third is ~85% dynamic and has
    # no STANDING at all.
    d_test, d_ood = dist(test), dist(ood)
    out = dict(part=part, seed=seed, n_cls=[n_cls(test), n_cls(ood), n_cls(ood_late)],
               lab_dist=dict(fit=dist(fit).round(4).tolist(), test=d_test.round(4).tolist(),
                             ood=d_ood.round(4).tolist(),
                             ood_late=dist(ood_late).round(4).tolist()),
               label_tv=float(np.abs(d_test - d_ood).sum() / 2), **meta)
    subj = d["subj"]
    for r, blocks, pre in ((res, BLOCKS, ""), (ung, UNGATED_BLOCKS, "ungated_")):
        Z = encode_all(r["enc"], r["Wt"])
        for b in blocks:
            # one probe fit (same train rows + seed -> same subsample) scored on each set
            p_ = {k: multi_position(Z, r["lab"], r["fast"], fit, ix, block=b, c_min=C_MIN)
                  for k, ix in (("in", test), ("ood", ood), ("ood_late", ood_late))}
            f1 = {k: v["slow_kept_f1"] for k, v in p_.items()}
            # A_test spans 24 subjects and B spans 6. One linear probe covering 24
            # people scores lower than the same probe covering 6, so the raw drop
            # comes out NEGATIVE (B beats A_test) even with the label mix matched.
            # Fix: draw 6 of the 24 A subjects and an equal number of B windows, so
            # both sides have the same subject count AND the same window count.
            # (Scoring each subject alone does NOT work: A_test averages 15 windows
            # per subject covering only 4.25 of the 6 classes, so macro-F1 over the
            # full label set charges them for classes that are simply absent.)
            per = (_matched(Z, r, fit, test, ood, subj, b)
                   if b in MATCHED_BLOCKS else dict(in_=float("nan"), ood=float("nan")))
            # the fast proxy's OWN domain drop: the test of whether subject shift
            # moves the fast structure as much as the slow one (an interpretation
            # that was previously asserted, not measured)
            out[pre + b] = dict(
                in_f1=f1["in"], ood_f1=f1["ood"], delta=f1["in"] - f1["ood"],
                ood_late_f1=f1["ood_late"], delta_late=f1["in"] - f1["ood_late"],
                in_matched_f1=per["in_"], ood_matched_f1=per["ood"],
                delta_matched=per["in_"] - per["ood"],
                in_leak=p_["in"]["leak_r2"], ood_leak=p_["ood"]["leak_r2"],
                delta_fast=p_["in"]["leak_r2"] - p_["ood"]["leak_r2"],
                chance_f1=p_["in"]["chance_f1"], rankme=p_["in"]["rankme"])
    return out


ALL_BLOCKS = list(BLOCKS) + ["ungated_" + b for b in UNGATED_BLOCKS]
# the reference each claim is measured against: same-model readout, and the real
# baseline (an ungated model's full embedding)
CLAIMS = (("vs_full", "z_full"), ("vs_ungated", "ungated_z_full"))


def summarize(cells, n_part):
    ms = lambda v: (float(np.mean(v)), float(np.std(v)))
    agg = {}
    for b in ALL_BLOCKS:
        agg[b] = {k: ms([c[b][k] for c in cells])
                  for k in ("in_f1", "ood_f1", "delta", "ood_late_f1", "delta_late",
                            "delta_fast", "in_matched_f1", "ood_matched_f1", "delta_matched")}
        for k in ("delta", "delta_late", "delta_matched"):                   # error bar over partitions
            agg[b][k + "_over_partitions"] = ms(
                [np.mean([c[b][k] for c in cells if c["part"] == p]) for p in range(n_part)])
    for k in ("delta", "delta_late", "delta_matched"):
        for name, ref in CLAIMS:
            paired = [c[ref][k] - c["z_slow"][k] for c in cells]
            agg[f"claim_{k}_{name}"] = dict(ref_minus_slow=ms(paired), n_runs=len(paired),
                                            n_supporting=int(np.sum(np.array(paired) > 0)))
    return agg


if __name__ == "__main__":
    n_part = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seeds = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    if len(sys.argv) > 4:
        globals()['STEM'] = sys.argv[4]
    if len(sys.argv) > 5:
        globals()['SPLIT_MODE'] = sys.argv[5]
    cells = []
    for p in range(n_part):
        for s in seeds:
            print(f"=== partition {p} seed {s} ({STEM}, {steps} steps) ===", flush=True)
            cells.append(run(p, s, steps))
            c = cells[-1]
            print(f"  n: fit {c['n_fit']} / A_test {c['n_test']} / B {c['n_ood']} "
                  f"/ B_late {c['n_ood_late']} | classes present {c['n_cls']} "
                  f"| label TV(A_test,B) {c['label_tv']:.3f} "
                  f"| B subjects {c['b_sub']}", flush=True)
    agg = summarize(cells, n_part)
    out = dict(config=dict(stem=STEM, steps=steps, seeds=seeds, n_partitions=n_part,
                           split_mode=SPLIT_MODE, n_blocks=N_BLOCKS,
                           n_b=N_B, fit_frac=FIT_FRAC, c_min=C_MIN, **HAPT),
               label_tv=float(np.mean([c["label_tv"] for c in cells])),
               cells=cells, agg=agg)
    path = ROOT / (f"runs_v2/domain_shift_hapt_{STEM}_{SPLIT_MODE}.json" if steps == STEPS
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
    for b in ALL_BLOCKS:
        a = agg[b]
        print(f"{b:16s} in {a['in_f1'][0]:.3f}±{a['in_f1'][1]:.3f}  "
              f"ood {a['ood_f1'][0]:.3f}±{a['ood_f1'][1]:.3f}  "
              f"Δ {a['delta'][0]:+.3f}±{a['delta'][1]:.3f} (runs) "
              f"±{a['delta_over_partitions'][1]:.3f} (partitions)  |  "
              f"Δlate {a['delta_late'][0]:+.3f}±{a['delta_late'][1]:.3f}  |  "
              f"Δfast {a['delta_fast'][0]:+.3f}  |  "
              f"matched in {a['in_matched_f1'][0]:.3f} ood {a['ood_matched_f1'][0]:.3f} "
              f"Δm {a['delta_matched'][0]:+.3f}±{a['delta_matched'][1]:.3f}")
    print("\n  Δfast = drop in the FAST proxy's R2 under the same shift. If it is as "
          "large as\n  Δ_slow, subject shift moves both structures and excluding the "
          "fast one cannot help.")
    for k in ("delta", "delta_late", "delta_matched"):
        for name, ref in CLAIMS:
            cl = agg[f"claim_{k}_{name}"]
            print(f"claim Δ_slow < Δ_{ref} ({k}): diff = {cl['ref_minus_slow'][0]:+.3f}"
                  f"±{cl['ref_minus_slow'][1]:.3f}, holds in "
                  f"{cl['n_supporting']}/{cl['n_runs']} runs")
    print("saved", path)
