"""HAPT EDA from RAW files (labels.txt + acc_exp*.txt). Observation only —
no pipeline import, recomputes label sequences directly. Prints structure,
saves figures to docs/eda/figs/. Run from repo root."""
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

RAW = "data/hapt/RawData"
FS = 50.0
PATCH, L, STRIDE = 4, 128, 64
WIN = PATCH * L                                   # 512 samples = 10.24 s
NAMES = {1: "WALK", 2: "UPSTAIRS", 3: "DOWNSTAIRS", 4: "SIT", 5: "STAND",
         6: "LAY", 7: "STAND_TO_SIT", 8: "SIT_TO_STAND", 9: "SIT_TO_LIE",
         10: "LIE_TO_SIT", 11: "STAND_TO_LIE", 12: "LIE_TO_STAND"}


def per_sample_labels(exp, n, labels):
    per = np.zeros(n, dtype=int)
    for _, _, a, s, e in labels[labels[:, 0] == exp]:
        per[s:e + 1] = a
    return per


def main():
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)
    accfs = sorted(glob.glob(f"{RAW}/acc_exp*.txt"))
    print(f"=== HAPT STEP 1: raw structure ===")
    print(f"acc files: {len(accfs)} | gyro files: {len(glob.glob(f'{RAW}/gyro_exp*.txt'))}")
    print(f"labels.txt rows: {len(labels)} | schema: [exp, user, act, start, end]"
          f" (sample indices, inclusive)")
    users = sorted({int(f.split('user')[1][:2]) for f in accfs})
    exps = sorted({int(f.split('exp')[1][:2]) for f in accfs})
    print(f"experiments: {len(exps)} | users: {len(users)} = {users}")

    durs, ranges, allper = [], [], []
    exp_of, user_of = [], []
    for accf in accfs:
        exp = int(accf.split("exp")[1][:2]); user = int(accf.split("user")[1][:2])
        acc = np.loadtxt(accf).astype(np.float32)
        durs.append(len(acc)); ranges.append((acc.min(), np.median(acc), acc.max()))
        allper.append(per_sample_labels(exp, len(acc), labels))
        exp_of.append(exp); user_of.append(user)
        if np.isnan(acc).any():
            print(f"  !! NaN in exp{exp}")
    durs = np.array(durs)
    print(f"per-file samples: min/med/max {durs.min()}/{int(np.median(durs))}/{durs.max()}"
          f" = {durs.min()/FS:.0f}/{np.median(durs)/FS:.0f}/{durs.max()/FS:.0f} s")
    print(f"acc value range (g units): global "
          f"[{min(r[0] for r in ranges):.2f}, {max(r[2] for r in ranges):.2f}]")

    print(f"\n=== HAPT STEP 2: label structure ===")
    # per-class segment durations (NOT pooled)
    seg_by_class = {c: [] for c in range(1, 13)}
    segs_per_subj = {u: 0 for u in users}
    void_frac = []
    transition_classes_present = set()
    for exp, per, user in zip(exp_of, allper, user_of):
        # segment via run-length on per
        change = np.flatnonzero(np.diff(per)) + 1
        bounds = np.concatenate([[0], change, [len(per)]])
        for i in range(len(bounds) - 1):
            c = per[bounds[i]]
            dur = bounds[i + 1] - bounds[i]
            if c == 0:
                continue
            seg_by_class[c].append(dur)
            if 1 <= c <= 6:
                segs_per_subj[user] += 1
            if c >= 7:
                transition_classes_present.add(c)
        void_frac.append(np.mean(per == 0))
    print(f"void (label 0) fraction of samples: mean {np.mean(void_frac):.3f}")
    print(f"postural-transition classes 7-12 PRESENT in labels.txt: "
          f"{sorted(transition_classes_present)}")
    print(f"prep keeps only classes 1-6 at window END (7-12 -> window skipped)\n")
    print(f"{'class':20s} {'n_seg':>6} {'dur_s min/med/max':>22} {'total_s':>9}")
    for c in range(1, 13):
        s = np.array(seg_by_class[c]) / FS
        if len(s) == 0:
            continue
        tag = "  (TRANSITION)" if c >= 7 else ""
        print(f"{c:2d} {NAMES[c]:16s} {len(s):6d}   {s.min():5.1f}/{np.median(s):5.1f}/{s.max():5.1f}"
              f"     {s.sum():8.0f}{tag}")
    seg16 = np.concatenate([np.array(seg_by_class[c]) for c in range(1, 7)]) / FS
    print(f"\nbasic-activity (1-6) segments: n={len(seg16)}, dur median {np.median(seg16):.1f} s, "
          f"<{WIN/FS:.1f}s(L128) {np.mean(seg16 < WIN/FS):.2f}, "
          f"<{2*WIN/FS:.1f}s(L256) {np.mean(seg16 < 2*WIN/FS):.2f}")
    spsu = np.array(list(segs_per_subj.values()))
    print(f"basic segments per subject: min/med/max {spsu.min()}/{int(np.median(spsu))}/{spsu.max()}")

    # activity ORDER example (first exp)
    p0 = allper[0]
    change = np.flatnonzero(np.diff(p0)) + 1
    seq = [p0[0]] + [p0[c] for c in change]
    seq = [s for s in seq]
    print(f"\nactivity sequence, exp{exp_of[0]:02d} (ordered, incl void/transition):")
    print("  " + " -> ".join(f"{s}:{NAMES.get(s,'VOID')}" if s else "VOID" for s in seq[:24]))

    # ---- STEP 3 figures ----
    fig_full(accfs[0], exp_of[0], labels)
    fig_windows(accfs[0], exp_of[0], labels)
    fig_class_dists(accfs, exp_of, labels)
    print("\nsaved: hapt_full_recording.png, hapt_window_transition.png, hapt_class_dists.png")

    json.dump({"seg_by_class_sec": {NAMES[c]: (np.array(seg_by_class[c]) / FS).tolist()
                                    for c in range(1, 13) if seg_by_class[c]},
               "basic_seg_median_s": float(np.median(seg16)),
               "void_frac": float(np.mean(void_frac)),
               "transition_classes_present": [int(c) for c in sorted(transition_classes_present)]},
              open("docs/eda/hapt_labels.json", "w"), indent=2)


def fig_full(accf, exp, labels):
    acc = np.loadtxt(accf).astype(np.float32)
    per = per_sample_labels(exp, len(acc), labels)
    t = np.arange(len(acc)) / FS
    fig, ax = plt.subplots(2, 1, figsize=(14, 5), height_ratios=[3, 1], sharex=True)
    for i, lbl in enumerate("xyz"):
        ax[0].plot(t, acc[:, i], lw=0.4, label=f"acc_{lbl}")
    ax[0].legend(loc="upper right", ncol=3, fontsize=8)
    ax[0].set_ylabel("acc (g)"); ax[0].set_title(f"HAPT exp{exp:02d} full recording, all axes + activity bands")
    cmap = plt.get_cmap("tab20")
    for c in range(1, 13):
        m = per == c
        if m.any():
            ax[1].fill_between(t, 0, 1, where=m, color=cmap((c - 1) / 12),
                               step="mid", label=NAMES[c] if c <= 6 else None)
    # mark segment boundaries
    for b in np.flatnonzero(np.diff(per)) + 1:
        ax[1].axvline(b / FS, color="k", lw=0.3, alpha=0.4)
    ax[1].set_ylim(0, 1); ax[1].set_yticks([]); ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("activity"); ax[1].legend(loc="upper right", ncol=6, fontsize=6)
    plt.tight_layout(); plt.savefig("docs/eda/figs/hapt_full_recording.png", dpi=110)
    plt.close()


def fig_windows(accf, exp, labels):
    acc = np.loadtxt(accf).astype(np.float32)
    per = per_sample_labels(exp, len(acc), labels)
    # find a window that spans a transition and one that does not
    span_start = clean_start = None
    for s in range(0, len(acc) - WIN, STRIDE * PATCH):
        end = s + WIN - 1
        if not 1 <= per[end] <= 6:
            continue
        spanlab = per[s:s + WIN]
        if (spanlab != per[end]).any() and span_start is None:
            span_start = s
        if (spanlab == per[end]).all() and clean_start is None:
            clean_start = s
        if span_start is not None and clean_start is not None:
            break
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for a, s0, title in [(ax[0], clean_start, "clean (single activity)"),
                         (ax[1], span_start, "contaminated (spans transition)")]:
        if s0 is None:
            continue
        w = acc[s0:s0 + WIN]; pl = per[s0:s0 + WIN]; t = np.arange(WIN) / FS
        for i, lbl in enumerate("xyz"):
            a.plot(t, w[:, i], lw=0.6, label=f"acc_{lbl}")
        cmap = plt.get_cmap("tab20")
        for c in np.unique(pl):
            a.fill_between(t, -3, 3, where=pl == c, alpha=0.12,
                           color=cmap((c - 1) / 12) if c else "gray",
                           label=(NAMES.get(c, "VOID") if c else "VOID"))
        a.set_title(f"{title}\nend-label = {NAMES[per[s0+WIN-1]]}")
        a.set_xlabel("time (s)"); a.legend(fontsize=6, loc="upper right")
    ax[0].set_ylabel("acc (g, raw)")
    plt.tight_layout(); plt.savefig("docs/eda/figs/hapt_window_transition.png", dpi=110)
    plt.close()


def fig_class_dists(accfs, exp_of, labels):
    mags = {c: [] for c in range(1, 7)}
    for accf, exp in zip(accfs, exp_of):
        acc = np.loadtxt(accf).astype(np.float32)
        per = per_sample_labels(exp, len(acc), labels)
        mag = np.linalg.norm(acc, axis=1)
        for c in range(1, 7):
            m = per == c
            if m.any():
                mags[c].append(mag[m])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    data = [np.concatenate(mags[c]) if mags[c] else np.array([0]) for c in range(1, 7)]
    ax[0].boxplot(data, labels=[NAMES[c] for c in range(1, 7)], showfliers=False)
    ax[0].set_ylabel("|acc| (g)"); ax[0].set_title("acc magnitude by activity class")
    ax[0].tick_params(axis="x", rotation=30)
    for c in range(1, 7):
        d = np.concatenate(mags[c]) if mags[c] else np.array([0])
        ax[1].hist(d, bins=80, histtype="step", density=True, label=NAMES[c])
    ax[1].set_xlabel("|acc| (g)"); ax[1].set_title("magnitude distribution by class")
    ax[1].legend(fontsize=7); ax[1].set_xlim(0, 2.5)
    plt.tight_layout(); plt.savefig("docs/eda/figs/hapt_class_dists.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
