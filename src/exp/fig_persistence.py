"""Introduction figure: spectral slowness is not factor persistence.

Everything is measured from the synthetic generator's ground truth, so the
figure makes the distinction the paper rests on without asserting it in prose.

The generator (datagen.py, gap=0.03) is built so the two axes point OPPOSITE ways:
  - the PERSISTENT factor s (dwell 3000) sets only the carrier FREQUENCY
    (0.097 / 0.100 / 0.103 cycles/step; amplitudes are identical), so it lives
    inside a component that oscillates every ~10 steps;
  - the TRANSIENT factor u (OU lifetime 50) enters additively at 0.2*u, which is
    slow compared with the carrier, so it is what a low-pass reading returns.

python3 src/exp/fig_persistence.py -> runs_v2/fig_persistence.{png,json}
"""
import json, pathlib, sys
import numpy as np
from scipy import signal as sps
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hglp"))
from datagen import generate, DWELL, U_TAU, _freqs                    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
GAP, N, SEED = 0.03, 400_000, 0
CARRIER = 0.100                       # cycles/step
LOWPASS = 0.02                        # well below the carrier, above 1/U_TAU
C_SLOW, C_FAST, C_SIG = "#1a5fb4", "#e8a33d", "#5b5b5b"


def lowpass(x, cut):
    b, a = sps.butter(4, cut / 0.5, btype="low")
    return sps.filtfilt(b, a, x)


def inst_freq(x, lo=0.06, hi=0.15, smooth=201):
    """Instantaneous frequency of the carrier, in cycles/step."""
    b, a = sps.butter(4, [lo / 0.5, hi / 0.5], btype="band")
    ph = np.unwrap(np.angle(sps.hilbert(sps.filtfilt(b, a, x))))
    f = np.gradient(ph) / (2 * np.pi)
    return sps.savgol_filter(f, smooth, 2)


def r2(view, target):
    v = view.reshape(-1, 1)
    return float(max(0.0, LinearRegression().fit(v, target).score(v, target)))


def acf(v, max_lag, step):
    v = np.asarray(v, float) - np.mean(v)
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag].real
    return (ac / ac[0])[::step]


def main():
    x, s, u, _ = generate(N, seed=SEED, gap=GAP)
    fr = _freqs(GAP)
    fs = fr[s]                                   # what s physically controls
    warm = 2000                                  # drop filter edge effects
    sl = slice(warm, N - warm)

    views = {"low-pass ($f<0.02$)": lowpass(x, LOWPASS)[sl],
             "carrier frequency": inst_freq(x)[sl]}
    targets = {"persistent factor $s$": fs[sl], "transient factor $u$": u[sl]}
    R = {vn: {tn: r2(v, t) for tn, t in targets.items()} for vn, v in views.items()}

    res = dict(config=dict(gap=GAP, n=N, seed=SEED, carrier=CARRIER,
                           lowpass_cut=LOWPASS, dwell=DWELL, u_tau=U_TAU,
                           regime_freqs=fr.tolist()),
               r2=R,
               acf_halflife=dict(
                   s=float(np.argmax(acf(fs[sl], 20000, 1) < np.exp(-1))),
                   u=float(np.argmax(acf(u[sl], 20000, 1) < np.exp(-1)))))
    OUT.mkdir(exist_ok=True)
    (OUT / "fig_persistence.json").write_text(json.dumps(res, indent=1))

    BOX = dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", lw=0.8, alpha=0.95)
    fig = plt.figure(figsize=(15.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.45, 1.0], wspace=0.52)

    # (a) the signal and the two factors that made it -------------------------
    a0 = fig.add_subplot(gs[0])
    w = slice(warm, warm + 6000)
    t = np.arange(6000)
    a0.plot(t, x[w], lw=0.4, color=C_SIG)
    a0.plot(t, 11 + (fs[w] - CARRIER) / GAP / CARRIER * 1.1, lw=2.2, color=C_SLOW)
    a0.plot(t, 5.5 + u[w] * 0.55, lw=0.9, color=C_FAST)
    a0.set_ylim(-6.4, 13.6)
    a0.set_yticks([11, 5.5, 0])
    a0.set_yticklabels([f"$s$  persistent\n(dwell {DWELL})",
                        f"$u$  transient\n(lifetime {U_TAU})",
                        "signal $x$\n(a sine wave)"], fontsize=8.5)
    a0.set_xlabel("time (steps)")
    a0.set_title("(a) one signal, two generating factors", fontsize=10, pad=10)
    a0.spines[["top", "right"]].set_visible(False)
    # What each factor does to x, in words. The zoomed carrier used to live here and
    # showed only that the signal oscillates, which the trace above already shows.
    # The generator equation belongs in the Data section, not in the teaser figure.
    a0.add_patch(plt.Rectangle((60, -5.7), 5880, 3.0, facecolor="#fafafa",
                               edgecolor="#cccccc", lw=0.8, zorder=3))
    a0.text(300, -3.3, "$s$ changes the frequency of $x$",
            fontsize=8.5, va="top", color=C_SLOW, zorder=4)
    a0.text(300, -4.6, "$u$ changes the frequency of $x$, and its baseline",
            fontsize=8.5, va="top", color=C_FAST, zorder=4)

    # (b) which view of the signal shows which factor --------------------------
    # u modulates the carrier frequency by FREQ_MOD=0.05, which is LARGER than the
    # 0.03 regime gap, so a lightly smoothed frequency reading is mostly u. Draw it
    # at an integration length where s dominates (panel (c) is that trade-off), and
    # print both R2 on each trace so the reader can check what is in it.
    SM = 1001
    a1 = fig.add_subplot(gs[1])
    zc = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    v2 = slice(warm, warm + 20000)
    tt = np.arange(20000)
    lp_all, ifr_all = lowpass(x, LOWPASS), inst_freq(x, smooth=SM)
    r_lp = (r2(lp_all[sl], fs[sl]), r2(lp_all[sl], u[sl]))
    r_if = (r2(ifr_all[sl], fs[sl]), r2(ifr_all[sl], u[sl]))
    res["panel_b"] = dict(smooth=SM, lowpass_r2_s=r_lp[0], lowpass_r2_u=r_lp[1],
                          instfreq_r2_s=r_if[0], instfreq_r2_u=r_if[1])

    TOP = 10.0
    a1.plot(tt, TOP + zc(u[v2]), lw=2.6, color=C_FAST, alpha=.55,
            label="true factor", solid_capstyle="round")
    a1.plot(tt, TOP + zc(lp_all[v2]), lw=0.7, color="k", alpha=.9,
            label="read from $x$")
    a1.plot(tt, zc(fs[v2]), lw=2.6, color=C_SLOW, alpha=.55, solid_capstyle="round")
    a1.plot(tt, zc(ifr_all[v2]), lw=0.9, color="k", alpha=.9)
    a1.legend(fontsize=7.5, loc="lower right", frameon=False, ncol=2)
    # R^2 boxes sit in the empty band between/above the traces, never on them
    a1.text(19600, TOP + 5.6, f"$R^2$    $s$ {r_lp[0]:.2f}    $u$ {r_lp[1]:.2f}",
            ha="right", va="center", fontsize=8.5, color="#333", bbox=BOX)
    a1.text(19600, 5.0, f"$R^2$    $s$ {r_if[0]:.2f}    $u$ {r_if[1]:.2f}",
            ha="right", va="center", fontsize=8.5, color="#333", bbox=BOX)
    a1.set_ylim(-4.6, 17.5)
    a1.set_yticks([TOP, 0])
    a1.set_yticklabels(["low-pass of $x$\n$\\rightarrow$ tracks $u$",
                        f"frequency of $x$\n(over {SM} steps)\n$\\rightarrow$ tracks $s$"],
                       fontsize=8.5)
    a1.set_xlabel("time (steps)")
    a1.set_title("(b) the SLOW part of the signal carries the TRANSIENT factor",
                 fontsize=9.5, pad=10)
    a1.spines[["top", "right"]].set_visible(False)

    # (c) persistence = how long you must integrate before the factor appears --
    # u modulates the carrier too (FREQ_MOD 0.05 > the 0.03 regime gap), so a
    # pointwise frequency reading is dominated by u. s only emerges once the
    # integration window exceeds u's lifetime -- which is what tau names.
    a2 = fig.add_subplot(gs[2])
    wins = [11, 31, 101, 301, 1001, 3001, 9001]
    curve = {"s": [], "u": []}
    for W in wins:
        f = inst_freq(x, smooth=W)[sl]
        curve["s"].append(r2(f, fs[sl])); curve["u"].append(r2(f, u[sl]))
    res["integration_curve"] = dict(windows=wins, **curve)
    (OUT / "fig_persistence.json").write_text(json.dumps(res, indent=1))

    a2.plot(wins, curve["s"], "o-", color=C_SLOW, lw=2, label="persistent $s$")
    a2.plot(wins, curve["u"], "s--", color=C_FAST, lw=2, label="transient $u$")
    a2.axvline(U_TAU, color="#c01c28", ls=":", lw=1.2)
    a2.text(U_TAU * 1.2, 0.10, f"$u$ lifetime\n{U_TAU}", color="#c01c28", fontsize=8,
            va="bottom", bbox=BOX)
    a2.set_xscale("log")
    a2.set_xlabel("integration window (steps)")
    a2.set_ylabel("$R^2$ from the frequency of $x$", fontsize=8.5)
    a2.set_ylim(0, 1.02)
    a2.legend(fontsize=8, frameon=False)
    a2.set_title("(c) $s$ appears only past $u$'s lifetime", fontsize=10, pad=10)
    a2.grid(alpha=.25, lw=.6); a2.set_axisbelow(True)
    a2.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT / "fig_persistence.png", dpi=200, bbox_inches="tight")
    print(json.dumps(res["r2"], indent=1))
    print("wrote runs_v2/fig_persistence.png + .json")


if __name__ == "__main__":
    main()
