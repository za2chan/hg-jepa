"""Why the block is called `z_slow`: slowness is real, but it lives one level up.

Fig. 1 shows the persistent factor is absent from the signal's slow band. That
invites the obvious objection -- then why name the block "slow"? Because slowness
is a property of a FUNCTION of the signal, not of the signal. Pick the right
statistic and the persistent factor is genuinely slowly varying; the learned
block is such a statistic, and its trajectory is measurably slow.

This is also why Slow Feature Analysis is a real competitor rather than a foil:
SFA searches for slow functions too, inside a fixed function class. The
difference is who chooses the class.

python3 src/exp/fig_why_slow.py -> runs_v2/fig_why_slow.{png,json}
"""
import json, pathlib, sys
import numpy as np
import torch
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HGLP = pathlib.Path(__file__).resolve().parents[1] / "hglp"
sys.path.insert(0, str(HGLP))
from datagen import generate, DWELL, U_TAU                          # noqa: E402
from model import P, L, D_SLOW, D_Z                                 # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
GAP, N, SEED, STEPS = 0.03, 400_000, 0, 2500
LOWPASS, SM = 0.02, 1001
N_WIN, MAX_LAG_STEPS = 256, 2000
C_SLOW, C_MIX, C_SIG, C_LP = "#1a5fb4", "#e8a33d", "#5b5b5b", "#3a9e6e"
INK, MUTE = "#222222", "#9aa0a6"


def acf(v, max_lag):
    v = np.asarray(v, float) - np.mean(v)
    f = np.fft.rfft(v, 2 * len(v))
    a = np.fft.irfft(f * np.conj(f))[:max_lag].real
    return a / a[0]


def acf_block(Z, mode="pc1"):
    """ACF of a block's trajectory, averaged over windows.

    mode='pc1' follows the block's dominant direction, which is what a linear
    probe reads; mode='mean' averages every dimension equally and is dominated by
    the block's near-noise directions.
    """
    n, Lw, D = Z.shape
    if mode == "pc1":
        F = Z.reshape(-1, D)
        F = F - F.mean(0)
        v = np.linalg.svd(F, full_matrices=False)[2][0]
        proj = Z @ v
        return np.mean([acf(proj[i], Lw // 2) for i in range(n)], 0)
    out = np.zeros(Lw // 2)
    for d in range(D):
        out += np.mean([acf(Z[i, :, d], Lw // 2) for i in range(n)], 0)
    return out / D


def halflife(a, lags):
    b = np.flatnonzero(a < np.exp(-1))
    return float(lags[b[0]]) if len(b) else float(lags[-1])


def main():
    x, s, u, _ = generate(N, seed=SEED, gap=GAP)

    # --- signal-derived views -------------------------------------------------
    b, a_ = sps.butter(4, LOWPASS / 0.5, btype="low")
    lp = sps.filtfilt(b, a_, x)
    bb, aa = sps.butter(4, [0.06 / 0.5, 0.15 / 0.5], btype="band")
    ph = np.unwrap(np.angle(sps.hilbert(sps.filtfilt(bb, aa, x))))
    cf = sps.savgol_filter(np.gradient(ph) / (2 * np.pi), SM, 2)

    w = slice(2000, N - 2000)
    lag_s = np.arange(MAX_LAG_STEPS)
    curves = {"raw signal $x$": acf(x[w], MAX_LAG_STEPS),
              "low-pass of $x$": acf(lp[w], MAX_LAG_STEPS),
              f"carrier frequency ({SM} steps)": acf(cf[w], MAX_LAG_STEPS)}

    # --- learned blocks -------------------------------------------------------
    from train import train, DEV
    r = train(loss_kind="nce", target_enc="ema", seed=SEED, gate=True, xcov=True,
              gap=GAP, steps=STEPS, log_every=10 ** 9)
    enc = r["enc"].eval()
    rng = np.random.default_rng(0)
    st = rng.integers(2000, N - L * P - 2000, N_WIN)
    xb = torch.from_numpy(np.stack([x[i:i + L * P].reshape(L, P) for i in st])).to(DEV)
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 64]) for i in range(0, len(xb), 64)]).cpu().numpy()
    lag_p = np.arange(L // 2) * P                       # patches -> steps
    blocks = {"$z_{per}$ (16 dims)": acf_block(Z[:, :, :D_SLOW]),
              "$z_{mix}$ (48 dims)": acf_block(Z[:, :, D_SLOW:])}
    blocks_mean = {k: acf_block(Z[:, :, sl_], "mean") for k, sl_ in
                   [("z_slow_meandim", slice(0, D_SLOW)), ("z_mix_meandim", slice(D_SLOW, D_Z))]}

    res = dict(config=dict(gap=GAP, n=N, seed=SEED, steps=STEPS, n_win=N_WIN,
                           smooth=SM, dwell=DWELL, u_tau=U_TAU, patch=P),
               halflife_steps={**{k: halflife(v, lag_s) for k, v in curves.items()},
                               **{k: halflife(v, lag_p) for k, v in blocks.items()},
                               **{k: halflife(v, lag_p) for k, v in blocks_mean.items()}})
    OUT.mkdir(exist_ok=True)
    (OUT / "fig_why_slow.json").write_text(json.dumps(res, indent=1))

    # One panel only. The signal-function comparison duplicates Fig 1(c), so the
    # figure keeps the single claim that Fig 1 cannot make: the LEARNED block is
    # the slow one.
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    hl = {}
    for (k, v), c in zip(blocks.items(), [C_SLOW, C_MIX]):
        ax.plot(lag_p, v, lw=2.4, color=c, label=k)
        hl[k] = halflife(v, lag_p)
        ax.plot([hl[k]], [np.exp(-1)], "o", color=c, ms=7, zorder=5)
    ax.axhline(np.exp(-1), color="#999", ls="--", lw=1)
    ax.text(560, np.exp(-1) + 0.035, "$1/e$", fontsize=8.5, color="#777", ha="right")
    ks = list(blocks)
    for k, c in zip(ks, [C_SLOW, C_MIX]):
        ax.plot([hl[k]] * 2, [0, np.exp(-1)], ls=":", lw=1.2, color=c)
        ax.text(hl[k], -0.085, f"{hl[k]:.0f}", fontsize=9, ha="center", color=c)
    ax.text(300, 0.80,
            f"decorrelation time\n{hl[ks[0]]:.0f} vs {hl[ks[1]]:.0f} steps "
            f"({hl[ks[0]]/max(hl[ks[1]],1e-9):.0f}$\\times$)",
            fontsize=10, ha="center", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
    ax.set_xlim(0, 600); ax.set_ylim(-0.13, 1.03)
    ax.set_xlabel("lag (steps)"); ax.set_ylabel("autocorrelation of the block")
    ax.spines["bottom"].set_position(("data", 0))
    ax.legend(fontsize=9.5, frameon=False)
    ax.set_title("the persistent block is the slowly-varying one", fontsize=10.5, pad=8)
    ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_why_slow.png", dpi=220, bbox_inches="tight")
    print(json.dumps(res["halflife_steps"], indent=1))
    print("wrote runs_v2/fig_why_slow.png + .json")


if __name__ == "__main__":
    main()
