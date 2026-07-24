"""Reviewer #4: linear probes may understate information. Nonlinear checks on
a saved model: (a) MLP probes of the fast factors from z_slow (nonlinear leak)
and z_fast (displaced-not-destroyed), leak-free contiguous split on a longer
eval series; (b) MINE lower bound on I(z_slow; u).

Usage: python3 nonlinear.py tag=nepa_g1_d1_s0_tau16_ds16_lam4
Writes runs/nl_<tag>.json.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor

from datagen import make_dataset
from train import Encoder, P, L, D_Z, DEV

D_SLOW = 16
N_EVAL = 2_000_000


def embed(enc, d, stride):
    x = d["x"]
    starts = np.arange(0, len(x) - L * P, stride)
    Z = []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack(
                [x[s:s + L * P].reshape(L, P) for s in bs])).to(DEV)
            Z.append(enc(xb)[:, -1].cpu().numpy())
    end = starts + L * P - 1
    return np.concatenate(Z), d["u"][end], np.stack(
        [d["sin_phi"][end], d["cos_phi"][end]], 1)


def mine(zs, y, iters=2000):
    # DV bound: I >= E[T(z,y)] - log E[e^T(z,y')]
    zs = torch.from_numpy(zs).float().to(DEV)
    y = torch.from_numpy(y).float().unsqueeze(-1).to(DEV)
    T = nn.Sequential(nn.Linear(zs.shape[1] + 1, 128), nn.ReLU(),
                      nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(DEV)
    opt = torch.optim.Adam(T.parameters(), lr=1e-3)
    ema = 1.0
    for _ in range(iters):
        perm = torch.randperm(len(y), device=DEV)
        j = T(torch.cat([zs, y], -1)).mean()
        m = T(torch.cat([zs, y[perm]], -1)).exp().mean()
        ema = 0.99 * ema + 0.01 * m.item()
        loss = -(j - m / ema)          # EMA-corrected DV gradient (Belghazi 2018)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        perm = torch.randperm(len(y), device=DEV)
        j = T(torch.cat([zs, y], -1)).mean()
        m = T(torch.cat([zs, y[perm]], -1)).exp().mean()
        return float(j - m.log())


def main(args):
    tag = args.get("tag", "nepa_g1_d1_s0_tau16_ds16_lam4")
    torch.manual_seed(0)
    ck = torch.load(f"runs/model_{tag}.pt", map_location=DEV)
    enc = Encoder().to(DEV); enc.load_state_dict(ck["enc"]); enc.eval()
    d = make_dataset(N_EVAL, seed=77)

    # (a) MLP probes, disjoint windows, contiguous split
    Z, yu, yphi = embed(enc, d, stride=L * P)
    ntr = len(Z) // 2
    res = {"tag": tag, "n_probe": int(len(Z))}
    for name, sl in [("z_slow", slice(0, D_SLOW)), ("z_fast", slice(D_SLOW, D_Z))]:
        for yn, y in [("u", yu), ("phase", yphi)]:
            m = MLPRegressor((64, 64), max_iter=800, random_state=0)
            res[f"mlp_{name}->{yn}_r2"] = float(
                m.fit(Z[:ntr, sl], y[:ntr]).score(Z[ntr:, sl], y[ntr:]))

    # (b) MINE I(z_slow; u), denser (strided) sampling for estimator stability
    Zs, yus, _ = embed(enc, d, stride=L * P // 4)
    res["mine_I(z_slow;u)_nats"] = mine(Zs[:, :D_SLOW], yus)
    res["mine_I(z_fast;u)_nats"] = mine(Zs[:, D_SLOW:], yus)
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs/nl_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main(dict(a.split("=") for a in sys.argv[1:]))
