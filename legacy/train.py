"""Synthetic pilot: horizon-gated latent prediction (HGLP) vs raw-signal
prediction (AR / next-token family), on two-timescale synthetic data.

Usage: python3 train.py loss=reg target=ema gate=1 xcov=0 seed=0
                        [tau=16|auto] [c=2.5] [dslow=16] [lam=4] [vfloor=1]
                        [regime_mode=freq|variance]
  loss=reg   : L2 to target embeddings (HGLP-Reg when target=ema)
  loss=nce   : InfoNCE vs in-batch negatives (HGLP-NCE when target=online)
  loss=ar    : predict future raw patches (next-token / reconstruction family)
  target=ema    : frozen EMA target encoder
  target=online : both-sided gradients, no EMA, no stop-grad (D2)
  gate=1     : predictor's access to z_fast decays for horizons beyond tau
  xcov=1     : squared cross-covariance penalty between blocks (L_xcov)
  vfloor=0   : disable the VICReg-style variance-floor term (D7 ablation)
  tau=auto   : tau = c * T_ac chosen by tac.estimate_tac (rule D1)
Deprecated aliases (one commit): mode=nepa -> loss=reg target=ema,
mode=cpc -> loss=nce target=ema, mode=ar -> loss=ar, dcor= -> xcov=.
Probe eval uses disjoint windows + a contiguous time split (leak-free).
Writes runs/<tag>.json with block-factor probe matrix + config + RankMe.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge

from datagen import make_dataset

P = 8                  # patch length (steps)
L = 256                # sequence length (patches) -> 2048-step window
D_MODEL, D_Z = 96, 64
OFFSETS = [1, 4, 16, 64, 128]  # horizons (patches) = 8..1024 steps
W = 4.0                        # gate softness (patches)
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(P, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, L, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(D_MODEL, D_Z)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):                      # x: (B, L, P)
        h = self.embed(x) + self.pos
        h = self.tf(h, mask=self.mask)
        return F.layer_norm(self.out(h), (D_Z,))


class Predictor(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.demb = nn.Embedding(len(OFFSETS), 16)
        self.net = nn.Sequential(nn.Linear(D_Z + 16, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, d_out))

    def forward(self, z, didx):
        return self.net(torch.cat([z, self.demb(didx)], -1))


def batches(x, rng, n):
    starts = rng.integers(0, len(x) - L * P - 1, n)
    idx = starts[:, None] + np.arange(L * P)[None]
    return torch.from_numpy(x[idx].reshape(n, L, P)).to(DEV), starts


def rankme(Z, eps=1e-7):
    """Effective rank (RankMe, Garrido et al. 2023): collapse monitor."""
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


ALIASES = {"nepa": ("reg", "ema"), "cpc": ("nce", "ema"), "ar": ("ar", None)}


def main(args):
    if "mode" in args:                       # deprecated alias (one commit)
        loss_kind, target = ALIASES[args["mode"]]
    else:
        loss_kind = args.get("loss", "reg")
        target = args.get("target", "ema")
    gated = args.get("gate", "1") == "1"
    xcov = args.get("xcov", args.get("dcor", "0")) == "1"
    # D7 applied 2026-08-02: no RankMe collapse at vfloor=0 (22 vs 14 with the
    # term ON) -> default off, term removed from the paper's loss equation
    vfloor = args.get("vfloor", "0") == "1"
    seed = int(args.get("seed", 0))
    d_slow = int(args.get("dslow", 16))
    lam = float(args.get("lam", 4))
    regime_mode = args.get("regime_mode", "freq")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # D3: seeds resample the data; error bars reflect data + training stochasticity
    train = make_dataset(1_000_000, seed=seed, regime_mode=regime_mode)
    evald = make_dataset(200_000, seed=100_000 + seed, regime_mode=regime_mode)

    # D1 (amended): tau = c * chosen, c = ln(1/eps); c= override = appendix sweep
    eps = float(args.get("eps", 0.05))
    c_override = float(args["c"]) if "c" in args else None
    tau_arg = args.get("tau", "16")
    if tau_arg == "auto":
        from tac import tau_from_tac
        tac_info = tau_from_tac(train["x"], patch=P, eps=eps, c=c_override)
        tau, tau_src = tac_info["tau"], "auto"
    else:
        tau, tau_src, tac_info = float(tau_arg), "manual", None

    stem = loss_kind if loss_kind == "ar" else f"{loss_kind}-{target}"
    tag = (f"{stem}_g{int(gated)}_x{int(xcov)}_s{seed}_tau{'auto' if tau_src == 'auto' else int(tau)}"
           f"_ds{d_slow}_lam{int(lam)}_vf{int(vfloor)}"
           + ("_vm" if regime_mode == "variance" else ""))

    enc = Encoder().to(DEV)
    pred = Predictor(P if loss_kind == "ar" else D_Z).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    use_ema = loss_kind != "ar" and target == "ema"
    if use_ema:
        tgt = Encoder().to(DEV)
        tgt.load_state_dict(enc.state_dict())
        for p in tgt.parameters():
            p.requires_grad_(False)

    gvals = torch.sigmoid((tau - torch.tensor(OFFSETS, dtype=torch.float32)) / W).to(DEV)
    n_anchor = 8
    for step in range(STEPS):
        xb, _ = batches(train["x"], rng, BATCH)
        z = enc(xb)
        anchors = torch.from_numpy(
            rng.integers(64, L - max(OFFSETS), (BATCH, n_anchor))).to(DEV)
        didx = torch.from_numpy(
            rng.integers(0, len(OFFSETS), (BATCH, n_anchor))).to(DEV)
        dvals = torch.tensor(OFFSETS, device=DEV)[didx]
        bi = torch.arange(BATCH, device=DEV)[:, None].expand_as(anchors)
        za = z[bi.flatten(), anchors.flatten()]
        g = gvals[didx.flatten()].unsqueeze(-1) if gated else 1.0
        za_in = torch.cat([za[:, :d_slow], za[:, d_slow:] * g], -1)
        zhat = pred(za_in, didx.flatten())
        if loss_kind == "ar":                   # predict raw future patch
            ztgt = xb[bi.flatten(), (anchors + dvals).flatten()]
        elif use_ema:
            with torch.no_grad():
                ztgt = tgt(xb)[bi.flatten(), (anchors + dvals).flatten()]
        else:  # D2 online target: both-sided gradients, no EMA, no stop-grad
            ztgt = z[bi.flatten(), (anchors + dvals).flatten()]
        if loss_kind == "nce":                  # InfoNCE, in-batch negatives
            logits = F.normalize(zhat, dim=-1) @ F.normalize(ztgt, dim=-1).T / 0.1
            loss = F.cross_entropy(logits, torch.arange(len(zhat), device=DEV))
        else:
            loss = ((zhat - ztgt) ** 2).mean()
        if vfloor:                              # P0-2: variance floor, D7 pending
            loss = loss + F.relu(1.0 - z.reshape(-1, D_Z).std(0)).mean()
        if xcov:
            zc = za - za.mean(0)
            C = (zc[:, :d_slow].T @ zc[:, d_slow:]) / (len(za) - 1)
            loss = loss + lam * (C ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if use_ema:
            with torch.no_grad():
                for pe, pt in zip(enc.parameters(), tgt.parameters()):
                    pt.mul_(EMA).add_(pe, alpha=1 - EMA)
        if step % 1000 == 0:
            zs = z.reshape(-1, D_Z).std(0)
            print(f"[{tag}] step {step} loss {loss.item():.4f} "
                  f"std {zs[:d_slow].mean():.3f}/{zs[d_slow:].mean():.3f}", flush=True)

    # ---- probe evaluation ----
    # Leak-free: tile the eval series into DISJOINT windows and split by a
    # contiguous time cut, so no probe-train window overlaps a probe-test one.
    enc.eval()
    x = evald["x"]
    starts = np.arange(0, len(x) - L * P, L * P)          # non-overlapping
    embs, ys, yu, yphi = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack([x[s:s + L * P].reshape(L, P) for s in bs])).to(DEV)
            embs.append(enc(xb)[:, -1].cpu().numpy())
            end = bs + L * P - 1
            ys.append(evald["s"][end]); yu.append(evald["u"][end])
            yphi.append(np.stack([evald["sin_phi"][end], evald["cos_phi"][end]], 1))
    Z = np.concatenate(embs); ys = np.concatenate(ys)
    yu = np.concatenate(yu); yphi = np.concatenate(yphi)
    ntr = len(Z) // 2                                     # contiguous cut (early=train)
    res = {"tag": tag, "n_probe": int(len(Z)), "rankme": rankme(Z),
           "config": {"loss": loss_kind, "target": target, "gate": gated,
                      "xcov": xcov, "vfloor": vfloor, "seed": seed,
                      "data_seed": seed, "eval_seed": 100_000 + seed,
                      "tau": tau, "tau_source": tau_src,
                      "eps": eps, "c_override": c_override,
                      "dslow": d_slow, "lam": lam, "regime_mode": regime_mode,
                      "tac": tac_info}}
    for name, sl in [("z_slow", slice(0, d_slow)), ("z_fast", slice(d_slow, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        B = Z[:, sl]
        res[f"{name}->regime_acc"] = LogisticRegression(max_iter=2000).fit(
            B[:ntr], ys[:ntr]).score(B[ntr:], ys[ntr:])
        res[f"{name}->u_r2"] = Ridge().fit(B[:ntr], yu[:ntr]).score(B[ntr:], yu[ntr:])
        res[f"{name}->phase_r2"] = Ridge().fit(B[:ntr], yphi[:ntr]).score(B[ntr:], yphi[ntr:])
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs/{tag}.json", "w"), indent=2)
    np.savez(f"runs/emb_{tag}.npz", Z=Z, regime=ys, u=yu, phase=yphi)  # for DCI/MIG
    if loss_kind != "ar":
        ck = {"enc": enc.state_dict(), "pred": pred.state_dict()}
        if use_ema:
            ck["tgt"] = tgt.state_dict()
        torch.save(ck, f"runs/model_{tag}.pt")


if __name__ == "__main__":
    import os; os.makedirs("runs", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
