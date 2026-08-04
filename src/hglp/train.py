"""HGLP v2 training (protocol B4/B5/B6). HGLP-Reg (L2 + EMA target).

B4: sample Δ log-uniform, then anchor in [min_context, L-1-Δ]; several Δ per
    anchor for the fixed-anchor gate contrast. With RoPE (B1 revised) there are
    no per-position parameters, so anchor coverage no longer risks starving
    anything — density is now purely about training signal per forward pass.
B5: target = EMA encoder, either
      cumulative  -> read position (anchor+Δ) of the full-window encoding (v1)
      bounded     -> encode the w-slice x[anchor+Δ-w : anchor+Δ], read last (v2)
    Slice indexed from 0. ponytail: the pilot uses Δ_min=w so w_eff==w for every
    pair (fully batchable); that is exactly the decided design in the Δ>=w regime
    where harmlessness is claimed. Δ<w (gate-open) is skipped in the pilot; the
    full run drops Δ_min to 1 and groups slices by w_eff.
B6: loss = ||zhat - ztgt||^2 + lam * xcov, NO variance floor. Continuous gate
    g(Δ) = sigmoid((tau - Δ)/W) on z_fast.

Two orthogonal stem axes (protocol naming / D2):
    loss_kind  = reg | nce      L2 regression vs InfoNCE with in-batch negatives
    target_enc = ema | online   EMA copy vs the SAME encoder with both-sided
                                gradients (D2: no EMA, no stop-grad)
  HGLP-Reg = reg+ema (headline, D4) · HGLP-NCE = nce+online · nce+ema = control.

NCE negative structure: pairs drawn from the SAME window share the slow factor,
so using them as negatives would push apart representations that agree on the
slow state — i.e. it would reward fast information, the opposite of our goal.
mask_same_window=True (default) removes them; set False to reproduce v1's cpc.
"""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from datagen import make_dataset
from model import Encoder, Predictor, P, L, D_Z, D_SLOW

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rankme(Z, eps=1e-7):
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def _windows(x, rng, n):
    starts = rng.integers(0, len(x) - L * P - 1, n)
    idx = starts[:, None] + np.arange(L * P)[None]
    return torch.from_numpy(x[idx].reshape(n, L, P)).to(DEV)


def train(target_mode="bounded", loss_kind="reg", target_enc="ema", seed=0,
          steps=2500, tau=16.0, W=4.0, lam=4.0, w=8, dmin=8, dmax=128, batch=64,
          n_anchor=16, n_delta=4, lr=3e-4, ema=0.996, min_context=16, temp=0.1,
          mask_same_window=True, log_every=500):
    assert dmin >= w, "pilot expects Δ_min>=w so w_eff==w (see module docstring)"
    use_ema = target_enc == "ema"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = make_dataset(1_000_000, seed=seed)
    x = data["x"]

    enc, pred = Encoder().to(DEV), Predictor().to(DEV)
    tgt = Encoder().to(DEV); tgt.load_state_dict(enc.state_dict())
    for p_ in tgt.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=lr)

    ar_w = torch.arange(w, device=DEV)
    # window index of each (anchor, Δ) pair is fixed across steps -> precompute
    bi_const = np.broadcast_to(np.arange(batch)[:, None, None],
                               (batch, n_anchor, n_delta)).reshape(-1)
    npair = len(bi_const)
    neg_mask = (torch.from_numpy(bi_const[:, None] == bi_const[None, :]).to(DEV)
                & ~torch.eye(npair, dtype=torch.bool, device=DEV))
    losses = []
    for step in range(steps):
        xb = _windows(x, rng, batch)                       # (B, L, P)
        z = enc(xb)                                        # (B, L, D_Z)

        # ---- B4: Δ-first, anchor bound = L-1-Δ (per pair) ----
        anchors = rng.integers(min_context, L - dmin, (batch, n_anchor))
        hi = np.minimum(dmax, L - 1 - anchors)             # (B, na)
        lo = np.log(dmin)
        u = rng.random((batch, n_anchor, n_delta))
        deltas = np.exp(lo + u * (np.log(hi)[..., None] - lo)).round().astype(np.int64)
        deltas = np.clip(deltas, dmin, hi[..., None])      # (B, na, nd)

        bi = np.broadcast_to(np.arange(batch)[:, None, None], deltas.shape).reshape(-1)
        ai = np.broadcast_to(anchors[:, :, None], deltas.shape).reshape(-1)
        di = deltas.reshape(-1)
        tp = ai + di                                       # target positions
        biT = torch.from_numpy(bi).to(DEV)
        za = z[biT, torch.from_numpy(ai).to(DEV)]          # (N, D_Z)
        dT = torch.from_numpy(di).to(DEV).float()

        g = torch.sigmoid((tau - dT) / W).unsqueeze(-1)    # continuous gate
        za_in = torch.cat([za[:, :D_SLOW], za[:, D_SLOW:] * g], -1)
        zhat = pred(za_in, torch.log2(dT).unsqueeze(-1))

        # B5 target: receptive field (cumulative|bounded) x encoder (ema|online).
        # online (D2) = the SAME encoder, gradients on both sides, no stop-grad.
        tenc = tgt if use_ema else enc
        with (torch.no_grad() if use_ema else contextlib.nullcontext()):
            if target_mode == "cumulative":
                ztgt = tenc(xb)[biT, torch.from_numpy(tp).to(DEV)]
            elif target_mode == "bounded":
                starts = torch.from_numpy(tp - w).to(DEV)[:, None] + ar_w   # (N, w)
                slices = xb[biT[:, None], starts]          # (N, w, P), indexed from 0
                ztgt = tenc(slices)[:, -1]
            else:
                raise ValueError(target_mode)

        if loss_kind == "reg":
            loss = ((zhat - ztgt) ** 2).mean()
        elif loss_kind == "nce":                           # InfoNCE, in-batch negatives
            logits = F.normalize(zhat, dim=-1) @ F.normalize(ztgt, dim=-1).T / temp
            if mask_same_window:                           # drop slow-factor-sharing negs
                logits = logits.masked_fill(neg_mask[:len(logits), :len(logits)], -1e4)
            loss = F.cross_entropy(logits, torch.arange(len(logits), device=DEV))
        else:
            raise ValueError(loss_kind)
        zc = za - za.mean(0)                               # B6 xcov, no vfloor
        C = (zc[:, :D_SLOW].T @ zc[:, D_SLOW:]) / (len(za) - 1)
        loss = loss + lam * (C ** 2).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        if use_ema:
            with torch.no_grad():
                for pe, pt in zip(enc.parameters(), tgt.parameters()):
                    pt.mul_(ema).add_(pe, alpha=1 - ema)
        losses.append(loss.item())
        if step % log_every == 0:
            zs = z.reshape(-1, D_Z).std(0)
            print(f"[{loss_kind}+{target_enc}/{target_mode}] step {step} "
                  f"loss {loss.item():.4f} "
                  f"std {zs[:D_SLOW].mean():.3f}/{zs[D_SLOW:].mean():.3f}", flush=True)

    return dict(enc=enc, tgt=(tgt if use_ema else enc), pred=pred, losses=losses,
                data=data,
                cfg=dict(target_mode=target_mode, loss_kind=loss_kind,
                         target_enc=target_enc, seed=seed, tau=tau, W=W, lam=lam,
                         w=w, dmin=dmin, dmax=dmax, steps=steps, temp=temp,
                         mask_same_window=mask_same_window))
