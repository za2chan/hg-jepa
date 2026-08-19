"""HGLP v2 training on pre-cut real windows (HAPT / PTB-XL), scoped to the
separation + shift-robustness story. HGLP-Reg, RF-bounded target (B5, adopted).

Data: (n, L, in_dim) windows + per-patch labels + per-patch fast proxy + group
id. A3 global per-axis normalization is fit on the TRAIN group only. C3 group
split. C1 multi-position probe (labeled positions only). B4/B5/B6 as in train.py.
"""
import contextlib

import hashlib, json, os, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from model import Encoder, Predictor, L, D_Z, D_SLOW
from probes import rankme, encode_all, multi_position, BLOCKS, C_MIN

DEV = "cuda" if torch.cuda.is_available() else "cpu"




def _norm_stats(W, n_ax):
    """A3 per-axis global mean/std over patch-samples of that axis."""
    T = W.shape[-1]; per = T // n_ax
    mu = np.zeros(n_ax, np.float32); sd = np.ones(n_ax, np.float32)
    for a in range(n_ax):
        cols = [a + n_ax * k for k in range(per)]     # dims of axis a across samples
        mu[a] = W[:, :, cols].mean(); sd[a] = W[:, :, cols].std() + 1e-6
    return mu, sd, per


def _apply_norm(W, mu, sd, per, n_ax):
    W = W.copy()
    for a in range(n_ax):
        cols = [a + n_ax * k for k in range(per)]
        W[:, :, cols] = (W[:, :, cols] - mu[a]) / sd[a]
    return W


_NPZ_MEMO = {}


def _load_npz(npz):
    """Memoise the raw arrays. rotation.py calls train_real six times per seed on
    the same file, and the read alone is ~35 s on the 3 GB Sleep-EDF set. Keyed on
    (path, size, mtime) so a regenerated file is never served from the memo."""
    st = os.stat(npz)
    k = (str(npz), st.st_size, int(st.st_mtime))
    if k not in _NPZ_MEMO:
        _NPZ_MEMO.clear()                    # one file at a time; these are GB-sized
        d = np.load(npz)
        _NPZ_MEMO[k] = (d["W"], d["lab"], d["fast"], d["subj"])
    return _NPZ_MEMO[k]


def train_real(npz, n_ax=3, seed=0, steps=2500, tau=40.0, W_gate=4.0, lam=4.0,
               w=12, dmin=12, dmax=128, batch=64, n_anchor=16, n_delta=4, lr=3e-4,
               ema=0.996, min_context=16, gate=True, xcov=True, blocknorm=True,
               gate_sym=False, d_slow=D_SLOW,
               loss_kind="reg", target_enc="ema", temp=0.1, mask_same_window=True,
               log_every=500, train_idx=None, cache=True):
    assert dmin >= w
    use_ema = target_enc == "ema"
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Wall, lab, fast, grp = _load_npz(npz)
    in_dim = Wall.shape[-1]
    Lw = Wall.shape[1]                                   # actual window length (HAPT 256, PTB-XL 100)

    if train_idx is None:
        # C3 group split: hold out ~1/3 of groups
        groups = np.unique(grp)
        test_g = set(rng.permutation(groups)[:max(1, len(groups) // 3)].tolist())
        is_test = np.array([g in test_g for g in grp])
    else:                                     # explicit train set (exp/domain_shift.py)
        is_test = np.ones(len(grp), bool); is_test[train_idx] = False
    tr = np.flatnonzero(~is_test)

    mu, sd, per = _norm_stats(Wall[tr], n_ax)            # A3 fit on train only
    Wn = _apply_norm(Wall, mu, sd, per, n_ax)
    Wt = torch.from_numpy(Wn).to(DEV)

    # --- checkpoint cache -------------------------------------------------
    # Keyed on EVERY argument that changes the trained weights, plus the data
    # file's size+mtime. A config that differs anywhere gets a different key, so
    # a stale checkpoint can never be silently reused. Only the encoder/target
    # weights are stored; splits and normalisation are recomputed deterministically
    # from `seed`, which is cheap.
    ck_path = None
    if cache:
        st = os.stat(npz)
        key = dict(npz=os.path.basename(str(npz)), bytes=st.st_size, mtime=int(st.st_mtime),
                   n_ax=n_ax, seed=seed, steps=steps, tau=tau, W_gate=W_gate, lam=lam,
                   w=w, dmin=dmin, dmax=dmax, batch=batch, n_anchor=n_anchor,
                   n_delta=n_delta, lr=lr, ema=ema, min_context=min_context,
                   gate=bool(gate), xcov=bool(xcov), blocknorm=bool(blocknorm),
                   d_slow=d_slow, loss_kind=loss_kind, target_enc=target_enc,
                   # only recorded when set, so every existing checkpoint keeps its
                   # hash and the 36 runs already on disk stay reusable
                   **({"gate_sym": True} if (gate and gate_sym) else {}),
                   temp=temp, mask_same_window=bool(mask_same_window),
                   train_idx=(None if train_idx is None else
                              hashlib.sha1(np.asarray(train_idx).tobytes()).hexdigest()[:12]))
        # gate=False makes tau/W_gate inert; drop them so the ungated control is
        # shared across tau settings instead of retrained per config.
        if not gate:
            key.pop("tau"); key.pop("W_gate")
        h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        cdir = pathlib.Path(__file__).resolve().parents[2] / "runs_v2" / "ckpt"
        cdir.mkdir(parents=True, exist_ok=True)
        ck_path = cdir / f"tr_{h}.pt"

    mk = lambda: Encoder(in_dim, blocknorm, d_slow).to(DEV)
    enc, pred = mk(), Predictor().to(DEV)
    tgt = mk(); tgt.load_state_dict(enc.state_dict())
    for p_ in tgt.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=lr)

    if ck_path is not None and ck_path.exists():
        sd_ = torch.load(ck_path, map_location=DEV)
        enc.load_state_dict(sd_["enc"]); tgt.load_state_dict(sd_["tgt"])
        print(f"  [cache hit] {ck_path.name}", flush=True)
        return dict(enc=enc, tgt=(tgt if use_ema else enc), Wt=Wt, lab=lab, fast=fast,
                    is_test=is_test, tr=tr, te=np.flatnonzero(is_test),
                    losses=sd_.get("losses", []), norm=(mu, sd, per, n_ax), in_dim=in_dim)


    ar_w = torch.arange(w, device=DEV)
    bi_const = np.broadcast_to(np.arange(batch)[:, None, None],
                               (batch, n_anchor, n_delta)).reshape(-1)
    neg_mask = (torch.from_numpy(bi_const[:, None] == bi_const[None, :]).to(DEV)
                & ~torch.eye(len(bi_const), dtype=torch.bool, device=DEV))
    losses = []
    for step in range(steps):
        bi_win = rng.choice(tr, batch)
        xb = Wt[bi_win]                                  # (B, L, in_dim), train groups
        z = enc(xb)

        anchors = rng.integers(min_context, Lw - dmin, (batch, n_anchor))
        hi = np.minimum(dmax, Lw - 1 - anchors)
        lo = np.log(dmin)
        u = rng.random((batch, n_anchor, n_delta))
        deltas = np.exp(lo + u * (np.log(hi)[..., None] - lo)).round().astype(np.int64)
        deltas = np.clip(deltas, dmin, hi[..., None])

        bidx = np.broadcast_to(np.arange(batch)[:, None, None], deltas.shape).reshape(-1)
        ai = np.broadcast_to(anchors[:, :, None], deltas.shape).reshape(-1)
        di = deltas.reshape(-1); tp = ai + di
        biT = torch.from_numpy(bidx).to(DEV)
        za = z[biT, torch.from_numpy(ai).to(DEV)]
        dT = torch.from_numpy(di).to(DEV).float()

        g = torch.sigmoid((tau - dT) / W_gate).unsqueeze(-1) if gate else 1.0
        # See train.py for the argument. Short version: with the default one-sided
        # gate, z_slow is always on, so transient content in it is free profit at
        # short Delta -- which is precisely why the Exclusion clause cannot be argued
        # from the objective. gate_sym mutes z_slow below tau so it pays nowhere.
        gs = (1.0 - g) if (gate and gate_sym) else 1.0
        za_in = torch.cat([za[:, :d_slow] * gs, za[:, d_slow:] * g], -1)
        zhat = pred(za_in, torch.log2(dT).unsqueeze(-1))
        tenc = tgt if use_ema else enc          # online (D2): both-sided grads
        with (torch.no_grad() if use_ema else contextlib.nullcontext()):
            starts = torch.from_numpy(tp - w).to(DEV)[:, None] + ar_w
            ztgt = tenc(xb[biT[:, None], starts])[:, -1]  # B5 bounded, indexed from 0

        if loss_kind == "reg":
            loss = ((zhat - ztgt) ** 2).mean()
        elif loss_kind == "l1":
            loss = (zhat - ztgt).abs().mean()
        else:                                    # InfoNCE, in-batch negatives
            logits = F.normalize(zhat, dim=-1) @ F.normalize(ztgt, dim=-1).T / temp
            if mask_same_window:
                logits = logits.masked_fill(neg_mask[:len(logits), :len(logits)], -1e4)
            loss = F.cross_entropy(logits, torch.arange(len(logits), device=DEV))
        if xcov:
            zc = za - za.mean(0)
            C = (zc[:, :d_slow].T @ zc[:, d_slow:]) / (len(za) - 1)
            loss = loss + lam * (C ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if use_ema:
            with torch.no_grad():
                for pe, pt in zip(enc.parameters(), tgt.parameters()):
                    pt.mul_(ema).add_(pe, alpha=1 - ema)
        losses.append(loss.item())
        if step % log_every == 0:
            print(f"  step {step} loss {loss.item():.4f}", flush=True)

    if ck_path is not None:
        tmp = ck_path.with_suffix(".pt.tmp")          # atomic: never leave a half file
        torch.save({"enc": enc.state_dict(), "tgt": tgt.state_dict(),
                    "losses": losses, "key": key}, tmp)
        tmp.replace(ck_path)

    return dict(enc=enc, tgt=(tgt if use_ema else enc), Wt=Wt, lab=lab, fast=fast, is_test=is_test,
                tr=tr, te=np.flatnonzero(is_test), losses=losses,
                norm=(mu, sd, per, n_ax), in_dim=in_dim)



def probe(res, block="z_slow", c_min=C_MIN, max_samples=60000):
    """C1 multi-position probe with the context floor (see probes.multi_position)."""
    Z = encode_all(res["enc"], res["Wt"])
    return multi_position(Z, res["lab"], res["fast"], res["tr"], res["te"],
                          block=block, c_min=c_min, max_samples=max_samples)


@torch.no_grad()
def shift_eval(res, strengths=(0.0, 0.25, 0.5, 1.0, 2.0), kind="noise", seed=0):
    """P3-4: fit the activity probe on CLEAN train data, evaluate on held-out
    test windows perturbed at increasing strength. Claim: z_slow degrades less
    than z_full. Perturbation is in normalized input space (deploy-time norm)."""
    enc, Wt, lab, fast = res["enc"], res["Wt"], res["lab"], res["fast"]
    labeled = (lab >= 1) & (lab <= 6)
    labeled[:, :C_MIN] = False                  # C1: same context floor as probe()
    g = torch.Generator(device=DEV).manual_seed(seed)

    def enc_all(Wc):
        return encode_all(enc, Wc)
    Zc = enc_all(Wt)
    Wte = Wt[res["te"]]; mte = labeled[res["te"]]; yte = lab[res["te"]][mte] - 1
    out = dict(strengths=list(strengths), kind=kind)
    for block, sl in [("z_slow", slice(0, D_SLOW)), ("z_full", slice(0, D_Z))]:
        Bc = Zc[:, :, sl]
        ftr = Bc[res["tr"]][labeled[res["tr"]]]; ytr = lab[res["tr"]][labeled[res["tr"]]] - 1
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(ftr, ytr)
        f1s = []
        for s in strengths:
            if kind == "noise":
                Wp = Wte + s * torch.randn(Wte.shape, generator=g, device=DEV)
            else:                                       # amplitude scale
                Wp = Wte * (1.0 + s)
            Zp = enc_all(Wp)[:, :, sl]
            f1s.append(float(f1_score(yte, clf.predict(Zp[mte]), average="macro")))
        out[block] = f1s
    out["retention_ratio"] = [b / (a + 1e-9) for a, b in
                              zip([out["z_full"][0]] * len(strengths), out["z_full"])]
    return out


if __name__ == "__main__":
    import json, os
    os.makedirs("../runs_v2", exist_ok=True)
    print("=== HAPT v2 (HGLP-Reg, bounded target, gate+xcov) ===")
    res = train_real("../data/hapt_v2.npz", seed=0)
    out = {b: probe(res, b) for b in ("z_slow", "z_fast", "z_full")}
    for b, r in out.items():
        print(f"  {b:7s} activity-F1 {r['slow_kept_f1']:.3f} | leak(accmag R2) "
              f"{r['leak_r2']:.3f} | RankMe {r['rankme']:.1f}")
    print("--- shift robustness (probe fit clean, eval on perturbed test) ---")
    sh = {k: shift_eval(res, kind=k) for k in ("noise", "scale")}
    for k, s in sh.items():
        print(f"  {k}: strengths {s['strengths']}")
        print(f"    z_slow F1 {[round(v,3) for v in s['z_slow']]}")
        print(f"    z_full F1 {[round(v,3) for v in s['z_full']]}")
    out["shift"] = sh
    json.dump(out, open("../runs_v2/hapt_v2_seed0.json", "w"), indent=2)
    print("saved runs_v2/hapt_v2_seed0.json")
