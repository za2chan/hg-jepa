"""PatchTST self-supervised (masked patch reconstruction) as a representation
baseline. Upstream is vendored unedited at commit 204c21ef; see
`third_party/PROVENANCE.md`. Every behaviour change lives HERE, not there.

WHAT IS REUSED VERBATIM: `PatchTST(head_type='pretrain')` (backbone + linear
reconstruction head), `create_patch`, `random_masking`, and the masked-MSE loss.
The upstream `Learner` is not used: it is epoch-driven and its `set_device`
picks a GPU by `utilization < 5%`, which fails on a shared GPU. Its optimiser
and schedule are reproduced exactly (Adam + OneCycleLR, pct_start=0.3), so the
only substantive difference is a step budget instead of an epoch budget.

FOUR DELIBERATE DEVIATIONS, all forced by common.py's fairness contract and all
recorded in `meta` so the paper can state them:

1. RevIN is OFF (upstream default `--revin 1`). Contract rule 1 forbids a second
   normalisation on top of A3. It is also the fair choice rather than a
   handicap: RevIN subtracts each window's own per-channel mean, which on HAPT
   would delete the gravity/posture DC component that our encoder does see.
2. patch_len == stride, chosen as the largest divisor of L that is <= 16, so the
   patches TILE L exactly. Upstream's `create_patch` drops a prefix of
   `L - (patch_len + stride*(num_patch-1))` positions; with a tiling choice that
   prefix is empty and every one of our L positions belongs to exactly one patch.
   (Upstream's own SSL default is also non-overlapping: patch_len=stride=12.)
3. The SSL objective only ever samples windows in `tr` (contract rule 2).
4. Representation = backbone output, reconstruction head discarded — which is
   upstream's own downstream convention (`transfer_weights(..., exclude_head=True)`).

PATCH -> POSITION MAPPING. Two halves, only one of them exact:
  * INPUT side is EXACT and lossless. With stride == patch_len and L % patch_len
    == 0, `create_patch` is a bijection: our position p lands in patch p//P at
    offset p%P, nowhere else. Asserted by `torch.equal` in the self-check.
  * OUTPUT side is an UPSAMPLE, NOT exact. The backbone emits one vector per
    patch, so `embed` repeats each patch vector P times. All P positions inside a
    patch receive the IDENTICAL representation. Position p therefore carries
    information about up to P-1 positions after it. This is subsumed by the
    larger issue that the model is bidirectional anyway (verified in the
    self-check), but it must be stated: the probe at position p is NOT reading a
    position-p-specific representation, it is reading patch p//P.
"""
import importlib
import importlib.util
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common
from common import DEV

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_SSL = _ROOT / "third_party" / "PatchTST" / "PatchTST_self_supervised"
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def _vendored():
    """Import upstream's package under the alias `patchtst_src`.

    WHY the alias: upstream's package is literally named `src`, which collides
    with OUR `src/`. A plain `sys.path` insert would let whichever `src` sorts
    first win — silently, and differently depending on the caller's cwd. Every
    upstream import we need is relative, so renaming the top level is safe.
    """
    if "patchtst_src" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "patchtst_src", _SSL / "src" / "__init__.py",
            submodule_search_locations=[str(_SSL / "src")])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["patchtst_src"] = mod
        spec.loader.exec_module(mod)
    return (importlib.import_module("patchtst_src.models.patchTST"),
            importlib.import_module("patchtst_src.callback.patch_mask"))


_model_mod, _mask_mod = _vendored()
PatchTST = _model_mod.PatchTST
create_patch, random_masking = _mask_mod.create_patch, _mask_mod.random_masking


def _tiling_patch_len(L, cap=16):
    """Largest divisor of L <= cap (16 = the paper's supervised patch length)."""
    return max(d for d in range(1, cap + 1) if L % d == 0)


def _masked_mse(pred, target, mask):
    """Verbatim `PatchMaskCB._loss`; inlined because the callback needs a Learner.
    pred/target: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]."""
    loss = ((pred - target) ** 2).mean(dim=-1)
    return (loss * mask).sum() / mask.sum()


def fit_encoder(npz, n_ax, train_idx=None, seed=0, steps=2000, batch=64, lr=1e-4,
                mask_ratio=0.4, patch_len=None, d_model=128, n_layers=3, n_heads=16,
                d_ff=512, dropout=0.2, log_every=None):
    """Masked-patch pretrain on the TRAIN windows only; return the contract dict.

    Defaults are upstream's `patchtst_pretrain.py` argparse defaults, except
    `steps` (upstream counts epochs; 10 epochs over HAPT's ~1.4k train windows
    would be ~220 steps) and `lr`, where upstream runs `lr_finder()` first and
    pretrains at its suggestion — we take the script's own `--lr` default rather
    than let an lr search see the data.
    """
    Wt, lab, fast, tr, te = common.load_windows(npz, n_ax, train_idx, seed)
    N, L, C = Wt.shape
    P = patch_len or _tiling_patch_len(L)
    assert L % P == 0, f"patch_len {P} must tile L={L}; else create_patch drops a prefix"
    num_patch = L // P

    torch.manual_seed(seed)
    model = PatchTST(c_in=C, target_dim=1,          # target_dim unused by the pretrain head
                     patch_len=P, stride=P, num_patch=num_patch,
                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                     shared_embedding=True, d_ff=d_ff, dropout=dropout,
                     head_dropout=dropout, act='relu', head_type='pretrain',
                     res_attention=False).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=lr)        # upstream Learner default
    sched = torch.optim.lr_scheduler.OneCycleLR(              # upstream fit_one_cycle default
        opt, max_lr=lr, total_steps=steps, pct_start=0.3)

    rng = np.random.default_rng(seed)
    curve = []
    model.train()
    for it in range(steps):
        idx = rng.choice(tr, batch, replace=len(tr) < batch)   # contract 2: train windows only
        xb = Wt[torch.from_numpy(idx).to(DEV)]                 # (B, L, C)
        xp, _ = create_patch(xb, P, P)                         # (B, num_patch, C, P)
        xm, _, mask, _ = random_masking(xp, mask_ratio)        # masks per (patch, channel)
        loss = _masked_mse(model(xm), xp, mask.bool())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        curve.append(loss.item())
        if log_every and (it + 1) % log_every == 0:
            print(f"  [patchtst] step {it + 1}/{steps}  masked-MSE {np.mean(curve[-log_every:]):.4f}")

    D = C * d_model      # per-patch state of every channel, concatenated

    @torch.no_grad()
    def embed(x):
        """(B, L, C) -> (B, L, D). eval() matters: BatchNorm + dropout are on in training."""
        model.eval()
        xp, _ = create_patch(x, P, P)
        h = model.backbone(xp)                                  # (B, C, d_model, num_patch)
        # Channels are concatenated, matching upstream's own classification/regression
        # heads (`nn.Flatten` over n_vars*d_model). They never interacted inside the
        # model -- see channel_independent in meta.
        h = h.permute(0, 3, 1, 2).reshape(x.shape[0], num_patch, D)
        return h.repeat_interleave(P, dim=1)                    # patch -> P positions (upsample)

    meta = dict(
        name="PatchTST-SSL", commit=COMMIT, licence="Apache-2.0",
        paper="arXiv:2211.14730", variant="masked patch reconstruction (PatchTST_self_supervised)",
        causal=False,                       # bidirectional attention; verified in __main__
        params=sum(p.numel() for p in model.backbone.parameters()),
        params_with_pretrain_head=sum(p.numel() for p in model.parameters()),
        patch_len=P, stride=P, num_patch=num_patch, mask_ratio=mask_ratio,
        channel_independent=True,           # (B*C, num_patch, d_model): channels never mix
        position_map="repeat_interleave: patch j -> positions [jP, (j+1)P); UPSAMPLE, not exact",
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        revin=False,                        # deviation 1 in the module docstring
        steps=steps, batch=batch, lr=lr, seed=seed,
        loss_first20=float(np.mean(curve[:20])), loss_last20=float(np.mean(curve[-20:])),
    )
    return dict(embed=embed, Wt=Wt, lab=lab, fast=fast, tr=tr, te=te, D=D,
                meta=meta, loss_curve=curve, model=model)


if __name__ == "__main__":
    torch.manual_seed(0)
    res = fit_encoder(_ROOT / "data" / "hapt_v2.npz", n_ax=3, steps=200, log_every=50)
    Wt, m, P = res["Wt"], res["meta"], res["meta"]["patch_len"]
    B, L, C = Wt.shape

    # check_contract embeds Wt[:8] but compares against Wt's own leading dim, so it
    # must be handed the same 8 windows it will embed.
    print("\n(a) contract:", common.check_contract(res, Wt[:8]), "D =", res["D"],
          "| backbone params", m["params"])

    # (b) the SSL objective actually optimises
    c = res["loss_curve"]
    print(f"(b) masked-MSE {m['loss_first20']:.4f} (first 20) -> "
          f"{m['loss_last20']:.4f} (last 20)")
    assert m["loss_last20"] < m["loss_first20"], "masked reconstruction loss did not decrease"

    # (c) causality, empirically: perturb the LAST patch, look at the FIRST patch.
    x = Wt[:4].clone()
    y = x.clone(); y[:, -P:] += 5.0
    zx, zy = res["embed"](x), res["embed"](y)
    early = (zx[:, :P] - zy[:, :P]).abs().max().item()
    scale = zx.abs().mean().item()
    print(f"(c) perturbed positions [{L - P}:{L}]; max |dz| at positions [0:{P}] = "
          f"{early:.4f} (mean |z| = {scale:.4f}) -> "
          f"{'CAUSAL' if early < 1e-5 else 'NOT CAUSAL (bidirectional)'}")
    assert early > 1e-3, "expected a bidirectional model; a causal one contradicts meta"
    assert not m["causal"]

    # (d) the documented patch <-> position mapping
    xp, npatch = create_patch(x, P, P)                       # (B, num_patch, C, P)
    assert torch.equal(xp.permute(0, 1, 3, 2).reshape(x.shape[0], L, C), x), \
        "input mapping is NOT a bijection -- create_patch dropped or reordered positions"
    zr = zx.reshape(x.shape[0], npatch, P, res["D"])
    assert torch.equal(zr, zr[:, :, :1].expand_as(zr)), \
        "output is not piecewise-constant per patch -- upsample claim is wrong"
    print(f"(d) input positions <-> (patch, offset) bijection: EXACT (torch.equal) | "
          f"output constant within each of the {npatch} patches: yes (upsample, not exact)")

    # (e) channel independence, empirically: perturb channel 0, read channels 1..C-1.
    # Caveat worth knowing: upstream's norm='BatchNorm' normalises over a batch of
    # shape (bs*nvars, ...), so at TRAIN time the BN statistics are pooled across
    # channels. Independence therefore holds for the eval-mode representation
    # function (running stats), not for the training dynamics.
    y2 = x.clone(); y2[:, :, 0] += 5.0
    dz = (res["embed"](y2) - zx).reshape(x.shape[0], L, C, m["d_model"]).abs().amax((0, 1, 3))
    print(f"(e) perturbed channel 0: max |dz| in ch0 block = {dz[0]:.4f}, "
          f"in ch1..{C - 1} blocks = {dz[1:].max():.2e} -> "
          f"{'CHANNEL-INDEPENDENT' if dz[1:].max() < 1e-5 else 'channels mix'}")
    assert dz[1:].max() < 1e-5 and m["channel_independent"]

    print("\nmeta:", {k: v for k, v in m.items() if k != "loss_curve"})
