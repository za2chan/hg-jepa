"""Model-based screen (label-free): briefly train an unconstrained NEPA, then
measure how predictable the FAR-FUTURE embedding is from the current one.

A two-timescale signal has a slow factor that persists, so z_t predicts
z_{t+long} well above a time-shuffled control; a single-scale signal does not.
This sidesteps descriptor choice -- the encoder finds whatever descriptor the
slow factor lives in. Cheap: 800 steps. Validates against observed separation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL, D_Z = 96, 64
STEPS = 800
SHORT, LONG = 2, 64          # horizons (patches) for the predictability curve


def make_encoder(L, patch):
    class E(nn.Module):
        def __init__(s):
            super().__init__()
            s.embed = nn.Linear(patch, D_MODEL)
            s.pos = nn.Parameter(torch.randn(1, L, D_MODEL) * 0.02)
            layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True,
                                               norm_first=True, dropout=0.0)
            s.tf = nn.TransformerEncoder(layer, 3)
            s.out = nn.Linear(D_MODEL, D_Z)
            s.register_buffer("m", torch.triu(torch.full((L, L), float("-inf")), 1))

        def forward(s, x):
            h = s.tf(s.embed(x) + s.pos, mask=s.m)
            return F.layer_norm(s.out(h), (D_Z,))
    return E().to(DEV)


def brief_nepa(W):
    N, L, patch = W.shape
    Wt = torch.from_numpy(W).float().to(DEV)
    enc = make_encoder(L, patch)
    tgt = make_encoder(L, patch); tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4)
    pred = nn.Sequential(nn.Linear(D_Z + 1, 128), nn.GELU(), nn.Linear(128, D_Z)).to(DEV)
    opt.add_param_group({"params": pred.parameters()})
    rng = np.random.default_rng(0)
    offs = [1, 4, 16, 64]
    for _ in range(STEPS):
        idx = torch.from_numpy(rng.integers(0, N, 64)).to(DEV)
        z = enc(Wt[idx])
        with torch.no_grad():
            zt = tgt(Wt[idx])
        a = torch.from_numpy(rng.integers(8, L - max(offs), (64, 8))).to(DEV)
        d = torch.from_numpy(rng.integers(0, len(offs), (64, 8))).to(DEV)
        dv = torch.tensor(offs, device=DEV)[d]
        bi = torch.arange(64, device=DEV)[:, None].expand_as(a)
        za = z[bi.flatten(), a.flatten()]
        inp = torch.cat([za, dv.flatten()[:, None].float() / 64], -1)
        zh = pred(inp)
        ztg = zt[bi.flatten(), (a + dv).flatten()]
        loss = ((zh - ztg) ** 2).mean() + F.relu(1 - z.reshape(-1, D_Z).std(0)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pt in zip(enc.parameters(), tgt.parameters()):
                pt.mul_(0.996).add_(pe, alpha=0.004)
    return enc, Wt


def predictability(enc, Wt, delta):
    """R2 of predicting z_{t+delta} from z_t across windows, minus shuffle."""
    L = Wt.shape[1]
    with torch.no_grad():
        Z = torch.cat([enc(Wt[i:i + 256]) for i in range(0, len(Wt), 256)]).cpu().numpy()
    t = L // 4
    delta = min(delta, L - 1 - t)
    Zt, Ztd = Z[:, t, :], Z[:, t + delta, :]
    n = len(Zt) // 2
    r2 = Ridge().fit(Zt[:n], Ztd[:n]).score(Zt[n:], Ztd[n:])
    rng = np.random.default_rng(1)
    sh = rng.permutation(len(Ztd))
    r2s = Ridge().fit(Zt[:n], Ztd[sh][:n]).score(Zt[n:], Ztd[sh][n:])
    return r2 - r2s


def load_windows():
    from screen import load_windows as lw
    return lw()


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"{'dataset':10s} {'short-pred':>10} {'long-pred':>10}  slow-structure")
    for name, W in load_windows().items():
        enc, Wt = brief_nepa(W.astype(np.float32))
        ps = predictability(enc, Wt, SHORT)
        pl = predictability(enc, Wt, LONG)
        verdict = "STRONG" if pl > 0.3 else "WEAK" if pl > 0.1 else "NONE"
        print(f"{name:10s} {ps:10.3f} {pl:10.3f}  {verdict}")
