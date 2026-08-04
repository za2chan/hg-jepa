"""HGLP v2 encoder + predictor (protocol B1/B2/B3).

Encoder: channel-mixing input (A4), learned absolute positions (B1),
causal transformer, per-block LayerNorm on the D_Z output (B2). Accepts
T<=L inputs; a short slice is indexed FROM 0 (B5 slice-indexing decision).

Predictor: continuous Δ conditioning (B3) — log2(Δ) expanded by a small MLP
to a vector, concatenated with the (gated) anchor embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

P, L = 8, 256
D_MODEL, D_Z, D_SLOW = 96, 64, 16


class Encoder(nn.Module):
    def __init__(self, in_dim=P):               # A4 channel-mixing: in_dim = patch_len * n_channels
        super().__init__()
        self.embed = nn.Linear(in_dim, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, L, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(D_MODEL, D_Z)
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):                       # x: (B, T, P), T<=L, indexed from 0
        T = x.shape[1]
        h = self.embed(x) + self.pos[:, :T]
        h = self.tf(h, mask=self.mask[:T, :T])
        z = self.out(h)
        zs = F.layer_norm(z[..., :D_SLOW], (D_SLOW,))            # B2 per-block LN
        zf = F.layer_norm(z[..., D_SLOW:], (D_Z - D_SLOW,))
        return torch.cat([zs, zf], -1)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dmlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 16))
        self.net = nn.Sequential(nn.Linear(D_Z + 16, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D_Z))

    def forward(self, z, log2delta):            # log2delta: (N, 1)
        return self.net(torch.cat([z, self.dmlp(log2delta)], -1))


if __name__ == "__main__":
    enc, pred = Encoder(), Predictor()
    full = enc(torch.randn(2, L, P))
    slc = enc(torch.randn(5, 8, P))             # short slice, indexed from 0
    assert full.shape == (2, L, D_Z) and slc.shape == (5, 8, D_Z)
    yh = pred(torch.randn(7, D_Z), torch.randn(7, 1))
    assert yh.shape == (7, D_Z)
    # per-block LN: each block is unit-scaled independently
    import torch as T
    z = full.reshape(-1, D_Z)
    assert abs(z[:, :D_SLOW].std().item() - 1) < 0.2 and abs(z[:, D_SLOW:].std().item() - 1) < 0.2
    print("shapes OK | full", tuple(full.shape), "slice", tuple(slc.shape))
