"""HGLP v2 encoder + predictor (protocol B1/B2/B3, B1 revised 2026-08-04).

B1 (revised): ROTARY position encoding (RoPE), not learned absolute. Rationale:
  - No per-position parameter exists, so the "starved position" failure mode is
    impossible by construction (learned absolute left 12 positions untrained).
  - B5 feeds VARIABLE-LENGTH slices to the target encoder; relative-only
    positions make that in-distribution with no indexing convention needed.
  - Probe position stops being special: any position is as trained as any other.
A2/A4: channel-mixing input. B2: per-block LayerNorm on the D_Z output.
B3: continuous Δ conditioning — log2(Δ) expanded to a vector by a small MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

P, L = 8, 256
D_MODEL, D_Z, D_SLOW = 96, 64, 16
N_HEAD, N_LAYER, D_FF = 4, 4, 256


def _rope_tables(seq, dim, device, base=10000.0):
    """cos/sin tables for RoPE. dim = per-head dim, must be even."""
    half = dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    ang = torch.outer(torch.arange(seq, device=device).float(), inv)   # (seq, half)
    return ang.cos()[None, None], ang.sin()[None, None]               # (1,1,seq,half)


def _apply_rope(x, cos, sin):
    """x: (B, H, T, Dh) — rotate (even, odd) pairs by the position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.h, self.dh = n_head, d_model // n_head
        assert self.dh % 2 == 0, "head dim must be even for RoPE"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        shape = lambda z: z.view(B, T, self.h, self.dh).transpose(1, 2)   # (B,H,T,Dh)
        q, k, v = shape(q), shape(k), shape(v)
        q = _apply_rope(q, cos[..., :T, :], sin[..., :T, :])
        k = _apply_rope(k, cos[..., :T, :], sin[..., :T, :])
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    """Pre-LN transformer block with RoPE attention."""
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_head)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ff(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, in_dim=P):                # A4: in_dim = patch_len * n_channels
        super().__init__()
        self.embed = nn.Linear(in_dim, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEAD, D_FF) for _ in range(N_LAYER)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, D_Z)

    def forward(self, x):                        # x: (B, T, in_dim), any T
        T = x.shape[1]
        cos, sin = _rope_tables(T, D_MODEL // N_HEAD, x.device)
        h = self.embed(x)
        for b in self.blocks:
            h = b(h, cos, sin)
        z = self.out(self.norm(h))
        zs = F.layer_norm(z[..., :D_SLOW], (D_SLOW,))              # B2 per-block LN
        zf = F.layer_norm(z[..., D_SLOW:], (D_Z - D_SLOW,))
        return torch.cat([zs, zf], -1)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dmlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 16))
        self.net = nn.Sequential(nn.Linear(D_Z + 16, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D_Z))

    def forward(self, z, log2delta):             # log2delta: (N, 1)
        return self.net(torch.cat([z, self.dmlp(log2delta)], -1))


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = Encoder().eval()
    full = enc(torch.randn(2, L, P))
    slc = enc(torch.randn(5, 8, P))              # short slice, no indexing convention
    assert full.shape == (2, L, D_Z) and slc.shape == (5, 8, D_Z)
    assert Predictor()(torch.randn(7, D_Z), torch.randn(7, 1)).shape == (7, D_Z)

    # RoPE property 1: no per-position parameter exists -> starvation impossible
    assert not any("pos" in n for n, _ in enc.named_parameters()), "positional param found"

    # RoPE property 2: PREFIX INVARIANCE. A slice encoded standalone equals the
    # same slice encoded as the prefix of a longer sequence (causal + relative
    # positions). This is exactly what makes B5's variable-length slices
    # in-distribution, and it is FALSE for learned absolute positions.
    x = torch.randn(1, 64, P)
    assert torch.allclose(enc(x[:, :8]), enc(x)[:, :8], atol=1e-5)

    # causality: changing the future must not change earlier outputs
    y = x.clone(); y[:, 32:] = torch.randn(1, 32, P)
    assert torch.allclose(enc(x)[:, :32], enc(y)[:, :32], atol=1e-5)
    print("model.py OK — RoPE, no positional params, prefix-invariant, causal")
