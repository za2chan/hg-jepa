# Vendored upstream code

Nothing under `third_party/` is edited. Behaviour changes live in the adapters
under `src/baselines/`, so that a `git diff` against upstream stays empty and the
baseline numbers are attributable to the published method, not to us.

## ts2vec

| | |
|---|---|
| Repo | https://github.com/zhihanyue/ts2vec |
| Commit | `b0088e14a99706c05451316dc6db8d3da9351163` (2023-06-06) |
| Licence | MIT — Copyright (c) 2022 Zhihan Yue (`third_party/ts2vec/LICENSE`) |
| Vendored | 2026-08-05 |
| Paper | Yue et al., *TS2Vec: Towards Universal Representation of Time Series*, AAAI 2022, arXiv:2106.10466 |

Used as an SSL baseline: `TS2Vec.fit` / `.encode` provide per-timestamp
representations of our windows, wrapped by `src/baselines/ts2vec_adapter.py`.
Only `ts2vec.py`, `models/`, and `utils.py` are imported — they need nothing
beyond torch and numpy. `datautils.py` and `tasks/` (which pull in statsmodels,
bottleneck, pandas) are never imported.

## PatchTST

| | |
|---|---|
| Repo | https://github.com/yuqinie98/PatchTST |
| Commit | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` (2023-08-11) |
| Licence | Apache License 2.0 (`third_party/PatchTST/LICENSE`) |
| Vendored | 2026-08-05 |
| Paper | Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023, arXiv:2211.14730 |

Used as an SSL baseline: the **self-supervised path only**
(`PatchTST_self_supervised/`) — masked patch reconstruction — wrapped by
`src/baselines/patchtst_adapter.py`. Only `src/models/patchTST.py` (model) and
`src/callback/patch_mask.py` (`create_patch`, `random_masking`, masked MSE) are
imported; they need nothing beyond torch and numpy. The upstream `Learner` /
`datautils` / `basics.set_device` stack is not used — `set_device` selects a GPU
by `utilization < 5%` and would fail outright on our shared GPU.
