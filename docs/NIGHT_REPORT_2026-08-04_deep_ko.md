# 야간 작업 보고 — 구현 심층판 (2026-08-04)

`NIGHT_REPORT_2026-08-04_ko.md`(중간판)의 2절을 함수·텐서 수준까지 펼친 문서입니다. 중간판이 "무엇을 왜 만들었나"라면, 이 문서는 **"정확히 어떻게 동작하나"**입니다. 코드를 다시 읽지 않고도 검증·수정·인수인계가 가능하도록 썼습니다.

---

## 0. 전체 데이터 흐름

```
원신호
  └─ [합성] datagen.generate(seed) ──────────────┐
  └─ [HAPT] prep_hapt.py → hapt_v2.npz ──────────┤
  └─ [PTB-XL] ptbxl.npz → ptbxl_v2.npz ──────────┤
                                                  ▼
                            (n, L, in_dim) 윈도우 + (n, L) 위치별 라벨
                                                  │
                              A3 전역 정규화 (학습 그룹 통계만)
                                                  ▼
   ┌──────────────── 매 학습 스텝 ────────────────┐
   │  xb (B, L, in_dim)                            │
   │    │                                          │
   │    ├─ enc(xb) → z (B, L, 64)  [앵커 쪽]       │
   │    │     └─ 앵커 za = z[b, a]                 │
   │    │           └─ 게이트 g(Δ)를 z_fast에 곱함 │
   │    │                 └─ pred(za_in, log2Δ) → zhat
   │    │                                          │
   │    └─ tenc(x[a+Δ−w : a+Δ]) → ztgt  [타깃 쪽]  │
   │            (0부터 인덱싱, 마지막 위치)         │
   │                                                │
   │  loss = reg: ‖zhat−ztgt‖²  |  nce: InfoNCE     │
   │        + λ·xcov(za_slow, za_fast)              │
   └────────────────────────────────────────────────┘
                                                  ▼
                        평가: 다위치 프로브 · shift eval · 무해성 곡선
```

---

## 1. `model.py` — 인코더와 predictor

### 1.1 상수

```python
P, L = 8, 256                      # 합성 기준 패치길이·시퀀스길이
D_MODEL, D_Z, D_SLOW = 96, 64, 16  # 트렁크 너비 / 출력 임베딩 / slow 블록
```
`D_MODEL=96`, `D_Z=64`는 최초 커밋에서 근거 없이 정해진 값이라 스윕 대상입니다(프로토콜 D절). `D_SLOW=16`은 v1의 폭 스윕("분리는 slow 블록의 **절대** 크기를 따라간다") 근거가 있어 기준값입니다.

### 1.2 `Encoder.forward` — 가변 길이 처리가 핵심

```python
def forward(self, x):                    # x: (B, T, in_dim), T <= L
    T = x.shape[1]
    h = self.embed(x) + self.pos[:, :T]  # 위치 임베딩을 앞에서 T개만 슬라이스
    h = self.tf(h, mask=self.mask[:T, :T])
    z = self.out(h)
    zs = F.layer_norm(z[..., :D_SLOW], (D_SLOW,))          # B2 블록별 LN
    zf = F.layer_norm(z[..., D_SLOW:], (D_Z - D_SLOW,))
    return torch.cat([zs, zf], -1)
```

**세 가지 설계 포인트:**

1. **`self.pos[:, :T]` — 이게 B5의 "조각을 0부터 인덱싱"을 구현합니다.** 길이 w짜리 조각을 넣으면 `pos[0..w−1]`이 붙습니다. causal 인코더에서 `pos[k]`는 "과거를 k개 가진 위치"와 짝지어 학습되므로, 조각의 마지막 토큰(과거 w−1개)에 `pos[w−1]`이 붙는 게 의미상 맞습니다. 절대 인덱스(`pos[t+Δ−w..t+Δ]`)를 쓰면 "창 중간 위치인데 문맥은 빈약함"이라는 학습에 없던 조합이 됩니다.
2. **`mask[:T,:T]` 슬라이싱** — causal 마스크도 함께 잘라야 길이가 맞습니다.
3. **블록별 LayerNorm** — `z_slow` 16차원과 `z_fast` 48차원을 **독립적으로** 정규화합니다. v1은 64차원 전체에 한 번 걸어서 z_fast의 분산이 커지면 z_slow가 함께 축소되는 결합이 있었습니다.

> ⚠️ **주의(측정 시 함정):** LayerNorm은 **벡터 내부**를 정규화하므로 각 (배치, 위치) 벡터의 내부 std는 1입니다. 하지만 **표본 간** std는 별개이고, 실제로 reg+ema에서 0.019까지 내려갑니다. 프로브에 `StandardScaler`가 필요한 이유입니다(6절).

### 1.3 `Predictor` — 연속 Δ 조건화 (B3)

```python
self.dmlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 16))
def forward(self, z, log2delta):                 # log2delta: (N, 1)
    return self.net(torch.cat([z, self.dmlp(log2delta)], -1))
```
스칼라 `log2(Δ)`를 **16차원 벡터로 펼쳐서** 넣습니다. 스칼라를 그대로 concat하면 첫 레이어에서 `w·logΔ`라는 rank-1 이동밖에 안 되어 Δ=1과 Δ=128의 질적 차이를 표현하지 못합니다.

---

## 2. `train.py` — 학습 루프 (합성)

### 2.1 B4 샘플링: Δ 먼저, 앵커는 그 Δ에 종속

```python
anchors = rng.integers(min_context, L - dmin, (batch, n_anchor))   # (B, na)
hi      = np.minimum(dmax, L - 1 - anchors)                        # (B, na) 앵커별 최대 Δ
lo      = np.log(dmin)
u       = rng.random((batch, n_anchor, n_delta))
deltas  = np.exp(lo + u * (np.log(hi)[..., None] - lo)).round().astype(np.int64)
deltas  = np.clip(deltas, dmin, hi[..., None])                     # (B, na, nd)
```

**왜 이 순서인가.** 프로토콜 초안은 앵커 상한을 `L − Δ_max`(=128)로 고정했는데, 이는 v1과 동일해서 `pos[128:256]`이 영원히 학습되지 않습니다. 여기서는 앵커를 먼저 뽑되 **그 앵커에서 가능한 최대 Δ(`L−1−anchor`)로 상한을 정해** 로그 균등 샘플링합니다. 결과적으로 앵커가 창 끝(예: 254)까지 갈 수 있고, 그런 앵커는 작은 Δ만 받습니다.

**부작용(문서화 필요):** 앵커 위치와 Δ가 상관됩니다 — 큰 Δ는 앞쪽 앵커에서만 나옵니다. 유한 창의 causal 예측에 내재된 기하이고, 우리 설정(L=256, Δ_max=128)에서는 먼 Δ도 앵커가 ~96개 확보되어 경미합니다.

**평탄화.** `(B, na, nd)` 그리드를 `(N,)`로 펴서 한 번에 처리합니다. `N = 64 × 8 × 4 = 2048`쌍/스텝 (v1은 8쌍이었으니 **256배**).

```python
bi = broadcast(arange(batch))[...].reshape(-1)   # 각 쌍의 윈도우 인덱스
ai = broadcast(anchors)[...].reshape(-1)         # 앵커 위치
di = deltas.reshape(-1)                          # Δ
tp = ai + di                                     # 타깃 위치
```

### 2.2 게이트

```python
g = torch.sigmoid((tau - dT) / W).unsqueeze(-1)          # Δ가 크면 → 0
za_in = torch.cat([za[:, :D_SLOW], za[:, D_SLOW:] * g], -1)
```
Δ > τ면 게이트가 닫혀 z_fast가 predictor에 전달되지 않습니다. `W`(=4.0 패치)가 전이의 부드러움입니다.

### 2.3 B5 타깃 구성

```python
tenc = tgt if use_ema else enc                              # 타깃 인코더 선택
with (torch.no_grad() if use_ema else contextlib.nullcontext()):
    if target_mode == "cumulative":                         # v1 방식(비교용)
        ztgt = tenc(xb)[biT, torch.from_numpy(tp).to(DEV)]
    else:                                                   # v2 수용장 제한
        starts = torch.from_numpy(tp - w).to(DEV)[:, None] + ar_w   # (N, w)
        slices = xb[biT[:, None], starts]                   # (N, w, P)
        ztgt = tenc(slices)[:, -1]
```

**두 축이 독립**입니다: `target_mode`(수용장: 누적/제한)와 `target_enc`(인코더: EMA/online). B5는 전자만 바꿉니다.

**`w_eff = min(w, Δ)`에 대한 구현상 주의.** 프로토콜은 Δ<w인 경우 조각 폭을 Δ로 줄이라고 정했지만, 현재 구현은 `dmin >= w`를 assert로 강제해 **모든 쌍에서 `w_eff == w`가 되도록** 회피합니다. 이유는 배치 처리 때문입니다 — Δ마다 조각 폭이 다르면 `(N, w)` 텐서로 못 묶고 그룹별 forward가 필요합니다. 무해성이 주장되는 영역은 Δ≥w이므로 현재 실험 범위에서는 정확히 결정된 설계와 같습니다. **전체 실행에서 Δ_min을 1로 낮추려면 `w_eff`별 그룹핑 구현이 필요합니다(미구현).**

### 2.4 손실

```python
if loss_kind == "reg":
    loss = ((zhat - ztgt) ** 2).mean()
elif loss_kind == "nce":
    logits = F.normalize(zhat, dim=-1) @ F.normalize(ztgt, dim=-1).T / temp   # (N, N)
    if mask_same_window:
        logits = logits.masked_fill(neg_mask[:len(logits), :len(logits)], -1e4)
    loss = F.cross_entropy(logits, torch.arange(len(logits), device=DEV))

if xcov:
    zc = za - za.mean(0)
    C = (zc[:, :D_SLOW].T @ zc[:, D_SLOW:]) / (len(za) - 1)     # (16, 48)
    loss = loss + lam * (C ** 2).mean()
```

**NCE 세부:**
- `logits`는 (2048, 2048) 행렬 — 대각이 정답(자기 타깃), 나머지가 네거티브.
- **온도 0.1** (v1 계승, 스윕 대상).
- **`-1e4`로 마스킹**(`-inf` 아님) — `-inf`는 행 전체가 마스킹될 경우 NaN을 유발할 수 있어 큰 음수를 씁니다.
- **`neg_mask` 사전 계산:** 윈도우 인덱스 구조(`bi_const`)가 매 스텝 동일하므로 루프 밖에서 한 번만 만듭니다.
  ```python
  bi_const = broadcast(arange(batch))[...].reshape(-1)          # (2048,)
  neg_mask = (bi_const[:,None] == bi_const[None,:]) & ~eye(2048)  # 같은 윈도우 & 자기 아님
  ```

**왜 같은-윈도우 네거티브를 뺐나.** 같은 윈도우의 쌍들은 **느린 인자를 공유**합니다(같은 regime/활동). 이를 네거티브로 쓰면 InfoNCE가 "느린 상태가 같은 표현끼리 구별하라"를 학습하게 되고, 그러려면 **빠른 정보를 써야** 합니다 — 우리 목표의 정반대입니다. 마스킹하면 네거티브가 전부 다른 윈도우(대체로 다른 regime)라서 느린 정보가 판별에 도움이 되고, 그게 z_slow로 몰립니다.
`mask_same_window=False`로 두면 v1의 `cpc` 셀을 재현합니다.

### 2.5 EMA 갱신

```python
if use_ema:                                    # online 타깃이면 건너뜀
    with torch.no_grad():
        for pe, pt in zip(enc.parameters(), tgt.parameters()):
            pt.mul_(ema).add_(pe, alpha=1 - ema)
```
`target_enc="online"`이면 `tenc`가 `enc` 자신이므로 EMA 갱신이 무의미하고, 그래디언트가 양방향으로 흐릅니다(D2의 정의).

---

## 3. `prep_hapt.py` — A2 라벨 정책의 구현

```python
PATCH, L, N_AX = 4, 256, 3
WIN    = PATCH * L          # 1024 샘플 = 20.48 s
STRIDE = (L // 2) * PATCH   # 512 샘플, 50% 겹침
patch_ends = np.arange(L) * PATCH + (PATCH - 1)     # 각 패치의 끝 샘플 오프셋

for s0 in range(0, len(acc) - WIN, STRIDE):         # ← 라벨을 보지 않고 자름
    w    = acc[s0:s0+WIN].reshape(L, PATCH*N_AX)    # (256, 12) 채널 혼합
    ends = s0 + patch_ends
    W.append(w); lab.append(per[ends]); fast.append(mag[ends]); subj.append(user)
```

**세 가지가 프로토콜을 직접 구현합니다:**

1. **라벨 없이 자르기(A2).** 루프에 라벨 필터가 없습니다. VOID·전환 클래스가 섞인 윈도우도 전부 남깁니다. v1은 끝점 라벨이 1–6이 아니면 버렸는데, 그러면 **학습 데이터 선택에 라벨이 개입**해 "학습은 라벨 없이"라는 주장이 훼손됩니다.
2. **위치별 라벨.** `per[ends]`로 **패치마다** 라벨을 저장합니다(`(n, 256)`). 채점은 이 중 1–6인 위치에서만 합니다.
3. **채널 혼합 레이아웃(A4).** `reshape(L, PATCH*N_AX)` → 패치 하나가 `[s0x,s0y,s0z, s1x,s1y,s1z, ...]` 순서로 12차원. **축 `a`의 열은 `a, a+3, a+6, ...`** 이고, 이 인덱싱을 틀린 게 6절의 버그입니다.

**실측 결과:** 윈도우 2104개, 피험자 30명, 라벨 있는 위치 비율 69.1%, **전환/VOID를 포함한 윈도우 94.8%** — 의도한 대로입니다(A2가 이를 허용).

---

## 4. `train_real.py` — 실데이터 학습과 평가

### 4.1 A3 전역 정규화 (학습 그룹 통계만)

```python
def _norm_stats(W, n_ax):
    T = W.shape[-1]; per = T // n_ax                    # per = 패치 내 샘플 수
    for a in range(n_ax):
        cols = [a + n_ax * k for k in range(per)]       # 축 a의 열들
        mu[a] = W[:, :, cols].mean(); sd[a] = W[:, :, cols].std() + 1e-6

mu, sd, per = _norm_stats(Wall[tr], n_ax)               # ← 학습 그룹에서만
Wn = _apply_norm(Wall, mu, sd, per, n_ax)               # 전체에 적용
```
**채널당 하나의 평균·표준편차**를 학습 피험자에서만 구합니다. 축 간 상대 크기(= 중력 방향 = 자세 단서)가 보존되는 게 핵심입니다. 윈도우별 z-score였다면 이게 지워집니다.

### 4.2 C3 그룹 분할

```python
groups  = np.unique(grp)
test_g  = set(rng.permutation(groups)[:max(1, len(groups)//3)].tolist())
is_test = np.array([g in test_g for g in grp])
tr      = np.flatnonzero(~is_test)
```
피험자(HAPT)/환자(PTB-XL) 단위로 1/3을 통째로 held-out. 학습 배치도 `rng.choice(tr, batch)`로 **학습 그룹에서만** 뽑습니다 — 사전학습 단계에서도 테스트 그룹을 보지 않습니다.

### 4.3 C1 다위치 프로브

```python
Z = torch.cat([enc(Wt[i:i+128]) for i in range(0, len(Wt), 128)]).cpu().numpy()
B = Z[:, :, sl]                                    # 블록 슬라이스
labeled = (lab >= 1) & (lab <= 6)

def gather(idx):
    m = labeled[idx]
    return B[idx][m], lab[idx][m] - 1, fast[idx][m]   # (위치별로 평탄화)

ftr, ytr, ztr = gather(res["tr"]); fte, yte, zte = gather(res["te"])
if max_samples:                                    # 위치들이 상관돼 있어 서브샘플
    ...
clf  = make_pipeline(StandardScaler(), LogisticRegression(...)).fit(ftr, ytr)
f1   = f1_score(yte, clf.predict(fte), average="macro")
leak = make_pipeline(StandardScaler(), Ridge()).fit(ftr, ztr).score(fte, zte)
```

**`B[idx][m]`가 핵심 한 줄입니다** — `(윈도우, 위치)` 2차원을 라벨 마스크로 걸러 `(표본, 특징)`으로 평탄화합니다. 즉 **모든 라벨된 위치가 각각 하나의 평가 표본**이 됩니다. HAPT에서 학습 표본이 ~37만개까지 나와 sklearn이 2~3분 걸렸고, 매트릭스 실행을 위해 6만개로 서브샘플링했습니다.

> ⚠️ **아직 안 한 것:** 프로토콜 C1은 "오차 막대는 **창 단위로 묶어라**"고 정했습니다(같은 창의 위치들은 상관). 현재는 시드 간 편차만 보고하고 창 단위 클러스터 부트스트랩은 미구현입니다. 논문 수치에는 반영 필요.

### 4.4 shift eval

```python
clf = fit on CLEAN train                          # 깨끗한 데이터로 프로브 학습
for s in strengths:
    Wp = Wte + s*randn(...)   or   Wte * (1+s)    # held-out 테스트만 교란
    f1s.append(f1_score(yte, clf.predict(enc(Wp)[..., sl])))
```
프로토콜대로 **프로브는 깨끗한 데이터에 적합하고, 교란된 held-out에 적용**합니다. 교란은 정규화된 입력 공간에서 가합니다(배포 시 정규화 통계는 이미 정해져 있으므로).
`StandardScaler`가 파이프라인 안에 있어 **깨끗한 데이터의 통계로 스케일링**되고 교란 데이터에 그대로 적용됩니다 — 이것도 배포 상황과 일치합니다.

---

## 5. `pilot_b5.py` — 무해성 곡선 (E4)

핵심은 **모델의 블록 구조를 전혀 쓰지 않는다**는 점입니다. 무해성은 데이터와 타깃 정의의 성질이지 우리 모델의 성질이 아니기 때문입니다.

```python
sa, ua = factor_at(starts[wi], s, u, ap)     # 앵커 시점의 정답 (regime, u)
Xs  = np.eye(3)[sa]                          # 느린 인자만 (원-핫)
Xsu = np.concatenate([Xs, ua[:, None]], 1)   # + 빠른 인자
r2  = lambda X: Ridge().fit(X[:m], Z[:m]).score(X[m:], Z[m:])
curve[D] = r2(Xsu) - r2(Xs)                  # 빠른 인자가 추가로 벌어준 몫
```
`Z`는 타깃 임베딩(Δ만큼 뒤 위치)입니다. 즉 **"앵커 시점의 빠른 인자를 알면 Δ 뒤 타깃을 더 잘 맞히는가"**를 Δ별로 잽니다. 0으로 수렴하면 무해성 성립.

**측정 결과(파일럿):** 제한 타깃은 Δ=8에서 0.10 → Δ=16부터 0에 평평. 누적 타깃은 모든 Δ에서 양수(0.024→0.008). 예측대로입니다.

> 초기 설계 오류 기록: 처음엔 "게이트 없이 학습한 모델의 z_fast를 0으로 만들어 오차 증가를 본다"였는데, 게이트가 없으면 앞 16차원을 "느리다"고 부를 근거가 없어 무의미했습니다. 정답 인자를 직접 쓰는 현재 설계로 교체했습니다.

---

## 6. 버그 상세

### 6.1 정규화 축 인덱싱 (크래시)

```python
cols = [a + per * k for k in range(per)]     # ✗ 잘못
cols = [a + n_ax * k for k in range(per)]    # ✓ 맞음
```
채널 혼합 레이아웃이 `[s0x,s0y,s0z, s1x,s1y,s1z, ...]`이므로 축 `a`의 열은 **`n_ax` 간격**입니다. `per`(=4) 간격으로 잘못 잡아 `IndexError: index 12 is out of bounds`로 즉시 크래시했고, 결과에는 영향이 없었습니다.

### 6.2 가변 윈도우 길이

앵커 샘플링이 모듈 상수 `L=256`을 참조해 PTB-XL(L=100)에서 범위를 벗어났습니다. 데이터의 실제 길이 `Lw = Wall.shape[1]`을 쓰도록 수정.

### 6.3 프로브 표준화 (위생) — 그리고 그것이 결론을 바꾸지 않았다는 검증

`reg+ema`의 z_slow **표본 간** std가 0.019로 매우 작아, 기본 `C=1.0`인 L2 규제 LogisticRegression이 가중치를 충분히 키우지 못해 성능이 과소평가될 수 있다고 의심했습니다.

**검증:** 모든 프로브에 `StandardScaler`를 넣고 재실행 → reg+ema slow-kept **0.507 → 0.535**. 거의 그대로였습니다. 즉 5절의 스템 역전은 **측정 아티팩트가 아닙니다.** 수정 자체는 정당한 위생 조치라 유지하고 실데이터도 전부 재실행했습니다(HAPT 0.811→0.809, PTB-XL 0.695→0.697 — 안정).

---

## 7. 미구현·기술 부채

| 항목 | 현재 상태 | 영향 |
|---|---|---|
| `w_eff = min(w, Δ)` 그룹핑 | `dmin >= w` assert로 회피 | Δ<w(게이트 열림) 영역 미실험. 전체 실행 시 필요 |
| 창 단위 오차 막대 (C1) | 시드 간 편차만 보고 | 논문 수치에 반영 필요 |
| E1 τ 추정 | 축별 raw가 HAPT에서 깨짐 | 고역통과/magnitude로 수정 필요 |
| MLP 프로브 · MINE (C4) | 미구현 | 선형 프로브만으로는 배제를 과대평가 |
| chance 기준 명시 (C4) | 코드에 없음 | 논문 표에 추가 필요 |
| 합성 데이터 디스크 영속화 | 해시는 계산하나 파일 저장은 선택적 | run JSON에 해시 기록 자동화 필요 |
| XJTU v2 | 미실행 | 예측된 음성 사례 — 우선순위 낮음 |

---

## 8. 재현 방법

```bash
cd src

# 1) 자체 점검
python3 datagen.py            # 해시·ACF·결정론
python3 model.py              # 텐서 shape·블록별 LN

# 2) B5 게이트 파일럿 (약 1분)
python3 pilot_b5.py           # → runs_v2/pilot_b5.{json,png}

# 3) 실데이터 전처리 (HAPT만; PTB-XL은 기존 npz에서 변환)
python3 prep_hapt.py          # → data/hapt_v2.npz

# 4) 단일 실행
python3 train_real.py         # HAPT: 분리 + shift  (약 5분)
python3 run_ptbxl.py          # PTB-XL: 분리 + shift (약 30초)

# 5) 스템 매트릭스 (3스템 × 3시드)
python3 matrix_v2.py synth    # 약 5분
python3 matrix_v2.py ptbxl    # 약 4분
python3 matrix_v2.py hapt     # 약 6분
```
전부 H200 기준. 결과는 `runs_v2/`에 JSON·PNG로 떨어집니다.
