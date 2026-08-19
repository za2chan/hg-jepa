# 논문 개요 — 2026-08-08

핸드오프(`WRITING_HANDOFF_2026-08-07_ko.md`)가 **무엇을 쓸지**를 정했다면, 이 문서는 **어떤 순서로 몇 쪽에 쓸지**를 정합니다. 수치는 핸드오프에서 가져옵니다.

**읽고 고칠 것**: 아래 영어 문장들(초록·기여·각 Results의 "배운 것")이 프레이밍의 전부입니다. 이것만 승인하면 나머지는 채우는 일입니다.

---

## 쪽수 배분 (IEEE 2단, 8쪽)

| 절 | 쪽 | 그림·표 |
|---|---|---|
| Abstract | 0.15 | — |
| I. Introduction | **1.0** | — |
| II. Related work | **0.75** | — |
| III. Data and methods | **2.0** | `fig_arch_B0`, `fig_gate_B2`, 데이터셋 표, 하이퍼파라미터 표 |
| IV. Results | **3.0** | `fig_ablation_2x2`, `fig_rotation`, `fig_cost` (난이도 곡선 `fig_difficulty`는 부록으로) |
| V. Conclusions | **0.5** | — |
| References | 0.6 | — |
| **부록** | **남는 만큼** | `fig_sensitivity`, SSL 표 |

**부록이 조절 밸브입니다.** 넘치면 부록부터 자릅니다(아래 "자를 순서").

---

## 제목 (확정)

**Horizon-Gated Latent Prediction: Disentangling Persistent Factors in Time Series**

⚠️ `Disentangling`은 Locatello 계열 문헌의 용어라, Related Works 2.3에서 그 문헌을 인용하는 만큼 리뷰어가 식별 가능성·완전 인수분해 기준을 들이댈 수 있습니다. 우리는 그 기준을 충족하지 않습니다. **초록 첫 몇 문장이 범위를 잡아주므로 그대로 갑니다.** 대안은 `Supervising Factor Separation with the Prediction Horizon`.

## Abstract (확정 — 2026-08-09)

> Time-series representation learning has advanced considerably, but a single series mixes factors acting on different timescales, which makes the embedding hard to interpret and use. Work that separates them at the embedding level has mostly addressed static factors, those that do not change over an entire sequence, because invariance by definition supplies a training signal. Persistent factors that vary slowly without being constant get no such free constraint, and they need an axis that fixes where the boundary between slow and fast lies. That axis is already present in prediction-based representation learning: the horizon sets how far into the future the model must predict, and therefore how long information must persist in order to be useful. We propose Horizon-Gated Latent Prediction (HGLP), which places a single threshold on that axis. Beyond it the gate forces the predictor to use only a designated subspace, z_slow, while a cross-covariance penalty between the blocks holds the two apart. A predictor cannot foresee upcoming transitions, so the only way to lower error at long horizons is to record the current state. We also propose a separating index (SEP) that scores how far apart the two blocks are. On synthetic and real data in which persistent and transient factors coexist, we confirm that the transient factor can be separated out of z_slow. Against post-hoc rotations of an ungated embedding, we lead on the synthetic data but trail Slow Feature Analysis on the real ones. We further quantify the downstream trade-off that the separated embedding incurs, and report that it gives up downstream accuracy while degrading less in a new domain when labels are scarce.

**269단어(0.29쪽), 엠대시 없음.** 기존 초안을 폐기하고 다시 썼습니다. 폐기 이유는 **금지된 주장 두 개**였습니다.

| 폐기된 표현 | 근거 |
|---|---|
| *"which **pushes persistent factors into** that block"* | `CLAUDE.md` §4 철회 목록 — "모은다" 금지. 무작위 16차원과 −0.018~+0.025 차이 |
| *"**comparable** on real data"* | 핸드오프 §3 — "실데이터에서 대등"으로 쓰면 안 됨. 1승 3패 |

**집필 시 건드리면 안 되는 세 곳**

1. *"the only way to lower error at long horizons is to **record the current state**"* — `record the persistent information`으로 바꾸면 결론을 전제로 되돌리는 동어반복이 됩니다(`CLAUDE.md` §5: "a memo about NOW, not a guess about the future").
2. *"**trail** Slow Feature Analysis on the real ones"* — `comparable`·`on par` 금지. 실데이터 1승 3패입니다.
3. *"**degrading less** in a new domain"* — `robust`·`더 낫다` 금지. 핸드오프 §4: 사전 등록 30조건 0/30이고, 최대 교란에서도 무메커니즘이 절대 성능에서 앞섭니다. "덜 열화한다"까지만.

**의도적으로 뺀 것**: SEP 수치(0.849/0.388), 6/6 1위, 데이터셋 이름, 손실 두 종, τ 기호, 절대 성능이 역전되지 않는다는 단서(Results·Conclusions에서 다룸).

---

## I. Introduction (1쪽, 7문단)

**확정 본문은 `docs/INTRO_2026-08-08_ko.md`(개정 4)에 있습니다.** 아래는 문단별 요지 — 본문을 고칠 때는 그 파일이 정본이고 이 목록은 구조 확인용입니다.

**P1 — 문제.** 예측 목적함수 위의 시계열 표현학습(van den Oord et al., 2018; LeCun, 2022; Assran et al., 2023). 서로 다른 시간척도의 요인이 뒤섞여 해석·활용이 어렵다. 정적 요인 분리는 성과가 있었으나(Hsu et al., 2017; Li & Mandt, 2018) **서서히 변하되 상수는 아닌 요인**에는 기댈 정의가 없다.

**P2 — 두 읽기. "느린 것"이 신호를 나누는 것(Cleveland et al., 1990; Huang et al., 1998)인지 요인을 나누는 것(Hyvärinen & Morioka, 2016)인지. 본 연구는 요인 쪽. 두 읽기는 반드시 일치하지 않는다.**

**P3 — 어긋남은 한 방향뿐. 이 문단이 가장 중요합니다.** 느린 신호는 지속적이다 (저주파 = 긴 자기상관). **역은 거짓** — 저역통과는 선형 연산이고 요인은 주파수· 분산 같은 비선형 통계에 실릴 수 있다. Fig. 1: 지속 요인이 주파수를 정하면 저주파 판독은 일시 요인만($R^2{=}0.90$), 지속 요인은 안 나온다($R^2{<}10^{-5}$).

**P4 — 두 읽기가 이어지는 관계.** 요인이 지속되는 동안 신호의 *통계*가 고정된다. 느린 것은 신호가 아니라 신호의 통계. 그 함수는 국소 진동 주파수(Fig. 1b), 일시 요인 수명보다 긴 창으로 적분할 때만 성립(Fig. 1c). 지속성=느림 가정(Pavliotis & Stuart, 2008)·SFA(Wiskott & Sejnowski, 2002)는 이 관계의 다른 표현이고, **합성 데이터는 그 일치를 의도적으로 깬다.**

**P5 — 정리 + 정의.** 대역 위치는 지속성을 한쪽으로만 말한다: **느린 대역이면 지속적이지만, 지속적이라고 느린 대역에 있지는 않다.** 선택 기준은 지속성. 논문에서 "지속적" = $\Delta$ 뒤 예측에도 기여함.

**P6 — 착상과 방법.** 지평은 예측 학습에 이미 있는 축(Bialek et al., 2001; Creutzig & Sprekeler, 2008). 그 지평에 게이트를 걸어 먼 지평엔 지정 블록만 쓰게 하는 **HGLP**. Fig. 2가 목표 상태. **주장은 "z_slow에 모인다"가 아니라 "z_slow에서 일시 요인이 비워진다"** (회전 대칭이므로).

**P7 — 기여 셋 + 발견 요약.**

> 기여: (i) 지평 게이트로 지속 요인을 지정 가능한 블록으로 가르는 귀납 편향. (ii) 블록×요인 행렬 평가와 요약 지표 SEP. (iii) 합성·PTB-XL·HAPT 실측.

발견 요약(양면적): 합성 우세(SEP 0.849 vs SFA 0.388) · 실제는 SFA에 뒤짐(넷 중 셋, 폭 0.032–0.072, 지속성=느림 일치의 귀결) · 분리 비용(HAPT macro-F1 −0.111).

**⚠️ 기저선 변동(지속성 ≠ 관련성) 문장은 인트로에서 뺐습니다 → Conclusions P2.5 한계로 이동.**

---

## II. Related work (0.75쪽, 5개 범주)

각 범주 = 2~3문장 + **우리가 다루는 한계 한 문장**.

1. **Latent prediction** — CPC, HEPA, LeNEPA. 타깃 형태 차이는 `CLAUDE.md` §5. 한계: horizon 집합을 고르는 절차가 문헌에 없다.
2. **Slow feature learning** — SFA [Wiskott & Sejnowski 2002], 느림순 ICA. **직접 경쟁자로 위치.** 한계: 학습 후 전체 시계열을 다시 훑어 회전을 적합해야 하고 도메인마다 재적합. **품질이 아니라 운용 비용의 차이라고 정직하게 쓸 것.**
3. **Variance-ordered** — PCA. 한계: 분산 순위 ≠ 지속성 순위 (합성 SEP 0.000).
4. **Time-series SSL** — TS2Vec, PatchTST. 한계: 블록 구조를 만들지 않는다.
5. **Model selection in disentanglement** — Locatello et al. 2019. **우리 하이퍼파라미터에 규칙이 없다는 사실의 자리.**

---

## III. Data and methods (2쪽)

**식은 7개만 씁니다.** 핸드오프 §6에 20개가 넘게 있는데 전부 실으면 읽히지 않습니다. **그림 B0·B2가 이미 담고 있는 것은 식으로 다시 쓰지 않습니다** — 블록 분할, 게이트의 부드러움, EMA·stop-grad, xcov가 게이트 이전의 $z_t$에 걸린다는 것, 손실의 전체 구조.

⚠️ **표기 통일**: `fig_gate_B2.png`가 게이트 폭을 $W$로 쓰는데 타깃 창 $w$와 지면에서 구분되지 않습니다. **$\kappa$로 통일**하고 그림을 고치십시오(핸드오프 §6.4).

---

### A. Datasets and factors (0.4쪽, 표 1개, 식 1개)

데이터셋 표 — 창 길이, 패치, 채널, **지속 요인**, **일시 요인 대리변수**, 우연 수준, $n$. **세 개뿐: 합성 ±3% · PTB-XL · HAPT.**

**식 (1) — 합성 생성기.** 이것만은 반드시 식으로. Introduction의 (A)/(B) 구분이 여기서 검증되기 때문입니다.

$$\phi_k = \textstyle\sum_{j\le k} 2\pi f_{s_j}\big(1+\alpha u_j\big), \qquad x_k = A_{s_k}\sin\phi_k + \beta u_k + \varepsilon_k$$

한 문장: **지속 요인 $s$는 캐리어의 주파수 $f_s$를 정하고, 일시 요인 $u$는 OU 과정으로 순간 주파수와 진폭을 함께 흔든다. 따라서 $s$는 신호의 느린 성분으로 나타나지 않는다.**

**Fig. 1을 여기서 다시 인용한다(새 그림 없음).** Fig. 1은 1쪽(Introduction)에 통째로 싣고, Data 절은 같은 그림을 재인용합니다 — 세 패널이 한 논증이라 쪼갤 수 없고, 패널을 나눠 실으면 (c)가 고아가 됩니다. **인트로는 관찰, Data는 이유**로 나눕니다. 인트로가 "넓은 창으로 주파수를 읽으면 지속 요인이 드러난다"까지 말했으니, 여기서는 식 (1) 바로 아래에 **왜 창이 넓어야 하는지**를 씁니다: 생성기 정의상 일시 요인 $u$가 순간 주파수를 함께 변조하므로, 창이 $u$의 수명(50스텝)보다 수십 배 넓어야 그 변조가 평균되어 사라진다.

**어느 조건을 깨는지도 여기서 명시한다.** 지속성과 느림이 일치하려면 ① 빠른 성분이 충분히 빨리 섞이고 ② 관심 요인이 신호에 더해지는 형태여야 한다. **우리 합성 데이터는 ①을 만족하고 ②만 깬다** — 일시 요인의 자기상관 반감기는 51스텝, 지속 요인은 1859스텝으로 척도 분리가 성립한다. ⛔ "우리 데이터는 척도 분리가 안 된다"고 쓰면 **틀린다.** (참고: ①을 깨는 사례는 coherent-periodic 성분이고 PTB-XL이 그 경우 — 적용 조건 (i).)

**여기서 두 주장을 반드시 구분해서 쓴다(곡선 없이 산문, 핸드오프 §3).** ⛔ **"FFT로는 풀 수 없다"고 쓰면 안 된다** — ±3%에서 훈련 없는 FFT 분류기가 0.660(우연 0.333)이다.

1. **선형 대역 분해로는 접근 불가.** 지속 요인이 진동의 주파수를 정하므로, 신호를 저역과 고역으로 가르는 연산으로는 복원되지 않는다($R^2 < 10^{-5}$, Fig. 1). **느린 대역에 있으면 지속적이지만, 지속적이라고 해서 느린 대역에 있지는 않다** — 이 데이터셋이 그 두 번째 경우다.
2. **스펙트럼 판독기로는 부분적으로 가능하며, 그 정도가 난이도 눈금이다.** FFT 파워는 **비선형 통계**라 주파수를 볼 수 있다. 훈련 없이 0.660에 이르므로 과제가 자명하지 않다. ±5%는 학습 전에 이미 0.784라 포함 항의 증거력이 약하고, ±1%는 0.459로 우연 근처라 포함 항이 해석되지 않는다 → **±3%를 본문 설정으로 쓴다.**

**FFT는 베이스라인 표에 넣지 않는다** — 합성에만 있고 채점 프로토콜이 다르다.

산문으로: 난이도는 주파수 간격으로 조절하고 그 눈금을 여기서 한 값(±3%)으로 고정한다. (난이도 실험은 Results에서 Data 절로 옮겼으므로 Results 절을 인용하지 않는다.) HAPT·PTB-XL의 일시 대리변수(가속도 크기 / 순간 전압)는 신호에서 직접 계산되므로 **주석이 아니다.**

### B. Model (0.5쪽, **그림 B0**, 식 2개)

**그림 B0을 전면 배치하고 산문은 그림이 못 말하는 것만.**

산문: 인과 트랜스포머 4층·4헤드·$d_{\text{model}}$=96·RoPE. 토크나이저는 $\mathrm{Linear}(p\cdot c,\,96)$ **하나** — 코드북도 양자화도 없다. 채널은 입력에서 섞이며 **의도된 설계**다(관심 요인이 축간 관계로 정의되는 경우). 예측기는 **MLP**로 시간 구조를 보지 않고 한 시점의 임베딩과 $\log_2\Delta$만 받는다. **어텐션 연산에는 어떤 수정도 없다** — 아키텍처가 아니라 학습 절차의 변경이다.

**식 (2) — 타깃.** v1과의 차이가 여기 있으므로 식으로.

$$z^{*} = \mathrm{sg}\big[\,f_{\bar\theta}\big(x_{[t+\Delta-w,\;t+\Delta]}\big)_{-1}\big], \qquad \bar\theta \leftarrow m\bar\theta + (1-m)\theta$$

한 문장: **타깃의 수용영역이 $(t,\,t+\Delta]$ 안에 들어가 앵커의 과거를 다시 포함하지 않는다.** ($\Delta\ge w$를 강제한다. $w_{\text{eff}}=\min(w,\Delta)$는 쓰지 않는다 — 한 번도 작동하지 않았다.)

**식 (3) — 최종 손실.**

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda\,\mathcal{L}_{\text{xcov}}, \qquad \mathcal{L}_{\text{pred}} \in \{\mathcal{L}_{1},\ \mathcal{L}_{\text{NCE}}\}$$

$\mathcal{L}_1$과 $\mathcal{L}_{\text{NCE}}$는 **식을 쓰지 않고 산문**으로: "$L_1$ 거리" / "코사인 유사도에 대한 InfoNCE, 온도 0.1". **다만 비표준인 한 가지는 강조**해야 합니다 — **같은 창에서 나온 음성쌍을 분모에서 제외한다.** 같은 창은 지속 요인을 공유하므로 음성쌍으로 쓰면 모델에게 그것을 버리라고 가르치게 된다.

### ⚠️ 그림 B0·B2는 한 장으로 합쳤습니다 → `fig_main.png`

`src/exp/fig_schematics.py`의 `fig_main()`이 만듭니다. (a) 아키텍처 전체(옛 B0) + (b)(c) 짧은/긴 지평 + (d) 게이트 곡선(옛 B2). B2가 다시 그리던 인코더·예측기·타깃 상자 6개는 (a)와 중복이라 뺐고, B2의 상단 제목 줄과 우하단 τ 설명 문단은 **캡션으로 옮겼습니다**. 양단폭 배치 시 **0.57쪽** — 둘을 따로 읽히는 크기로 싣던 0.87쪽보다 0.30쪽 적습니다. `fig_arch_B0.png`·`fig_gate_B2.png`도 계속 생성되므로 부록에 쓸 수 있습니다.

**확정 캡션 (그대로 옮길 것)**

> **Fig. 3. Horizon-gated latent prediction.** **(a)** A causal encoder maps a window to an embedding $z_t$, split into a persistent block $z_{slow}$ (16 of 64 dimensions) and the rest $z_{mix}$. A predictor conditioned on the horizon $\Delta$ must match a target read at one position inside $(t,\,t+\Delta]$, produced by an EMA copy of the encoder. A cross-covariance penalty is applied to $z_t$ before the gate. **(b, c)** The same pipeline at a short and a long horizon. $z_{slow}$ is never gated; only $z_{mix}$ is, and only past $\tau$, so at long horizons the only route to lowering the loss runs through $z_{slow}$. **(d)** The gate is a sigmoid of width $\kappa$, not a switch. $\tau$ is set from the transient factor's lifetime and $\Delta_{max}$ is set separately, so the axis is drawn in units of $\tau$ rather than absolute horizons.

⚠️ **캡션에 "게이트가 지속 요인을 z_slow에 모은다"는 뜻이 들어가지 않게 하십시오**(`CLAUDE.md` §4 철회 목록). 위 문구는 "먼 지평에서 손실을 낮추는 경로가 z_slow뿐"까지만 말하고 멈춥니다.

---

### C. The horizon gate (0.5쪽, **`fig_main` (d) 패널**, 식 2개) — **Method의 핵심**

**식 (4) — 게이트.**

$$g(\Delta) = \sigma\!\Big(\frac{\tau-\Delta}{\kappa}\Big), \qquad \tilde z_t(\Delta) = \big[\,z_t^{\text{slow}}\,;\ g(\Delta)\,z_t^{\text{mix}}\,\big]$$

**식 (5) — 기울기 분할. 이 한 줄이 두 정직성 조항을 서술이 아니라 유도로 만듭니다.**

$$\frac{\partial\mathcal{L}}{\partial z_t^{\text{slow}}} = \frac{\partial\mathcal{L}}{\partial \tilde z_t^{\text{slow}}}, \qquad \frac{\partial\mathcal{L}}{\partial z_t^{\text{mix}}} = g(\Delta)\,\frac{\partial\mathcal{L}}{\partial \tilde z_t^{\text{mix}}}$$

이어지는 두 문단:

**(가) 가용성은 강제된다.** $g\to0$인 먼 지평에서 손실을 낮추는 유일한 경로가 앞쪽 $d_s$차원이므로, 그 지평에서 예측에 기여하는 정보는 $z^{\text{slow}}$에서 읽어낼 수 있어야 한다. 게이트는 용량이 아니라 정보를 제거하므로 예측기를 키워도 복원되지 않는다.

> ⛔ **"모은다"로 쓰지 말 것.** 무작위 16차원과 −0.018~+0.025 차이다. 회전 대칭 때문에 어떤 16차원이든 지속 요인을 담는다.

**(나) 배제는 보장되지 않는다.** $z^{\text{slow}}$는 가까운 지평 예측에도 계속 쓰이고 거기서는 일시 정보가 손실을 낮춘다. 목적함수에 그것을 밖으로 밀어낼 항이 없다. **배제는 xcov가 별도로 공급해야 한다.**

**회전 대칭 문단도 여기.** 잠재 예측 손실은 회전에 근사 불변이고, **우리 세 장치가 정확히 그 대칭을 깨는 항**이다. 가정하지 않고 측정한다(Results §A, 0.015).

### D. Keeping the blocks apart (0.25쪽, 식 1개)

**식 (6) — xcov.**

$$C = \frac{1}{N-1}\bar Z^{\top}_{(1:d_s)}\bar Z_{(d_s+1:D)}, \qquad \mathcal{L}_{\text{xcov}} = \frac{1}{d_s(D-d_s)}\sum_{a,b} C_{ab}^{2}$$

**게이트를 통과하기 전의 $z_t$에 걸린다**(그림 B0의 점선). 블록별 LayerNorm은 **xcov와 한 세트**로 산문 두 문장 — 전체 $D$에 걸친 LayerNorm은 공유 평균·분산으로 두 블록을 다시 묶어 xcov와 싸우기 때문. 따라서 **"메커니즘 없음"은 셋 모두 없음**을 뜻한다.

### E. Measurement (0.3쪽, 식 1개)

**식 (7) — SEP.**

$$\mathrm{SEP} = \underbrace{[s_{\text{per}}(z^{\text{slow}})]_0^1}_{\text{inclusion}} \cdot \underbrace{[s_{\text{tra}}(z^{\text{mix}})]_0^1}_{\text{allocation}} \cdot \underbrace{[1-s_{\text{tra}}(z^{\text{slow}})]_0^1}_{\text{exclusion}}$$

세 항이 각각 막는 가짜를 **우리 표의 실제 셀로** (핸드오프 §2). **분모 없음**과 **두 단서**(배제가 포함을 상쇄 가능 → 포함 열 병기 / 데이터셋 간 비교 금지)를 여기서. 프로브 프로토콜은 산문 — 그룹 분할, 문맥 하한, 마지막 위치 채점, 무작위 분할 통제.

### F. What we chose by hand (0.15쪽, 표 1개)

$\lambda$, $d_s$, $w$, 패치 크기, $\varepsilon$, PTB-XL의 $\tau$. **Related work 5번과 Conclusions의 Limitation 1·2로 연결.**

---

### 산문으로 내리거나 생략할 것 (핸드오프 §6에서)

| 항목 | 처리 |
|---|---|
| 인코더 블록별 LN 식 | **산문 2문장** (그림 B0이 분할을 보임) |
| 예측기 MLP 차원 | **산문 한 구절** |
| $\mathcal{L}_1$ · InfoNCE 식 | **산문.** 단 같은-창 음성쌍 제외는 강조 |
| 앵커·지평 표집 식 | **산문 두 문장.** 로그 균등이라는 것과, **$\Delta_{\min}<\tau$가 아니면 $z^{\text{mix}}$가 기울기를 못 받는다**는 것만 |
| $w_{\text{eff}}=\min(w,\Delta)$ | **생략** (한 번도 작동 안 함) |
| $c_{\text{tr}}$ vs $c_{\text{pr}}$ | **괄호 한 번** (16 vs 64, 이름을 다르게) |
| RoPE 채택 이유 | **각주 한 줄** |
| 학습 절차 7단계 | **생략.** 하이퍼파라미터 표로 대체 |
| L2 · online 어블레이션 | **부록** |

⚠️ **구현대로 쓸 것**: 코드는 앵커를 먼저 뽑고 지평을 앵커에 맞춰 자릅니다. `PROTOCOL.md` B4의 "Δ를 먼저"는 구현과 반대입니다(결과는 같음).

## IV. Results (3쪽) — 각 실험마다 목적·그림·배운 것

### A. ~~How hard is the synthetic task?~~ — **Results에서 삭제, Data §A로 이동** 핸드오프 §3의 지시입니다: 난이도 근거에 그림 한 칸을 쓸 가치가 없으므로 `fig_difficulty`는 본문에서 빼고 Data 절 산문으로 처리합니다(필요하면 부록). 따라서 **Results는 B부터 시작**하고 아래 번호가 하나씩 당겨집니다.

### B. What creates the separation? (`fig_ablation_2x2`) **목적**: 게이트·xcov·둘 다의 기여 분해. **배운 것**:
> The gated model with the penalty is first in all six settings, and neither device alone reaches it. On synthetic with the contrastive stem the gate alone is indistinguishable from no mechanism, yet adding it to the penalty raises SEP from 0.488 to 0.749. In the mechanism-free cell the coordinate split and a random split agree to within 0.015 in all six settings, confirming the rotation symmetry the rest of the comparison rests on.

### C. Can post-processing do the same? (`fig_rotation`) **목적**: 후처리로 대체 가능한가 — **가장 날카로운 반박**. **배운 것**:
> Our method has one knob and the unmixings have none, so we report the whole curve rather than one point. On synthetic the curve passes above every post-hoc point on both axes. On real data it does not: SFA reaches equal inclusion at higher exclusion, and over six settings the verdict is three to three. What remains ours is that the coordinates are fixed during training, so no second fitting stage is needed at deployment — an operational difference, not a difference in separation quality.

### D. What does separation cost, and what does it buy? (`fig_cost`) **목적**: 다운스트림 대가와, 도메인 이동 하의 열화. **배운 것**:
> Against a model trained with no mechanism at all, the gated block costs essentially nothing on synthetic and PTB-XL and up to 0.11 macro-F1 on HAPT — the one dataset whose target factor is not constant within a window. On that same dataset the gated block degrades least when the probe is fitted in a perturbed domain with few labels: with 25 labels it loses 0.11 macro-F1 less than a width-matched block from the mechanism-free model, five times the seed standard deviation. It does not overtake that model in absolute terms at any perturbation strength we tested, so the gate trades accuracy for stability rather than gaining both. The effect is within noise on the other five settings.

### E. Does the target have to be latent? (표 하나, 그림 없음) **목적**: JEPA 계열의 전제 — 원시 신호가 아니라 잠재 공간에서 예측해야 하는가. **배운 것**:
> Replacing the latent target with the raw patch halves SEP on both stems (0.809 to 0.415 and 0.844 to 0.432, each at its own best xcov weight). Both models carry the same penalty, so we compare the whole curve rather than one point and neither side is flattered by the choice. The gap is in inclusion: with a raw target the persistent factor is lost as the penalty grows (0.807 at zero weight down to 0.573), while the latent target holds above 0.97 throughout. The raw signal still contains the transient component, and the pressure to predict it displaces the persistent one.

### F. How much do the hand-chosen values matter? (`fig_sensitivity`, 부록으로 밀어도 됨) **배운 것**:
> The xcov weight changes SEP by up to a factor of two, and the value we fixed in advance is a poor one for one stem on synthetic. The block width does not matter between 8 and 16 and collapses at 32. The horizon threshold rule never selects the worst value in the sweep but is up to 0.219 below the best.

---

## V. Conclusions (0.5쪽)

**P1 — 발견 요약** (2026-08-08 수정). 다음을 담는다:
- **메커니즘 작동**: 게이트와 xcov가 분리를 만들며 어느 쪽도 단독으로는 부족하다(6/6). **그리고 그 분리에는 잠재 타깃이 필요하다** — 원시 신호를 맞히게 하면 SEP이 절반이 된다(포함 항이 무너짐).
- **합성 우세, 실데이터는 못 미침**: 통제된 합성에서는 라벨 없는 사후 회전을 넘지만, **실제 데이터에서는 사후 회전(SFA)에 못 미친다** — 네 실데이터 설정 중 셋에서 진다(전체 3승 3패). ⚠️ 옛 "실제 대등"은 핸드오프 §4 철회 대상이라 삭제함.
- **비용**: 분리에는 다운스트림 비용이 따른다(HAPT macro-F1 −0.111). 게이트는 절대 성능을 안정성과 맞바꾼다.

**P2 — Responsible AI** (`CLAUDE.md` §6, 약 0.25쪽). **2026-08-08 사용자 지정 내용으로 교체(옛 세 기둥 폐기).** 세 항목:
1. **Explainable by design, not after the fact.** 대부분의 XAI는 학습 뒤에 설명을 얹지만, 여기서는 축이 설계에 박혀 있다 — z_slow가 보기도 전에 지속 상태로 예약된다. 단, **구조가 해석 가능한 것이지 인코더가 해석 가능한 것은 아니다.**
2. **Robust where it matters.** HAPT에서 z_slow는 모델이 못 본 피험자에서도 버틴다. 일시 요인은 바뀌어도 지속 상태는 살아남는다. ⚠️ 집필 시 "robust"는 D/§4 단서("덜 열화한다"까지, 절대 성능 추월 못함)를 반드시 동반.
3. **We can price it.** 이 구조를 더하는 데는 비용이 들고, 우리는 그것을 잴 수 있다.

**P2.5 — Limitations** (4개, 이 순서로):

1. **하이퍼파라미터를 라벨 없이 고를 수 없다.** λ · d_slow · 타깃 폭 · 패치 크기에는 규칙이 없고 다운스트림을 보고 정했다. **τ만 부분적 예외**로 자기상관에서 유도되나 coherent-periodic 신호에서 실패한다(PTB-XL: 기록별 추정치가 [0.1, 49.9]패치로 499배 산포, 규칙값 56.6이 학습 Δ 범위 [8,48] 밖). 다만 **배제 항은 과제 라벨 없이 측정된다** — 일시 요인 대리변수는 신호에서 직접 계산되므로, 분리가 실패했는지는 라벨 없이 감지된다.

2. **타깃 폭 `w`를 분석하지 않았다.** 세 데이터셋에서 8 / 12 / 8을 썼는데 `w / T_ac`가 1.53 / 0.91 / 0.42로 일정하지 않다. 프로토콜은 `{1,2,4}×T_ac` 스윕을 계획했으나 실행하지 못했다. `w`는 타깃의 수용장을 정하므로 **일시 요인이 타깃 안에서 얼마나 평균되는지를 직접 좌우한다.**

3. **적용 조건을 깨끗이 만족하는 실제 데이터가 없다.** PTB-XL은 심박이 coherent-periodic이라 조건 (i)을 위반하고 지속 요인이 레코드당 상수여서 추적이 아니다. HAPT는 창의 57%가 활동 전환을 가로질러 지속 요인이 창 안에서 상수가 아니며, 일시 대리변수가 크기라 어느 16차원에도 잘 담기지 않는다 (무작위 분할의 배제가 0.845~0.959). 합성 데이터는 우리가 설계했다.

4. **실용적 이점을 보이지 못했다.** 라벨 효율성은 무메커니즘 임베딩 대비 이득이 없다. 교란 도메인 열화는 **여섯 설정 중 하나**(HAPT 회귀 스템)에서만 오차막대를 넘으며, 그마저 **절대 성능은 어느 교란 강도에서도 추월하지 않는다.** 다운스트림 성능은 분리를 얻기 위해 지불하는 비용이다.

5. **"지속적이지만 느린 대역에 없다"는 존재 주장이고, 근거가 합성 데이터 하나다.** 전칭 명제의 반증에는 반례 하나로 충분하므로 논리적으로는 성립한다. 그러나 **그 반례를 우리가 설계했고**, PTB-XL도 HAPT도 이 경우를 보이지 않는다. 그런 요인이 실제 데이터에서 얼마나 흔한지는 답하지 않았다. 회전기계 진단의 포락선 분석(Randall & Antoni, 2011)이 요인이 변조에 실리는 부류가 실재함을 보여주지만, 그 사례들이 **우리 뜻의 지속성**을 갖는지는 별도로 확인해야 한다. 한계 3(적용 조건을 만족하는 실데이터 부재)과 같은 뿌리다.

6. **우리 축이 고르는 것은 지속성이지 관련성이 아니다.**(인트로에서 옮겨 옴) 지속적인 잡음 요인 — 심전도의 기저선 변동처럼 호흡과 전극 움직임이 만드는 저주파 표류 — 도 같은 기준으로 z_slow에 들어온다. **어떤 지속 요인이 관심 대상인지는 이 방법이 답하지 않는다.** CLAUDE.md §1 적용 조건 (iii)과 맞물린다.

**P3 — 향후 방향** (3개):

- **지속 요인 전용 방법과의 비교** — SlowVAE 등. 우리는 라벨 없는 사후 회전과만 비교했고, 분리를 목적함수에 명시적으로 넣는 방법과는 비교하지 못했다.
- **적용 조건을 만족하는 데이터셋에서의 검증** — 지속 요인이 **기록 안에서 변하면서 창 안에서는 상수인** 데이터. PTB-XL은 전자를, HAPT는 후자를 만족하지 않는다. 수면 단계 분류가 그 성질을 갖는 대표 사례다.
- **`z_slow` 궤적으로 실용성 입증** — 분리된 블록의 시간 궤적을 상태 지표로 쓰는 사용 프로토콜. 한계 4를 직접 겨냥한다.

---

## References (Introduction 인용 — 저자-연도, 원본대조용)

INTRO 본문은 저자-연도로 인용합니다. 아래는 그 매핑 + 최종 IEEE 번호 배정 (번호는 본문 등장 순서, 원본대조 시 이 표로 대조). **`—`는 인트로 밖(Related work·한계)에서만 쓰이므로 인트로 번호 없음.**

| 저자-연도 | 제목 | 출처 |
|---|---|---|
| van den Oord et al., 2018 | Representation Learning with Contrastive Predictive Coding (CPC) | arXiv:1807.03748, 2018 |
| LeCun, 2022 | A Path Towards Autonomous Machine Intelligence | OpenReview 포지션 페이퍼, 2022 |
| Assran et al., 2023 | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) | CVPR 2023 |
| Hsu et al., 2017 | Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data (FHVAE) | NeurIPS 2017 |
| Li & Mandt, 2018 | Disentangled Sequential Autoencoder | ICML 2018 |
| Cleveland et al., 1990 | STL: A Seasonal-Trend Decomposition Procedure Based on Loess | J. Official Statistics, 1990 |
| Huang et al., 1998 | The Empirical Mode Decomposition and the Hilbert Spectrum … | Proc. R. Soc. Lond. A, 1998 |
| Hyvärinen & Morioka, 2016 | Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA | NeurIPS 2016 |
| Pavliotis & Stuart, 2008 | Multiscale Methods: Averaging and Homogenization | Springer, 2008 |
| Wiskott & Sejnowski, 2002 | Slow Feature Analysis: Unsupervised Learning of Invariances | Neural Computation, 2002 |
| Bialek et al., 2001 | Predictability, Complexity, and Learning | Neural Computation, 2001 |
| Creutzig & Sprekeler, 2008 | Predictive Coding and the Slowness Principle | Neural Computation, 2008 |
| Locatello et al., 2019 | Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations | ICML 2019 (Related work·한계 절용, 인트로 밖) |
| Ennadir et al., 2025 | Joint Embeddings Go Temporal (TS-JEPA) | arXiv:2509.25449, 2025 (확인 완료, 인트로 밖) |

**인트로 등장 순서**: van den Oord 2018 → LeCun 2022 → Assran 2023 → Hsu 2017 → Li & Mandt 2018 → Cleveland 1990 → Huang 1998 → Hyvärinen & Morioka 2016 → Pavliotis & Stuart 2008 → Wiskott & Sejnowski 2002 → Bialek 2001 → Creutzig & Sprekeler 2008.

### Related Works 추가 인용 (원문 대조 완료분)

⚠️ **서지 정본은 `docs/CITATION_CHECK_2026-08-08_ko.md`입니다.** 아래는 인트로 밖에서 새로 들어온 것만 적은 요약이고, 전체 목록과 대조 근거는 그 파일에 있습니다. 두 곳을 다 고치지 말고 그 파일만 고치십시오.

**2026-08-08 전건 대조 완료**: TS2Vec(Yue et al., 2022) · TNC(Tonekaboni et al., 2021) · HEPA(Petersen et al., 2026) · LeNEPA(Chemeris et al., 2026) · SlowVAE(Klindt et al., 2021) · β-VAE(Higgins et al., 2017) · PatchTST(Nie et al., 2023) · Khinchin 1934 · Randall & Antoni 2011 · PFA(Richthofer & Wiskott, 2013) · Blaschke et al. 2007.

| 저자-연도 | 제목 | 출처 | 상태 |
|---|---|---|---|
| Richthofer & Wiskott, 2013 | Predictable Feature Analysis (PFA) | arXiv:1311.2503, 2013 | ✅ 전문 대조 |
| Blaschke et al., 2007 | Independent Slow Feature Analysis and Nonlinear Blind Source Separation | Neural Computation 19(4):994–1021, 2007 | ✅ 서지 확보 (아웃라인의 "느림순 ICA") |

⚠️ **혼동 주의**: Creutzig & Sprekeler **2008** (Neural Computation 20(4):1026–1041)과 Creutzig, Globerson & Tishby **2009** (Past-future information bottleneck, Phys. Rev. E 79:041925)은 **다른 논문**입니다. 우리가 인용하는 것은 2008입니다.

⚠️ **PFA에 "predictable ≠ slow" 주장이 없습니다.** 원문은 기준을 느림에서 예측 가능성으로 바꿀 뿐입니다. 그 대조는 **우리 관찰**이므로 PFA에 귀속시키지 마십시오.

---

## 분량 초과 시 자를 순서

1. 부록의 SSL 위생 점검 표 (본문에서 인용 안 하므로 통째로 삭제 가능)
2. `fig_sensitivity` → 표 하나로 축약
3. 부록 A(민감도, 옛 F)를 Conclusions의 한 문단으로 흡수. **Results §B(타깃 공간, 옛 E)는 남길 것** — 설계 선택의 유일한 명확한 근거다
4. Related work 4번(TS2Vec/PatchTST)을 1번에 흡수
5. 데이터셋 표의 열 축소

**자르면 안 되는 것**: `fig_rotation`, `fig_ablation_2x2`, 회전 대칭 문단, 하이퍼파라미터 표(III-F), 초록의 마지막 세 문장.

---

## 집필 세션에 줄 순서

| 턴 | 내용 |
|---|---|
| 1 | Abstract · I · II |
| 2 | III (A~F) |
| 3 | IV A~C |
| 4 | IV D~F · V |
| 5 | 부록 · References · 8쪽 맞추기 |
| 6 | **별도 세션**: 철회 목록 대조 검수 |
