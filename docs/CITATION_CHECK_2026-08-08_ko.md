# 인용 원문 대조 기록 — 2026-08-08

`INTRO_2026-08-08_ko.md`와 `RELATED_2026-08-08_ko.md`가 인용한 문헌을 원문과 대조한 기록. `CLAUDE.md` §4-3(손으로 옮긴 주장 금지)의 문헌판이다. **상태**: ✅ 확인 / ⚠️ 수정함 / ⬜ 미확인.

---

## ⚠️ Richthofer & Wiskott 2013 — PFA (원문 대조 완료, 우리 서술을 고침)

**서지 확정.** Stefan Richthofer, Laurenz Wiskott, "Predictable Feature Analysis", arXiv:1311.2503v1 [cs.LG], 2013년 11월 11일 제출. 소속 Institute for Neural Computation, Ruhr-Universität Bochum. **arXiv 페이지에 학술지 게재 정보 없음** — 별도 게재본이 있는지는 미확인이므로 arXiv 프리프린트로 인용한다.

**우리가 처음 쓴 문장이 틀렸다.** 초안은 *"예측 가능한 특징 분석은 예측 가능성과 느림이 같지 않다는 점을 이미 지적했다"*였는데, **원문에 그런 주장이 없다.** PFA는 두 기준이 다르다는 명제를 세우지 않고, 기준을 갈아끼울 뿐이다. 원문 §1의 정확한 문장:

> "PFA extracts sub-signals from the input using the same methods like SFA does, but instead of the slowest features, it selects those that are best predictable by a certain prediction model."

**PFA가 실제로 하는 일** (§2.2): 예측 가능성을 **선형 자기회귀 모형**으로 잰다. 신호가 잘 예측된다는 것은 각 값이 최근 $p$개 값의 선형 결합으로 근사된다는 뜻이다 — $a^Tz(t) \approx b_1a^Tz(t-1) + \dots + b_pa^Tz(t-p)$. 추출 방식(구면화 + 고유벡터)과 제약(영평균·단위분산·쌍별 무상관)은 SFA와 같다.

⚠️ **주의 — 과장하지 말 것.** PFA의 이력 정의 $\mathrm{hist}_{z,p,\Delta}$에 $\Delta$가 있다. 다만 이것은 **이력 표본의 간격**이고 기본값 1이며, **예측 지평이 아니다.** 예측 대상은 여전히 다음 한 걸음이다. "PFA에는 지평 개념이 없다"고 단정하지 말고 **"예측 대상이 한 걸음 뒤"**라고 쓸 것.

**따라서 우리 차별점은 세 가지로 정리된다** (모두 원문 대조로 뒷받침됨): ① 예측 대상이 한 걸음 뒤가 아니라 **조절 가능한 지평**이고 우리가 쓰는 구간은 먼 지평이다 ② 그 지평을 **좌표 블록에 대응**시킨다 ③ 블록이 학습 중 고정되어 사후 재적합이 없다.

**부수 소득 — "predictable ≠ slow"는 선행 연구가 아니라 우리 관찰이다.** 초록·서론 어디에도 그 대조가 없으므로, 인트로 P3의 논거를 남에게 귀속시키지 않아도 된다. 다만 PFA를 **기준을 예측 가능성으로 바꾼 선행 연구**로는 반드시 세워야 한다.

---

## ✅ PFA 참고문헌에서 함께 검증된 서지 4건

PFA 논문의 참고문헌 목록은 Wiskott 본인이 작성한 것이므로 신뢰할 수 있는 2차 출처다.

| 우리 인용 | PFA 참고문헌의 서지 | 상태 |
|---|---|---|
| Bialek et al., 2001 | [1] W. Bialek, I. Nemenman, N. Tishby. "Predictability, complexity, and learning." *Neural Computation*, 13:2409–2463, Nov 2001. | ✅ 일치 |
| Creutzig & Sprekeler, 2008 | [8] F. Creutzig, H. Sprekeler. "Predictive coding and the slowness principle: An information-theoretic approach." *Neural Computation*, 20(4):1026–1041, 2008. | ✅ **연도 2008 확정** |
| Wiskott & Sejnowski, 2002 (SFA) | [13]은 Scholarpedia 항목(2011). 2002 원논문은 PFA가 직접 인용하지 않음 | ⬜ 원논문 별도 확인 필요 |
| 느림순 ICA (아웃라인 Related work 2번) | [6] T. Blaschke, T. Zito, L. Wiskott. "Independent slow feature analysis and nonlinear blind source separation." *Neural Computation*, 19(4):994–1021, 2007. | ✅ **정식 서지 확보** |

⚠️ **2008 vs 2009 혼동의 원인을 찾았다.** PFA [7]은 **다른 논문**이다 — F. Creutzig, A. Globerson, N. Tishby. "Past-future information bottleneck in dynamical systems." *Physical Review E*, 79:041925, **2009**. 우리가 인용하는 것은 [8](2008, Neural Computation)이다. 두 편을 섞지 말 것.

---

## ✅ SFA의 비선형 확장 — 우리 서술 검증됨

`RELATED` 2.1(b)에 *"SFA는 입력을 다항식으로 확장해 쓰므로 비선형 통계에 접근할 수 있다"*고 썼다. PFA §2.1(Wiskott 본인 서술)이 이를 확인한다:

> "To make the method more powerful, a non-linear expansion $h$ can be applied to the signal – usually using monomials of low degree."

또한 우리가 쓴 *"단위 분산 제약 아래 출력의 시간 미분을 최소화"*도 §2.1의 최적화 문제(목적함수 $a_i^T\langle \dot z\dot z^T\rangle a_i$, 제약 영평균·단위분산·쌍별 무상관)와 일치한다. **단, 2002년 원논문으로 재확인하는 편이 낫다** — 지금 근거는 2차 출처다.

---

## 남은 대조 대상

### 확정 서지, 주장 확인 필요
- ⬜ Wiskott & Sejnowski 2002 — 다항 확장·제약 서술 (위 2차 출처로 잠정 확인)
- ⬜ Hyvärinen & Morioka 2016 (TCL) — "비정상성 아래 요인 복원"이라는 우리 서술의 가정이 맞는지
- ⬜ Pavliotis & Stuart 2008 — "지속성=느림 일치는 가정"을 이 책에 귀속시켜도 되는지
- ⬜ LeCun 2022 · Assran et al. 2023 — 잠재 타깃 서술
- ⬜ Locatello et al. 2019 — 라벨 없는 모델 선택 불가 주장
- ⬜ Cleveland et al. 1990 (STL) · Huang et al. 1998 (EMD) — 서지만
- ⬜ Hsu et al. 2017 (FHVAE) · Li & Mandt 2018 (DSAE) — 정적/동적 분리 서술
- ⬜ Ennadir et al. 2025 (TS-JEPA) — 사용자 확인 완료 표시 있음, 주장만 확인

### 미확인 후보 (`[※]`, 존재 여부부터)
- ⬜ 시계열 MAE 계열 · TS2Vec · TNC · HEPA · LeNEPA · SlowVAE · β-VAE 계열

---

## ✅ Khinchin 1934 — 위너–힌친 (방향 1의 근거)

**서지 확정.** A. Khintchine, "Korrelationstheorie der stationären stochastischen Prozesse", *Mathematische Annalen*, 109(1):604–615, 1934. doi:10.1007/BF01449156. (Wiener의 결정론적 판본은 N. Wiener, "Generalized Harmonic Analysis", *Acta Mathematica*, 55:117–258, 1930. doi:10.1007/bf02546511.)

**정리 내용**: 광의 정상(wide-sense stationary) 확률과정의 파워 스펙트럼 밀도는 자기상관 함수의 푸리에 변환과 같다. **우리가 인용하는 것은 확률과정 판본이므로 Khinchin 1934가 맞다.**

⚠️ **정상성 전제.** 정리는 광의 정상성을 요구한다. 인트로의 다리 문장("저주파 성분은 자기상관이 느리게 감쇠하고, 남아 있는 동안 먼 지평 예측에 기여한다")은 이 전제 위에 있다. **사용자 승인(2026-08-08): 인트로 산문에는 조건을 달지 않는다** — 방향 1은 대비를 세우는 배경이고 조건을 달면 정작 힘을 실을 방향 2가 흐려지기 때문. 이 전제는 여기 기록으로만 남긴다.

---

## ✅ Randall & Antoni 2011 — 포락선 분석 (반례의 현실성 근거)

**서지 확정.** R. B. Randall, J. Antoni, "Rolling element bearing diagnostics — A tutorial", *Mechanical Systems and Signal Processing*, 25(2):485–520, 2011. **독립 검증**: arXiv:2304.08249의 참고문헌 [1]이 동일한 서지를 싣고 있다.

**우리가 기대는 주장과 그 근거.** 같은 논문(arXiv:2304.08249, §1)이 Randall & Antoni를 인용하며 기전을 이렇게 서술한다:

> "Many methods analyze the spectrum of the envelope of the vibration signal as the fault frequencies become more apparent after demodulating the signal [1]."

그리고 포락선 스펙트럼은 **힐베르트 변환**으로 얻는다고 명시한다. 즉 결함 정보는 신호의 변조에 실려 있고, **비선형 판독 단계를 거쳐야** 드러난다. 이것이 `RELATED` 2.1(a) 둘째 갈래가 기대는 전부다.

⚠️ **과장 금지 두 가지.**
1. 위 arXiv 논문 본문에 "resonance(공진)"라는 말은 **나오지 않는다.** "고주파 공진을 캐리어로 변조한다"는 서술은 검증되지 않은 2차 웹 출처에서 온 것이므로 **쓰지 말 것.** 검증된 것은 "결함 주파수가 복조 후에 뚜렷해진다"까지다.
2. 베어링 결함이 **우리 뜻의 지속성**(Δ 뒤 예측에 기여)을 갖는지는 확인하지 않았다. 이 인용이 뒷받침하는 것은 **"요인이 변조에 실리는 부류가 실재한다"**이지, "그 부류가 지속적이면서 느린 대역에 없다"가 아니다. 주장을 여기서 멈출 것.

---

## 2026-08-08 적용 완료 항목

1. ✅ **정의 B로 통일 + 다리 문장** — `INTRO` P3에 정의를 앞당기고 자기상관→예측기여 다리를 넣음. P5의 후행 정의는 삭제.
2. ✅ **Khinchin 1934 인용** — `INTRO` P3 다리 문장에.
3. ✅ **Randall & Antoni 2011** — `RELATED` 2.1(a) 둘째 갈래에 한 문단.
4. ✅ **주장 범위 명시** — `INTRO` P5에 "존재 주장이지 흔하다는 것이 아니다", `OUTLINE` Conclusions 한계 5로 확장.

**헤지 정리(같은 날)**: `INTRO` P2 "두 가지로 읽힌다" → 단언으로. P7 "귀결로 읽힌다" → "우리는 ~로 해석한다 / 다만 그 인과를 직접 측정하지는 않았다"로 명시화.

---

# 미확인 후보 7건 — 전건 실재 확인 (2026-08-08)

**결과: 7건 모두 실재하며 서지 확정.** 잘못 기억한 이름은 없었다. 초안에서 `[※]` 표시를 전부 제거하고 저자-연도로 대체했다.

| 우리가 쓰던 이름 | 확정 서지 | 상태 |
|---|---|---|
| TS2Vec | Z. Yue, Y. Wang, J. Duan, T. Yang, C. Huang, Y. Tong, B. Xu. "TS2Vec: Towards Universal Representation of Time Series." *AAAI* 36:8980–8987, 2022. arXiv:2106.10466 | ✅ |
| TNC | S. Tonekaboni, D. Eytan, A. Goldenberg. "Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding." *ICLR* 2021. arXiv:2106.00750 | ✅ |
| HEPA | J. Petersen, G.-A. Lombardi, R. Maggioni, C. Mazzoleni, F. Martelli, P. Petersen. "HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series." arXiv:2605.11130, 2026. Spotlight, FMSD @ ICML 2026 | ✅ |
| LeNEPA | A. Chemeris, M. Jin, R. Balestriero. "LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series Representation Learning." arXiv:2607.00958, 2026. KDD MILETS 2026 workshop | ✅ |
| SlowVAE | D. A. Klindt, L. Schott, et al. "Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding." *ICLR* 2021. arXiv:2007.10930 | ✅ |
| β-VAE 계열 | I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, A. Lerchner. "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." *ICLR* 2017 | ✅ |
| 시계열 MAE 계열 | Y. Nie, N. H. Nguyen, et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (PatchTST). *ICLR* 2023. arXiv:2211.14730 — 마스킹 사전학습을 포함 | ✅ |

## ⚠️ SlowVAE — 우리 서술을 고쳤다

초안은 *"느림을 사전분포로 넣는 접근"*이라 썼는데 **부정확하다.** SlowVAE가 두는 것은 **희소(sparse) 사전분포**다. 인접 프레임 사이의 변화가 대체로 작지만 가끔 크게 튀는 성질(temporally sparse)을 사용하며, 논문 제목 자체가 "Temporal **Sparse** Coding"이다. **"느림"이 아니라 "희소"로 쓸 것.** 2.3을 그에 맞게 고쳤다.

## ✅ HEPA — CLAUDE.md §5가 원문과 전건 일치

`arxiv.org/html/2605.11130v4` 전문에서 확인했다. CLAUDE.md §5의 HEPA 서술은 **하나도 틀리지 않았다.**

| CLAUDE.md §5의 서술 | 원문 |
|---|---|
| 구간 요약 타깃, 양방향 + 어텐션 풀링 | "The same encoder $f_\theta$ applied bidirectionally to $\mathbf{x}(t,t+\Delta t]$ with attention pooling, produces the target representation" |
| 가중치 공유·공동 학습, EMA·stop-grad 없음 | "a weight-shared copy of $f_\theta$ rather than an EMA copy or a stop-gradient branch" |
| SIGReg | "a SIGReg term on the predictor output prevents representation collapse, replacing the exponential moving average (EMA) momentum schedule" |
| L1 | "combines an L1 prediction objective ... with the SIGReg regulariser" ($\mathcal{L}=(1-\alpha)\lVert\hat h-h^*\rVert_1+\alpha\mathcal{L}_{\text{SIG}}$, $\alpha=0.1$) |
| 사인 절대 위치 | "sinusoidal positional encodings" |
| 연속 스칼라 Δ 조건화 | "the predictor $g_\phi$ is a 2-layer MLP that takes the encoder output $\mathbf{h}_t$ together with a prediction horizon $\Delta t$" |

## ⛔ 새로 발견 — 로그 균등 지평 표집도 우리 기여가 아니다

HEPA 원문: **"The horizon $\Delta t$ is sampled from a log-uniform distribution during pretraining to force multiscale learning."**

우리 핸드오프 §6.8도 지평을 **로그 균등**으로 표집한다. CLAUDE.md §5는 연속 Δ 조건화가 우리 기여가 아니라고 이미 적었지만, **로그 균등 표집도 마찬가지**라는 점은 기록에 없었다. **Method·Related Works 어디서도 이 둘을 우리 것으로 제시하지 말 것.** `RELATED` 2.2 잠재 예측 계열 마지막 문장에 명시해 두었다.

## 우리 기여로 남는 것 (위 대조 뒤)

지평이라는 축 자체도, 연속 조건화도, 로그 균등 표집도 HEPA에 있다. **남는 것은 그 축에 게이트를 걸어 좌표 블록에 대응시킨 것 하나**이며, 타깃 형태(구간 안의 한 위치 vs 구간 전체 풀링)가 부차적 차이다. 기여 주장을 이 범위 밖으로 넓히지 말 것.

---

# 확정 서지 8건 — 주장 확인 (2026-08-08, A그룹 완료)

웹으로 원문 접근이 가능한 7건. **모두 확인, 두 건은 서술을 정밀화했다.**

| 문헌 | 확정 서지 | 우리 주장 | 결과 |
|---|---|---|---|
| Hyvärinen & Morioka 2016 | "Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA", NeurIPS 2016, arXiv:1605.06336 | "비정상성 아래 요인 복원" | ⚠️ **정밀화** (아래) |
| Locatello et al. 2019 | ICML 2019 (PMLR), arXiv:1811.12359. 저자 7인: Locatello, Bauer, Lucic, Rätsch, Gelly, Schölkopf, Bachem | "라벨 없이 모델을 고를 수 없다" | ✅ 원문 "well-disentangled models seemingly cannot be identified without supervision" |
| Ennadir et al. 2025 (TS-JEPA) | "Joint Embeddings Go Temporal", arXiv:2509.25449. 저자 3인: Ennadir, Golkar, Sarra. **게재처: NeurIPS 2024 Workshop on Time Series in the Age of Large Models** | 가려진 패치 타깃, 지평 없음 | ✅ 전건 일치 (아래) |
| Assran et al. 2023 (I-JEPA) | **CVPR 2023, pp. 15619–15629.** 저자 8인 | 원시 대신 표현을 타깃으로 | ✅ 원문 "predict the representations of various target blocks" |
| LeCun 2022 | "A Path Towards Autonomous Machine Intelligence", v0.9.2, OpenReview, 2022-06-27. `openreview.net/pdf?id=BZ5a1r-kVsf` | JEPA를 제안한 위치 논문 | ✅ |
| Hsu et al. 2017 (FHVAE) | NeurIPS 2017, arXiv:1709.07902. 저자 3인: Hsu, Zhang, Glass | 두 수준 잠재 분리 | ⚠️ **정밀화** (아래) |
| Li & Mandt 2018 (DSAE) | arXiv:1803.02991, ICML 2018 | 정적/동적 분리 | ✅ 원문 "split into a static and dynamic part ... time-dependent features (dynamics) from features which are preserved over time (content)" |

## ⚠️ Hyvärinen & Morioka 2016 — 식별 강도를 정확히 쓸 것

원문: **"TCL combined with linear ICA estimates the nonlinear ICA model up to point-wise transformations of the sources, and this solution is unique."**

두 가지가 우리 초안에 빠져 있었다. ① **TCL 단독이 아니라 선형 ICA와 결합**해야 한다. ② 복원은 정확한 것이 아니라 **요인별 점별(point-wise) 변환을 제외한** 것이다. 보조 변수는 **시간 구간(segment)**이고, 비정상성이 그 구간을 구별 가능하게 만드는 역할을 한다. 2.3을 그에 맞게 고쳤다.

## ⚠️ Hsu et al. 2017 — "발화 수준/구간 수준"은 우리 용어였다

원문은 **sequence-dependent priors / sequence-independent priors**로 부르며, 전자가 시퀀스 안에서 일정한 것(화자), 후자가 구간마다 바뀌는 것(음성 내용)을 담는다. "발화 수준·구간 수준"이라는 이름은 원문에 없으므로 **서술로 풀어 쓰도록** 고쳤다.

## ✅ TS-JEPA — 그림 `fig_targets_B1.png`의 주장이 전건 일치

`arxiv.org/html/2509.25449v1` 전문 확인. 그림이 주장하는 네 가지가 모두 맞다.

| 그림의 주장 | 원문 |
|---|---|
| 균등 70% 패치 마스킹 | 70%, "following a uniform masking strategy" |
| 비인과(양방향) 인코더 | "standard transformer architecture incorporating self-attention", 비마스킹 패치만 입력 |
| EMA 타깃 + stop-grad | "the weights of the EMA-encoder are updated as an exponential moving average", m=0.998 |
| 지평 변수 없음 | 예측기는 관측 토큰 → 마스킹 토큰 위치 매핑일 뿐, 시간 오프셋 변수 없음 |

## ✅ Cleveland et al. 1990 (STL) — 사용자 확인

*Journal of Official Statistics*, Vol. 6, No. 1, 1990, pp. **3–73**, Statistics Sweden. (웹 검색에서는 3–33과 3–73이 엇갈렸으나 **3–73**으로 확정.)

## 남은 것

- ⬜ **Pavliotis & Stuart 2008** — 인용 방식 결정 대기 (아래 별도 항목)
- ⬜ Wiskott & Sejnowski 2002 — 서지 확정(*Neural Computation* 14(4):715–770). 주장은 PFA §2.1(Wiskott 본인)로 이미 검증. **저위험**
- ⬜ Huang et al. 1998 (EMD) — 주장이 "데이터 자체에서 진동 모드를 뽑는다" 한 줄. **저위험**

---

# 검색 요약에만 기대던 3건 — 원문으로 승격 (2026-08-08)

이번 세션의 오류 두 건(PFA의 "predictable ≠ slow", 베어링의 "고주파 공진 캐리어")이 **모두 검색 요약을 믿은 데서** 나왔다. 그래서 본문에 서술적 주장을 실으면서 요약에만 기대던 3건을 원문으로 다시 확인했다.

## ✅ PatchTST (Nie et al., 2023) — 복원 계열 귀속 확정

전문(arXiv PDF) §4 확인. 우리 주장이 전건 맞다.

> "We then select a subset of the patch indices uniformly at random and mask the patches according to these selected indices with zero values. **The model is trained with MSE loss to reconstruct the masked patches.**"

마스킹 비율 **40%**, 비중첩 패치, 입력 길이 512·패치 크기 12(=42패치). 저자 4인: Nie, Nguyen, Sinthong, Kalagnanam. ICLR 2023.

## ⚠️ TNC (Tonekaboni et al., 2021) — 표현을 원문에 맞춤

우리 초안은 *"신호의 **국소 정상성**을 이용해"*였는데, 원문은 정상성을 신호가 아니라 **결과로 얻어지는 이웃**의 성질로 말한다.

> "takes advantage of the **local smoothness of a signal's generative process** to define neighborhoods in time **with stationary properties**"

→ *"생성 과정의 국소적 매끄러움을 이용해 정상성을 갖는 시간 이웃을 정의한다"*로 고쳤다.

## ⚠️ TS2Vec (Yue et al., 2022) — 초록이 보증하는 범위로 축소

우리 초안은 *"**여러 시간 규모의** 문맥 뷰를 계층적으로 대조"*였는데, "여러 시간 규모"는 초록이 말하지 않는다. 초록이 보증하는 것은 다음까지다.

> "performs contrastive learning in a hierarchical way over augmented context views, which enables a robust contextual representation for **each timestamp**"

→ *"증강된 문맥 뷰 위에서 계층적으로 대조해 타임스탬프 단위 표현을 얻는다"*로 고쳤다. (계층적 대조가 시간축 풀링으로 구현되는지는 확인하지 않았으므로 쓰지 말 것.)

---

# ✅ Pavliotis & Stuart 2008 — PDF 불필요로 해결 (사용자 승인)

**문제였던 것**: 초안이 *"둘이 일치한다는 것은 **가정이며**(Pavliotis & Stuart, 2008)"*로 써서, **"지속성 = 느림은 가정이다"라는 메타 명제를 그 책에 귀속**시키고 있었다. 그 책은 다중척도 평균화·균질화 교재이지 그 명제를 논하는 책이 아니다. PFA와 같은 유형의 귀속 오류.

**해결**: 메타 서술을 없애고 **사실 진술 두 개**로 바꿨다. 인용은 "느린–빠른 척도 분리 틀"만 받친다.

> 두 축은 빠른 성분이 충분히 빨리 섞이는 상황에서는 일치하지만(Pavliotis & Stuart, 2008), **항상 일치하지는 않는다.** 우리 합성 데이터가 일치하지 않는 경우다.

`INTRO` P4와 `RELATED` 2.1(b) 두 곳에 동일하게 적용. **책 본문 확인이 필요 없어졌다.**

---

# 최종 상태 (2026-08-08)

| 검증 등급 | 문헌 |
|---|---|
| **원문 전문** | PFA · HEPA · TS-JEPA · PatchTST |
| **저자 작성 초록** | TCL · Locatello · I-JEPA · FHVAE · DSAE · TS2Vec · TNC |
| **2차 출처(허용)** | Khinchin(정리 진술) · Randall & Antoni(인용 논문 경유) · Wiskott&Sejnowski 2002(PFA §2.1 경유) |
| **사용자 제공** | Cleveland et al. 1990 (JOS 6(1):3–73, Statistics Sweden) |
| **저위험·미확인** | Huang et al. 1998 (주장이 "데이터에서 진동 모드를 뽑는다" 한 줄) · SlowVAE·β-VAE·LeCun 2022 (주장이 제목·통념 수준) |

**`INTRO`와 `RELATED`에 검증되지 않은 주장은 없다.** 집필 세션은 이 파일을 읽지 않아도 된다.

---

# ⚠️ 수치 오류 정정 — Fig. 1c 서술 (2026-08-08)

`INTRO` P4의 *"그 읽기는 일시 요인의 수명보다 긴 창으로 적분할 때에만 성립한다"*는 **부정확했다.** `runs_v2/fig_persistence.json`의 `integration_curve`를 대조한 결과:

| 창 | 지속 요인 $R^2$ | 일시 요인 $R^2$ |
|---|---|---|
| 11 | 0.201 | 0.723 |
| 31 | 0.208 | 0.730 |
| 101 | 0.239 | 0.641 |
| 301 | 0.318 | 0.399 |
| 1001 | 0.501 | 0.116 |
| 3001 | 0.654 | 0.021 |
| 9001 | 0.576 | 0.005 |

일시 요인 수명은 **50스텝**(`u_tau`=50, 자기상관 반감기 51.0). 창이 수명의 2배(101)여도 지속 요인은 0.239로 읽히지 않는다. **뒤집히는 것은 창 1001(수명의 20배)부터**이고 최댓값은 3001(60배)이다. 따라서 **"수명보다 긴"이 아니라 "수명의 수십 배"**로 써야 한다. 문장을 그에 맞게 다시 썼고, 검증된 수치(0.72 / 0.65 / 0.02)를 본문에 넣었다.

⚠️ **9001에서 지속 요인이 0.576으로 다시 떨어진다** — 창이 지나치게 넓으면 지속 요인 자체의 변화도 뭉개지기 때문이다. 본문은 "수십 배"까지만 말하고 단조 증가라고 쓰지 않는다.

---

# STL · EMD — 제목 수준 주장으로만 사용 (2026-08-08)

Related Works 2.1에 세 문장으로 되살렸다. **두 문헌 모두 원문을 보지 않았으므로, 논문 제목이 보증하는 범위 안에서만 서술한다.**

| 문헌 | 서지 | 본문에 쓴 주장 | 근거 |
|---|---|---|---|
| Cleveland et al., 1990 | *Journal of Official Statistics*, 6(1):3–73, 1990, Statistics Sweden (사용자 확인) | "계절 성분과 추세를 국소 회귀로 분해한다" | 제목 "STL: A Seasonal-Trend Decomposition Procedure Based on **Loess**"가 그대로 보증 |
| Huang et al., 1998 | *Proc. R. Soc. Lond. A*, 1998 | "신호에서 진동 모드를 경험적으로 뽑아낸다" | 제목 "The **Empirical Mode Decomposition** and the Hilbert Spectrum…"이 그대로 보증 |

⚠️ **이 이상은 쓰지 말 것.** 예를 들어 EMD가 IMF를 고주파부터 저주파 순으로 내놓는다거나, STL의 강건성·반복 구조 같은 것은 확인하지 않았다. 세 번째 문장 *"이들이 내놓는 것은 신호의 성분이지 그 신호를 만든 요인이 아니다"*는 두 방법이 **신호 분해 기법**이라는 사실에서 나오는 것이며, 특정 문헌에 귀속시키는 주장이 아니다.
