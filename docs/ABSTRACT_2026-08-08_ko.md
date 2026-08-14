# 표제부와 Abstract — 확정본 (2026-08-08)

**이 문서의 위상**: 아래 제목·저자·초록이 **사용자가 확정한 정본**이다. 집필 세션은 그대로 옮긴다. 아웃라인 §Abstract의 이전 초안은 폐기한다.

---

## 제목 (확정)

> **Horizon-Gated Latent Prediction: Disentangling Persistent Factors in Time Series**

부제나 각주(`\thanks{...}`)는 넣지 않는다.

## 저자 블록 (확정, 그대로 옮길 것)

```latex
\author{\IEEEauthorblockN{Jaechan Lee}
\IEEEauthorblockA{\textit{University of Waterloo} \\
Waterloo, Canada \\
j2342lee@uwaterloo.ca}
}
```

단독 저자이므로 템플릿의 `\and` 블록 세 개를 지우고, `\thanks{Identify applicable funding agency...}`와 템플릿 부제도 지운다.

## 키워드

IEEE 템플릿의 `\begin{IEEEkeywords}`에 넣을 항목: *time-series representation learning, self-supervised learning, disentanglement, joint-embedding predictive architecture, interpretability*. "JEPA"는 검색 키워드로만 쓰고 방법 이름으로는 쓰지 않는다(CLAUDE.md §2).

---

## Abstract (정본)

> Time-series representation learning has advanced considerably, but a single series mixes factors acting on different timescales, which makes the embedding hard to interpret and use. Work that separates them at the embedding level has mostly addressed static factors, those that do not change over an entire sequence, because invariance by definition supplies a training signal. Persistent factors that vary slowly without being constant get no such free constraint, and they need an axis that fixes where the boundary between slow and fast lies. That axis is already present in prediction-based representation learning: the horizon sets how far into the future the model must predict, and therefore how long information must persist in order to be useful. We propose Horizon-Gated Latent Prediction (HGLP), which places a single threshold on that axis. Beyond it the gate forces the predictor to use only a designated subspace, z_slow, while a cross-covariance penalty between the blocks holds the two apart. A predictor cannot foresee upcoming transitions, so the only way to lower error at long horizons is to record the current state. We also propose a separation index (SEP) that scores how far apart the two blocks are. On synthetic and real data in which persistent and transient factors coexist, we confirm that the transient factor can be separated out of z_slow. Against post-hoc rotations of an ungated embedding, we lead on the synthetic data but trail Slow Feature Analysis on the real ones. We further quantify the downstream trade-off that the separated embedding incurs, and report that it gives up downstream accuracy while degrading less in a new domain when labels are scarce.

---

## 정합성 확인 (2026-08-08)

**✅ 철회 목록 준수 확인.** 초록은 금지된 표현을 쓰지 않는다.

- "the transient factor can be separated **out of** z_slow" — 배제 프레이밍이다. 금지된 "게이트가 지속 요인을 z_slow에 모은다"로 읽히지 않는다.
- "**trail** Slow Feature Analysis on the real ones" — 실데이터 열세를 명시한다. 금지된 "실데이터에서 대등"이 아니다.
- "**gives up** downstream accuracy while degrading less" — 맞바꿈으로 서술하며 "산다/buy" 비유를 쓰지 않는다. "degrading less"도 절대 성능 우위를 주장하지 않는다.

**✅ 용어 확인.** HGLP · z_slow · SEP · persistent/transient가 CLAUDE.md §2의 명명과 일치한다.

**✅ 용어 통일 완료 — separation index.** 초록을 *separation index*로 맞춰 Results의 Table 2 캡션(*Separation index by mechanism*)과 일치시켰다. 약어는 양쪽 모두 SEP이다.

**ℹ️ 이전 초안 대비 빠진 것 두 가지 — 의도된 것인지 확인.**

1. **6/6 상보성 결과.** 이전 초안은 "the gated model scores highest in all six settings, and the gate and the penalty are complementary"를 담았다. 확정본은 "we confirm that the transient factor can be separated out of z_slow"로 더 일반적으로 말한다. Results §A의 주요 결과가 초록에 수치로 등장하지 않는다.
2. **라벨 없이 못 고른 하이퍼파라미터.** 이전 초안의 마지막 문장 "We report the hyper-parameters we could not select without labels."가 빠졌다. 아웃라인은 초록의 마지막 문장들을 "자르면 안 되는 것"으로 지정했으나, 확정본도 부정적 결과 두 가지(실데이터 열세·다운스트림 비용)를 이미 담고 있어 정직성 요건 자체는 충족한다. 하이퍼파라미터 문제는 Method §F와 Conclusions 한계 1에서 정면으로 다룬다.

**분량.** 목표 0.15쪽. 현재 영문 약 260단어로 IEEE 2단 초록으로는 다소 길다. 넘치면 세 번째 문장("Persistent factors that vary slowly…")과 다섯 번째 문장("A predictor cannot foresee…") 가운데 하나를 줄이는 것이 손해가 가장 적다 — 둘 다 본문(Introduction P5, Method §C)이 다시 말한다.
