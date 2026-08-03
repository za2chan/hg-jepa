# HGLP 체크리스트 v3 — 결정 확정 + 8/7 마감 5일 플랜

> 위임용 원본은 영문판 `CHECKLIST_v3.md`다 (Claude Code에는 영문판을 줄 것).
> 이 문서는 열람용 미러 — 내용 수정 시 두 판을 함께 갱신한다.

리포: `za2chan/hg-jepa` @ `b4e6c70`. 오늘 = 8/2. **마감 = 8/7 23:59,
IEEE 컨퍼런스 형식 8페이지 (참고문헌 포함).**

구성: A. 확정된 결정 → B. 5일 스프린트 (8/7까지 반드시) → C. 마감 후
백로그 (워크숍 판). 상세 배경은 CHECKLIST_v2_ko.md 참조 — 이 문서가 우선한다.

---

## A. 확정된 결정 (사용자 승인 완료 — Claude Code는 이를 변경하지 않는다)

- **D1. τ 앵커 규칙 (2026-08-02 수정, 사용자 승인):** `estimate_tac`의
  `chosen` = **파생열별 T_ac의 최솟값을 u 스케일로 환산한 값** (제곱/energy
  계열 ×2 — OU 과정을 제곱하면 상관시간이 절반; raw/envelope ×1). fast 수명의
  하한 추정치. **τ = c·chosen, c = ln(1/eps), eps = 0.05 (CLI 인자) ⟹
  c ≈ 3.0**; `c=`는 부록 스윕용 오버라이드로만 유지. 합성 검증 기준:
  energy T_ac ≈ 3.125패치(±25%), 환산 후 chosen ≈ 5.24, τ ≈ 15.7 (기존
  하드코딩 τ=16이 소급 설명됨 — 관측이지 튜닝이 아님). 유도와 발견 기록:
  `docs/tau.md`. (원래의 6.25패치 기준은 관측 불가능한 u 자체의 수명을
  요구한 스펙 오류였음.)
- **D2. NCE online 타깃:** **양방향 그래디언트 (SimCLR식), EMA 없음,
  stop-grad 없음.** 가장 단순한 형태. 불안정하면 멈추고 보고 (자율 후퇴 금지).
- **D3. 시드/데이터:** **(a) `make_dataset(..., seed=seed)`로 수정, 합성 전면
  재실행.** 재실행은 B의 매트릭스 런에 통합 — 어떤 설정도 두 번 학습하지
  않는다.
- **D4. Part 2 대표 줄기:** 판정 규칙만 확정 — "메커니즘 매트릭스에서
  slow-kept과 leak 양쪽 모두 우세한 줄기를 대표로; 지표가 갈리면 **Reg
  유지** (실데이터 재실행 비용 회피)." 결과가 나오면 규칙대로 기계 적용.
  **2026-08-03 확정 → Reg.** NCE-online은 선형적으로 무정보(0.456 vs
  0.776), 낮은 leak은 공허 → 지표 불일치 → Reg. 메커니즘은 안정화된 잠재
  타깃을 요구(nce+ema는 작동, nce+online은 실패), 손실 형태는 부차적.
  실데이터 NCE 재학습 불필요.
- **D5. 컷 순서:** 승인됨 — NCE d_slow → 결맞음 스윕(탈락 데모는 유지) →
  HI → HSIC → TS2Vec(P2-4 답 확보 시에만). 기준: 헤드라인 주장이 의존하는
  실험은 절대 컷 불가, 중복/보너스 증거부터.
- **D6. 단기 목표:** 8/7 23:59, IEEE 형식 8pp. 워크숍 판은 그 이후 확장.
- **D7. vfloor 판정 규칙:** vfloor=0에서 RankMe가 붕괴를 보이면 → 논문
  수식에 세 번째 항으로 편입 / 안 보이면 → vfloor=0을 기본값으로 하고 제거.
  Day 1에 `git log -S "relu(1.0"`으로 도입 시점·사유 확인해 기록.
  **2026-08-03 확정 → 제거.** 붕괴 없음(항 없이 RankMe 13.8→22로 오히려
  상승); 중립적이지 않았음(s0 leak 이상치도 이 항이 운반). `vfloor=` 모든
  트레이너에서 기본값 0; 메인 매트릭스 = vf0, vf1 → 부록 절제 근거. 항은
  손실 수식에서 제거. (도입: 최초 커밋 `d741957`, 별도 사유 없음.)

---

## B. 5일 스프린트 (8/3 ~ 8/7) — 8/7 판에 반드시 들어가는 것

### 원칙
1. 8/7 판의 모든 문장은 **그 시점에 존재하는 결과**로만 뒷받침한다. 미완
   실험은 "in progress"로 정직하게 표기 — 자리 채우기 수치 금지.
2. 체크포인트 재활용 우선: `runs/model_*.pt`(인코더), `runs/emb_*.npz`
   (임베딩+라벨)가 이미 있다. **shift(P3-4)와 label-eff(P3-5)는 재학습 없이
   평가 계층에서 돈다** — Part 2의 3분의 2가 사실상 공짜.
3. 매일 끝에 "오늘 확보된 결과로 논문에 쓸 수 있는 문장" 목록 갱신.

### Day 1 (8/3) — 게이트 & 정합성 소형 수정
- [ ] ⏱ **타이밍 측정 (스케줄용):** 하드웨어 확정 — 전용 H200 1대. 0.5M
      모델이라 1런 수 분 예상 → **재실행 경로 확정, 동결 경로 폐기.**
      측정값은 야간 배치의 소요 시간 산정에만 쓴다.
- [ ] P0-1: `tac.py` + `tau=auto` (수정된 D1 규칙). **AC: 합성에서 energy
      T_ac ≈ 3.125패치(±25%), chosen ≈ 5.24.** 어긋나면 기록하고 보고 —
      eps 몰래 조정 금지.
      통과 시: 기존 tau∈{4,64} 스윕이 c-스윕으로 소급 유효 (부록 재료 확보).
- [ ] P0-2: `vfloor=1|0` 스위치 + run JSON 기록. `git log -S` 조사 기록.
- [ ] P0-3 최소판: 모드 분리 `loss=reg|nce` × `target=ema|online` (D2 사양).
      옛 이름 별칭 유지. **리포 전체 개명은 하지 않는다** — 논문 텍스트만
      HGLP 채택, 코드 개명은 마감 후 (스프린트 중 대규모 리팩터 금지).
- [ ] P0-4: `make_dataset(..., seed=seed)` 수정 (D3). 프로토콜 문장 초안.
- [ ] P3-1 준비: `datagen.py`에 `regime_mode=freq|variance` + 자기 테스트
      assert 조건부화 (H200 덕에 variance-only가 스프린트로 복귀).
- [ ] 🌙 **Day 1 야간 배치 발사:** P0 착지 확인 후 `run_matrix.sh` 일괄 —
      Reg 전체 재실행(seed=seed) + NCE(online) 신규 셀(gate+xcov/gate-only/
      xcov-only/no-gate) + vfloor=0 셀(2줄기) + P3-1 variance-only(2줄기
      ×3시드) + NCE d_slow(공짜 필러, D5 컷 1순위였으나 컴퓨트 비용 소멸).
      H200에서 전체 수 시간 — 아침에 결과 회수.

### Day 2 (8/4) — 매트릭스 회수 + 규칙 적용 + Part 2 확보
- [x] **야간 배치 회수 & 규칙 기계 적용 (2026-08-03 완료):** D7 → 제거
      (붕괴 없음, RankMe 13.8→22); vfloor= 기본값 0을 모든 트레이너에 전파;
      메인 매트릭스 = vf0, vf1 → 부록. D4 → Reg가 Part 2 담당 (NCE-online
      무정보; 안정화된 타깃이 핵심 인자, reg+ema/nce+ema/nce+online 3점 측정).
      실데이터 NCE 재학습 불필요. 신규 미해결: horizon-count / Δ-set 유도
      (`docs/tau.md`); HAPT horizon 확장은 raw-data 감사에 종속 — 옵션 2
      (L=256) 불가능, 옵션 1이 상한 (사용자 승인 대기).
- [ ] **P3-4 shift_eval.py (최우선):** 교란 평가 데이터 생성 (OU lifetime
      50→{20,150}, OU scale ×{0.5,2}, noise) → **새 체크포인트**(야간 배치
      산출물)로 인코딩 → 깨끗한 데이터로 학습한 probe를 교란 데이터에 적용,
      z_slow vs z_full 낙차. **초록의 자리표시 문장이 이 결과를 기다린다.**
- [ ] **P3-5 label_eff.py:** 새 `emb_*.npz` 위에서 probe 라벨 {1%,10%,100%}
      서브샘플링, z_slow vs z_full, **양쪽 모두 λ 튜닝**. 재학습 0.
- [ ] P0-5 Check 1 착수: `check_coherent_periodic` (ACF 포락선 감쇠 +
      Fisher g, **샘플 레벨에서** — 패치 평균 위 금지).
      ⛔ **중단 지점: 합성이 Check 1에서 탈락하면 임계값 캘리브레이션으로
      복귀, 통과 전까지 스크린 관련 논문 문장 동결.**
      **8/7 폴백:** Check 1이 5일 안에 안정화 안 되면 → 8/7 판은 기존
      C1-C3 스크린으로 제출 + "coherent-periodic 검출은 향후 확장" 한 문장.
      워크숍 판에서 완성. (탈락 데모용 주기 주입 신호는 P3-2에서 겸용.)

### Day 2-3 (8/4-5) — 잔여 러너 + 선택 실험
(매트릭스 본체는 Day 1 야간으로 이동 완료. 기존 cpc_* 런은 nce+ema
대조군으로 보존 — 삭제·재실행 금지.)
- [ ] variance-only 결과 → 메모리 노트의 **결과 문장 1개** 논문에 기입
      ("분산으로만 구분되는 regime에서 분리 확인 여부 + 줄기 간 차이 유무").
- [ ] (여유 시, 구현 반나절) P3-3 구간요약 타깃 ablation — 컴퓨트는 공짜지만
      양방향 인코더+attention pooling **구현**이 비용. Day 3 정오까지 구현이
      순조로우면 야간 배치 2호로 발사, 아니면 워크숍 판으로.

### Day 3 (8/5) — 위험 실험 + 집필 개시
- [ ] **P2-4 고전 베이스라인 (FM복조→이동평균→HMM), 30줄.** 결과가 어느
      쪽이든 F1 문단을 그에 맞게 쓴다 — 답 모르고 제출 금지.
- [ ] 집필: IEEEtran 템플릿 포팅 (`report.tex` → 8pp 목차 매핑은 이미 확정
      되어 있음: Intro / Related 2.1-2.4 / Method 3.1-3.4 / Exp 4.0-4.2 /
      Limitations / Appendix).
      **수업 컨텍스트 (Responsible AI):** 이 판에 한해 RAI 의의를 명시적으로
      강조할 것 —
      (a) 초록 1문장 + 서론 payoff 문단을 해석성/신뢰성 어휘로 재서술,
      (b) **Discussion에 "Relevance to Responsible AI" 소절 ~0.25p**, 네 기둥:
          ① interpretability by design — 설명 생성이 아니라 설명이 불필요한
            좌표계 학습, block×factor 매트릭스로 해석성이 *측정됨* (사후
            SHAP류 대비; 렉처의 "measurable intrinsic interpretability vs
            post-hoc explanation" 프레임 재사용),
          ② responsible scoping — 스크린 + 적용조건 표 + predicted negative
            (XJTU) + 정상-only 배제 = "언제 쓰면 안 되는지"의 내장 판정 절차,
          ③ reliability — shift 결과 = 분포 변화 하 신뢰성의 정량 증거,
          ④ epistemic honesty — Exclusion의 비보증 명시, linear-level 한계,
            one-directional 공개.
      **HEPA 스코프:** 4중 대비(EMA/SIGReg/L1/구간요약) 전체는 related works
      한두 문장으로 강등. 단 **타깃 형태(point vs 구간요약) 한 문장은
      Method에 유지** — 비교가 아니라 게이트 논리의 전제 진술이므로.
      "as in HEPA" 오귀속 삭제는 그대로 필수. 산문 수정 일괄:
      - HEPA 오귀속 교체 (P1-4 문안)
      - "조건부 기댓값 — z_slow는 현재에 대한 메모" 문단
      - 적용 조건 표 (윈도우 오염 축 추가: dwell-vs-Δ_max와 "윈도우가 인자
        전환을 가로지르는가?"는 별개 축 — HAPT는 PTB-XL 같은 static이 아니라
        독자 범주; 출처 `datasets_table.py`) + 정상-only 배제 precondition +
        HAPT end-region readout 주의문 (F1 크기 ≠ slow 통합; clean 부분집합
        F1 0.74가 방어 가능한 수치; `hapt_anchor_probe.py`)
      - usage protocol 문단 (probe / 궤적 / 블록별 질의 + 폴백 문장)
      - F3 vfloor 문구 수정, "one-directional by design" 문장
      - Limitations 순서 (선형 배제 → 모델 선택 → xcov 극성 → 2-timescale)
      - Prop.→Gating/Exclusion, L_dcor→L_xcov, HGLP 명명 (텍스트만)

### Day 4 (8/6) — 그림 & 숫자 봉인
- [ ] `figures_paper.py` 확장: **모든** 표·수치가 run JSON에서 생성. 손 전사
      금지 — 이 논문은 그림↔본문 숫자 불일치 전과 2회.
- [ ] Method 그림: 손실 상자 → `ℓ(·, z̄)` + Reg/NCE 분기
      (`fig_method_pptx.py` 수정, 라벨 Gating/Exclusion).
- [ ] 개념 그림 이식 (`fig_concept.py`): 캡슐 분할 / "요인 위치를 정하는
      것" 라벨 / 순서축 캡션.
- [ ] 스크린 플로우차트 (Check 1→2 직렬; 폴백 경로면 C1-C3 그대로 그림).
- [ ] shift figure (Day 2 결과) — Part 2의 앵커 그림.
- [ ] 8pp 예산 검사: Intro 0.75 / Related 1 / Method 1.5 / Datasets 0.5 /
      Part1 1.5 / Part2 1.25 / Limit 0.5 / refs 1.

### Day 5 (8/7) — 정합성 패스 & 제출
- [ ] 전체 숫자 재생성 → 그림 vs 본문 대조 (자동화 산출물만 사용).
- [ ] 주장-증거 감사: 논문의 모든 정량 주장 옆에 근거 run 태그를 주석으로
      달아본다. 근거 없는 주장은 삭제 또는 "in progress"로 강등.
- [ ] IEEE 형식 검사 (마진/폰트/참고문헌 스타일), PDF 컴파일, 제출.
- [ ] 버퍼: 반나절은 비워둘 것. Day 2-4에서 밀린 것의 착지 지점.

### 8/7 판에서 의도적으로 빠지는 것 (논문에 "future work"로 정직 표기)
P3-6 HI · P4-2 TS2Vec · HSIC · 결맞음 시간 스윕 · 리포 전체 개명 ·
(여유에 따라) P3-3 구간요약 타깃, P0-5 Check 1(폴백 사용 시).
— P3-1 variance-only와 NCE d_slow는 H200 확정으로 스프린트에 **복귀**했다.
컷 목록에 남은 건 전부 구현 비용이 병목인 항목들이다.

---

## C. 마감 후 백로그 (워크숍 판) — v2_ko의 잔여 전부

우선순위 순: P0-5 Check 1 완성(폴백 썼다면) → P3-3 구간요약 타깃(스프린트
미착지 시) → P0-6 Δ_max 3단 판정(`screen_model.py` 재활용) → P1-1 리포
전체 개명 + 태그 이전 → P4-2 TS2Vec(+unmixing 공정성 쌍) → P4-3 부록 스윕
보강(soft/hard, HSIC, τ 추정 변형, `select.py` 재실행) → P3-6 HI →
결맞음 시간 스윕.

---

## Claude Code 위임 지시문에 붙일 것
1. "**실행 대상은 CHECKLIST_v3_ko.md뿐이다.** CHECKLIST_v2_ko.md는 배경·
   설계 근거·마감 후 백로그의 참조 문서다 — v2의 항목을 v3가 명시하지 않는
   한 착수하지 않는다. 두 문서가 충돌하면 v3가 이긴다."
2. "⛔ 중단 지점(P0-5, D2 불안정 시)에서는 멈추고 보고. 자율 우회 금지."
3. "섹션 A의 결정 D1-D7은 변경 불가. D4/D7은 결과가 나오면 적힌 규칙을
   기계적으로 적용."
4. "매일 종료 시: 확보된 결과 목록 + 논문에 쓸 수 있게 된 문장 목록 보고."
