# HGLP 데이터 EDA — 원신호 & 라벨 (2026-08-03)

관찰 전용. prep 스크립트 상수가 아니라 **원본 파일**에서 직접 산출.
스크립트: `eda_hapt.py`, `eda_rest.py`, `eda_prep_audit.py` (각자 수치·그림
재생성). 지정된 스킬 `/mnt/skills/user/exploratory-data-analysis/SKILL.md`은
**디스크에 존재하지 않아**, 요청자의 STEP 0–5를 방법론으로 따랐다.

> 이 문서는 영어 원본 `README.md`의 한국어 미러이며, 같은 그림을 참조한다.
> 권위 있는 원본은 영어판. 수치가 갱신되면 영어판을 먼저 고칠 것.

---

## STEP 0 — 위치 확인 (무엇이 존재하는가)

| 데이터셋 | prepped | 디스크상 원본 | 상태 |
|---|---|---|---|
| HAPT | `data/hapt.npz` (2926×128×12) | `data/hapt/RawData/` — acc 61 + gyro 61 `.txt`, `labels.txt` | ✅ 원본 존재 |
| PTB-XL | `data/ptbxl.npz` (5000×100×10) | `data/ptbxl/…-1.0.3/` (3.4 G, `records100/500` + `ptbxl_database.csv`) | ✅ 원본 존재 |
| XJTU | `data/xjtu_raw.npz` (15 베어링), `data/xjtu.npz` | `/mnt/workspace/data/MFM_data/…/XJTU-SY_Bearing_Datasets/` — **경로 해석됨**, `.parquet` (`.csv` 아님) | ✅ 원본 존재 |
| Synthetic | 생성기만 | `datagen.py` (`generate()`), 저장 출력 없음 | ✅ 재생성 가능 |

누락 없음. 중요하게 드러난 prep 가정: `hapt_prep`은 `acc_*`만 읽음(gyro
미사용); `xjtu_raw_prep`은 `Horizontal` 채널만 읽고(`Vertical` 미사용)
`.parquet`을 기대함.

---

## STEP 1 — 원신호 구조

**HAPT.** 61개 기록(실험), 30명 피험자(≈2 실험/피험자), `(T,3)` 가속도계
(g 단위), 50 Hz. 파일당 길이 198 / 359 / 642 s(min/med/max) = 9 898 /
17 963 / 32 089 샘플. 값 범위 [−2.01, 2.01] g, NaN 없음. 총 374분.

**PTB-XL.** 21 799 기록, 18 869 환자; 각 10 s, 12-lead, 100 Hz(`records100`)와
500 Hz 제공. prep은 lead II @100 Hz 사용 → 기록당 정확히 1000 샘플. mV 단위.

**XJTU-SY.** 3 작동조건 × 5 베어링 = 15개 run-to-failure 시험. 각 시험 =
**1분마다** 기록된 1.28 s 스냅샷(`32768 @ 25.6 kHz`, 2채널)의 시퀀스;
스냅샷 개수 = 베어링 수명(분) (52–161). 즉 베어링 1개 = 독립 기록 1개,
총 15개 기록.

**Synthetic.** 임의 길이; 타임스텝별 정답(아래).

![HAPT 전체 기록](figs/hapt_full_recording.png)

전체 기록만 봐도 "활동이 연속적인 ~12 s dwell로 이어진다"는 그림이 깨진다:
활동 세그먼트들은 **큰 미라벨(VOID) 구간**으로 분리돼 있다. 특히 걷기/계단
버스트(150–360 s)가 그렇다.

---

## STEP 2 — 라벨 구조 (기존 추론이 틀렸던 지점)

### HAPT

`labels.txt`: 1214행, 스키마 `[exp, user, act, start, end]`(양끝 포함 샘플
인덱스). **활동 클래스 6개가 아니라 12개** — 기본 6개(1–6) + **자세전환
클래스 6개(7–12)** (STAND_TO_SIT … LIE_TO_STAND). 이 전환 클래스들이 이
데이터에 **실제로 존재**한다(이전엔 불확실했음).

클래스별 세그먼트 길이(풀링 안 함), 초 단위:

| 클래스 | n_seg | 길이 min/med/max | 총 s |
|---|---|---|---|
| 1 WALK | 127 | 2.8 / 19.4 / 28.2 | 2442 |
| 2 UPSTAIRS | 183 | 4.4 / 12.7 / 17.9 | 2334 |
| 3 DOWNSTAIRS | 186 | 4.3 / 11.8 / 17.6 | 2159 |
| 4 SIT | 120 | 12.5 / 20.8 / 32.4 | 2534 |
| 5 STAND | 120 | 15.2 / 22.6 / 40.6 | 2762 |
| 6 LAY | 120 | 15.6 / 22.1 / 33.9 | 2737 |
| 7–12 전환 | 각 ~60 | 1.5 / ~3.5 / 9.9 | 총 ~1344 |

- **전체 원샘플의 27.0%가 VOID(라벨 0)** — 미라벨. 기존 추론은 타임스텝
  라벨이 온전히 존재한다고 암묵적으로 가정했지만, 신호의 1/4에 라벨이 없다.
- 기본활동(1–6) 세그먼트: n=856, 중앙값 17.2 s; L=128 윈도우(10.24 s)보다
  짧은 것 5%, L=256 윈도우(20.5 s)보다 짧은 것 70%.
- 피험자당 기본 세그먼트 ~28개(촘촘함: 28/28/33).
- 활동 순서(exp01): `VOID→STAND→[STAND_TO_SIT]→SIT→[SIT_TO_STAND]→STAND→…
  →WALK→VOID→WALK→VOID→WALK…` — 정적 활동은 전환 클래스를 통해 연속으로
  이어지고, **동적 활동(걷기/계단)은 VOID로 분리된 짧은 버스트로 등장**.

![HAPT 클래스 분포](figs/hapt_class_dists.png)

정적 클래스 SIT/STAND/LAY는 모두 ≈1.0 g(중력) — 중력의 **방향**(자세)으로만
구분되며, 3축 샘플 하나로 즉시 읽힌다; 동적 활동은 크기 분산이 넓다. 즉
"활동"은 상당 부분 단기/순간 판독이며, 이는 직전 턴의 anchor-probe가 측정한
바와 정확히 일치한다.

### PTB-XL

기록당 진단, 10 s 내내 일정. NORM = "정상 ECG"로 여러 scp_code 중 하나
(상위 동시코드: SR, NDT, ASMI, LVH…); "non-NORM"은 모든 비정상을 합친 것.
전체 DB 균형 9514 NORM / 12285 non-NORM; prep은 2500/2500으로 강제 균형.
**기록 내 시간 주석 없음**(`burst_noise` 플래그만). 4개 중 가장 거친 입도.

### XJTU

라벨 = **life fraction = 스냅샷_인덱스/(n−1)** — 런 내 정규화 위치이며
**물리적 RUL이 아님**. 80 ms 윈도우 내에서는 일정. 스냅샷은 1분 간격;
prep은 베어링당 16개를 균등 간격으로 추출.

### Synthetic

정답이 **타임스텝별**: regime `s(t)`(3-state 마르코프), OU 인자 `u(t)`,
위상 `phi(t)`. 4개 중 가장 미세한 입도 — 유일하게 조밀하고 무잡음인 slow 라벨.

![Synthetic 윈도우](figs/synth_window.png)

---

## STEP 3 — 시각화

**HAPT 윈도우, 클린 vs 오염:**

![HAPT 윈도우](figs/hapt_window_transition.png)

"오염된" 윈도우의 오염원은 **VOID**(회색)이지 두 번째 활동이 아니다 —
첫 ~5 s는 미라벨 동작이 STAND로 안정화되는 구간. 이것이 전형적 케이스
(STEP 4 참조).

**PTB-XL 기록**(`figs/ptbxl_records.png`)과 **XJTU 스냅샷:**

![XJTU 스냅샷](figs/xjtu_snapshots.png)

수명 초기 = 저진폭 광대역 잡음; 수명 후기 = 진폭 포락선이 커지는 강한 주기적
결함 임펄스. 열화(slow 인자)는 스냅샷 **사이**에서 분(minute) 스케일이지만
**80 ms 윈도우 내에서는 일정** — 윈도우 내부에 이용 가능한 시간스케일 갭이
없다는 predicted-negative의 근거.

**ACF, 전 데이터셋, 로그-lag, raw + 파생열:**

![ACF 전체](figs/acf_all.png)

- **PTB-XL은 ~90–100 lag에서 주기적 결맞음 스파이크 = 심박**
  (~0.8–1 s @100 Hz) — 진짜 coherent-periodic 성분, 스크린 Check-1 대상.
  last-crossing T_ac 규칙은 심박 하모닉을 붙잡아 187로 부풀린다.
- **XJTU raw T_ac = 1999는 잡음바닥 아티팩트**; 실제 캐리어는 ~7 샘플에
  탈상관. last-crossing 규칙이 여기서 취약.
- HAPT |acc|와 synthetic x는 진동(캐리어 유래) ACF에 감쇠 포락선 —
  decay-but-slow이지 coherent-periodic 아님.

---

## STEP 4 — prep 산출물 vs 원본 진실

**HAPT.** 374분 원신호에서 prep은 **2926 윈도우** 유지; end-point가 VOID인
1109개, 전환 클래스(7–12)인 261개 윈도우를 스킵. 파일/피험자 경계를 넘는
윈도우 0개(루프가 파일 단위). 윈도우별 end-라벨 커버리지: 중앙값 0.867,
평균 0.716, **유지 윈도우 중 완전 클린은 42.7%뿐.**

**57.3% "오염"** 수치의 분해 (기존 해석 수정):

| 오염 종류 | 개수 | 유지분 대비 |
|---|---|---|
| clean (스팬 전체 단일 라벨) | 1250 | 0.427 |
| **void-only** (다른 곳에 라벨 0) | 733 | **0.251** |
| **transition-class** (스팬에 7–12) | 589 | **0.201** |
| 다른 기본활동 (1–6) | 354 | **0.121** |

즉 57% 중 **활동-대-활동 혼입은 12.1%뿐**; 대부분은 VOID(25%)나 라벨된
자세전환(20%). "윈도우가 활동 전환을 가로지른다"는 잘못된 설명이었다.

정규화: HAPT는 **윈도우별·축별 시간축 z-score** `((w−mean(0))/std(0))` —
각 축의 윈도우 내 평균·스케일을 제거(절대 자세/중력방향이 일부 정규화되어
사라짐; `accmag` fast 라벨은 정규화 전 *원본* acc의 end-point에서 취함).
PTB-XL: 기록별 z-score, lead II만, 샘플 드롭 없음, ~16.8k 기록 미사용.
XJTU: 스냅샷별 z-score, Horizontal 채널만, 스냅샷의 ~90% 드롭.

---

## STEP 5 — 기존 가정과 모순되는 지점

| 기존 주장 | 판정 | 수정 |
|---|---|---|
| 라벨은 타임스텝별이고 온전히 존재 | **부분적으로 틀림** | 타임스텝별은 맞지만 **HAPT의 27%가 VOID**; 클래스도 **6개가 아니라 12개** |
| "윈도우의 57%가 전환을 가로지름" | **수치는 맞고 의미는 틀림** | 활동-대-활동은 **12%뿐**; void-only 25%, 전환클래스 20% |
| end-point 라벨링(`per[end]`) | **확인됨** | — |
| 윈도우-vs-dwell(중앙 dwell 17.2 s) | **불완전** | 세그먼트 중앙값은 맞지만 활동들이 **VOID로 분리**돼 연속이 아님 — "윈도우가 dwell 안에 있다"는 27% 갭을 무시 |
| HAPT는 "PTB-XL 같은 static" (그 이전 판단) | **여전히 독자 범주** | 오염은 실재하나 VOID/전환 위주이지 활동혼입이 아님 |

이전에 기록되지 않았던 새 사실: **PTB-XL 심박은 coherent-periodic 성분**
(스크린 Check-1 관련); **XJTU life = 스냅샷 인덱스이지 물리적 RUL 아님**;
last-crossing T_ac는 PTB-XL(심박)·XJTU(잡음)에서 **아티팩트에 취약**;
**정적 활동은 중력 방향으로만 구분**(순간 판독 가능 — end-region readout
발견 강화); gyro/Vertical 채널/16.8k PTB-XL 기록 미사용; HAPT 정규화가
윈도우 내 축별 스케일을 제거.
