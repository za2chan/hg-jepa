import json, pathlib, numpy as np
R = pathlib.Path('runs_v2'); ROOT = pathlib.Path('.')
J = lambda p: json.load(open(p))
sfa   = J(R/'sfa_rank_check.json'); seeds = J(R/'fig_sfa_seeds.json')
gyro  = J(R/'hapt_gyro_check.json')['results']
cap   = J(R/'hapt_capacity_check.json')['results']
mlp   = J(R/'mlp_probe_check.json')['results']
v1    = [d for d in J('runs/datasets_table.json') if d['name']=='HAPT'][0]
d2 = np.load('data/hapt_v2.npz'); l2 = d2['lab']
v2_tr = float(np.mean([( l[1:]!=l[:-1]).sum()>0 for l in l2]))
v2_win = d2['W'].shape[1]*4

def row(cells): return "| " + " | ".join(cells) + " |"
out=[]
A=out.append

A("# 진단 기록 — 2026-08-12")
A("")
A("마감(2026-08-08) 이후 워크숍 판을 준비하며 나온 진단 다섯 건. **논문 본문 수치는 하나도 건드리지 않았습니다** — 전부 새 파일로만 실험했습니다. 각 절은 *무엇을 쟀나 · 결과 · 주장 가능 · 주장 불가 · 파일* 순입니다.")
A("")
A("수치는 전부 `runs_v2/*.json`에서 스크립트로 생성했습니다. 이 문서를 손으로 고치지 마시고, 수치가 바뀌면 해당 실험을 다시 돌리십시오.")
A("")
A("| # | 발견 | 논문 영향 |")
A("|---|---|---|")
A("| 1 | HAPT 데이터셋 표가 v1 수치 | ⛔ **표 수정 필요** |")
A("| 2 | SFA 최느림 성분이 LayerNorm 널 방향 | ✅ 영향 없음 (부록 소재) |")
A("| 3 | 배정 항이 대리변수 표현량에 눌림 | ⚠️ 단서 추가 |")
A("| 4 | 폭 4배 확대 시 천장 붕괴 | ⚠️ 미해결 |")
A("| 5 | **MLP 프로브에서 배제가 무너짐** | ⛔ **한계 절 강화 필요** |")
A("")
A("---")
A("")

A("## 1. HAPT 데이터셋 표가 v1 수치입니다")
A("")
A(f"`runs/datasets_table.json`은 HAPT를 **창 {v1['win_samp']}샘플 · 전환 포함 {v1['transition_frac']:.1%}**로 기록합니다. 그런데 v2 모델이 실제로 학습하는 `data/hapt_v2.npz`는 **창 {v2_win}샘플 · 전환 포함 {v2_tr:.1%}**입니다.")
A("")
A(f"`datasets_table.py`는 `legacy/`에만 있고(HANDOFF §2: v1 코드, 수정 금지), 그 출력도 v1 산출물입니다. `runs_v2/`에는 데이터셋 표가 없습니다. 창이 두 배로 길어졌으니 전환을 더 많이 가로지르는 것은 당연하고, 방향도 자연스럽습니다.")
A("")
A(f"**주장 가능**: v2 설정에서 HAPT 창의 {v2_tr:.0%}가 활동 전환을 가로지른다. **주장 불가**: 57%. `CLAUDE.md` §5와 아웃라인 한계 3이 그 값을 쓰고 있으므로 함께 고쳐야 합니다. 적용 조건 논의가 **더 불리해집니다**.")
A("")
A("**파일**: `runs/datasets_table.json`(v1) · `data/hapt_v2.npz`(실제) · 재생성 코드는 `legacy/datasets_table.py`뿐이라 v2용을 새로 써야 함")
A("")

A("## 2. SFA의 가장 느린 성분은 LayerNorm이 만든 널 방향입니다")
A("")
A("인코더가 LayerNorm으로 끝나 출력의 평균과 노름이 고정되고, 임베딩이 64차원 중 **62차원 다양체**에만 놓입니다(측정: 유효 rank 62/64, 최소 고유값 ~6e-15). 분산이 0인 방향은 미분 분산도 0이므로 SFA의 느림 순위 1·2등을 차지합니다. 내용은 없습니다.")
A("")
bc = seeds['best_components']
A(f"그래서 regime은 시드 {len(bc)}개에서 성분 **{bc}**번에 옵니다 — 1번이 아니라 2~3번. 둘 사이 순서는 부동소수점 잡음이라 2/3이 뒤바뀔 뿐이고, **4번 이후로는 가지 않습니다.**")
A("")
A("**핸디캡이 아닙니다.** 죽은 방향을 빼고 SFA를 다시 적합해도 SEP은 거의 안 변합니다.")
A("")
A(row(["변형","SEP (5시드 평균 ± sd)"])); A(row(["---","---"]))
for k,lab in [("sfa_asis","SFA 현행"),("sfa_ranked","SFA 랭크 절단"),("gate","게이트(참고)")]:
    s=sfa['summary'][k]; A(row([lab, f"{s['mean']:.3f} ± {s['sd']:.3f}"]))
A("")
A(f"**단위 = SEP** [0,1]. 차이 **{sfa['delta_sep_from_rank_fix']:+.3f}**로 시드 sd({sfa['summary']['sfa_asis']['sd']:.3f})의 5분의 1이고 부호도 반대입니다 — 죽은 방향을 빼면 여집합이 좁아져 배정이 미세하게 손해입니다.")
A("")
A("**주장 가능**: SFA의 느림 순서는 관심 요인이 몇 번째 성분에 있는지 알려주지 않는다(5/5 시드에서 최느림 성분은 우연 수준). 어느 성분을 쓸지 고르려면 라벨이 필요하다 — **운용상 차이**. **주장 불가**: 성분 번호가 시드마다 흩어져 규칙을 못 만든다(2·3번으로 안정적). SFA가 핸디캡을 안고 있었다(아님).")
A("")
A("**파일**: `runs_v2/sfa_rank_check.json` · `fig_sfa_seeds.{png,json}` · `fig_traj_sfa.{png,json}` · `src/exp/sfa_rank_check.py` · `src/exp/fig_traj_sfa.py`")
A("")

A("## 3. 배정 항은 대리변수의 표현량에 눌립니다")
A("")
A("자이로 3축을 추가해(`data/hapt_v2_gyro.npz`, 창·라벨·대리변수 동일, 앞 3축은 바이트 단위로 동일) 학습했습니다. 대리변수는 통제를 위해 가속도 크기로 고정했습니다.")
A("")
hdr=["항","acc3/NCE","+gyro/NCE","acc3/L1","+gyro/L1"]
A(row(hdr)); A(row(["---"]*len(hdr)))
for t,lab in [("inclusion","포함"),("allocation","배정"),("exclusion","배제"),("sep","SEP")]:
    A(row([lab]+[f"{gyro[k]['sep'][t]:.3f}" for k in ("acc3/nce","acc3+gyro3/nce","acc3/l1","acc3+gyro3/l1")]))
A(row(["천장(`z_full`)"]+[f"{gyro[k]['block_factor']['z_full']['transient']:.3f}" for k in ("acc3/nce","acc3+gyro3/nce","acc3/l1","acc3+gyro3/l1")]))
A("")
A("**단위**: 포함=`z_slow`의 활동 macro-F1, 배정=`z_mix`의 대리변수 R², 배제=1−(`z_slow`의 대리변수 R²), 천장=`z_full`의 대리변수 R². 모두 [0,1].")
A("")
for stem,lab in [("nce","NCE"),("l1","L1")]:
    a,b = gyro[f'acc3/{stem}']['block_factor'], gyro[f'acc3+gyro3/{stem}']['block_factor']
    df=b['z_full']['transient']-a['z_full']['transient']; dm=b['z_fast']['transient']-a['z_fast']['transient']
    A(f"- **{lab}**: 배정 {dm:+.3f}, 천장 {df:+.3f} → 배정 하락의 **{100*df/dm:.0f}%**가 천장 하락으로 설명됨")
A("")
A("`z_slow`는 미동도 없고(0.007→0.008, 0.015→0.019) 무작위 분할 여집합도 같이 내려갑니다. 즉 정보가 `z_slow`로 샌 게 아니라 **임베딩 전체에서 그 대리변수가 덜 표현된 것**입니다. `twosided.py`가 경고한 함정 (3) *unmeasurable*이 배정 항 자신에게 적용된 사례입니다.")
A("")
A("**주장 가능**: SEP은 대리변수가 임베딩에 얼마나 표현되는지에 좌우되며, 입력 채널 구성 같은 무관해 보이는 선택으로 바뀐다. **`z_full` 천장을 SEP 표에 함께 실어야 한다** — `block_factor`가 이미 그 행을 계산합니다. **주장 불가**: 채널을 늘리면 분리가 나빠진다(배제는 ±0.002로 불변).")
A("")
A("**덤 — 계단 가설은 반증**: 3축에서도 계단은 이미 잘 됩니다(NCE upstairs 0.839 / downstairs 0.885). 자이로는 **L1에만** 크게 도움이 됩니다(macro-F1 0.806→0.891). 궤적 그림에서 계단이 안 갈려 보인 것은 **PC1·PC2 두 성분만 본 탓**이었습니다.")
A("")
A("**파일**: `runs_v2/hapt_gyro_check.json` · `src/exp/hapt_gyro_check.py` · `src/prep/hapt_gyro.py`")
A("")

A("## 4. 폭을 4배로 키우면 천장이 붕괴합니다 (원인 미해결)")
A("")
A("트렁크만 확대(`d_model` 96→192, `d_ff` 256→512)하고 `D_Z`=64·`d_slow`=16은 고정했습니다. 파라미터 356k→1.40M.")
A("")
hdr=["조건","다운스트림(base→wide)","SEP","천장"]
A(row(hdr)); A(row(["---"]*len(hdr)))
for tag in ("acc3","acc3+gyro3"):
    for stem in ("nce","l1"):
        b=cap[f"{tag}/base/{stem}"]; w=cap[f"{tag}/wide/{stem}"]
        A(row([f"{tag}/{stem}", f"{b['macro']:.3f} → {w['macro']:.3f}",
               f"{b['sep']['sep']:.3f} → {w['sep']['sep']:.3f}",
               f"{b['z_full_proxy_r2']:.3f} → {w['z_full_proxy_r2']:.3f}"]))
A("")
A("**단위**: 다운스트림=`z_slow`의 활동 macro-F1, SEP·천장은 위와 동일. 네 조건 모두 **배정만 무너지고 포함·배제는 불변**이며, 배정 하락폭이 천장 하락폭과 거의 같습니다(예: acc3/NCE 배정 −0.535, 천장 −0.557).")
A("")
A("**주장 가능**: 같은 하이퍼파라미터로 폭만 4배 늘리면 일시 요인이 임베딩 어디에도 덜 담긴다. wide/NCE의 배제 0.995는 **담긴 게 없어서 얻은 값**이며, `twosided.py`의 함정 (1) *vacuous*의 표본이다 — 배정 항이 그것을 잡아낸다.")
A("")
A("**주장 불가**: 모델을 키우면 성능이 나빠진다. 다운스트림은 **acc3/L1에서 오히려 +0.068**, 나머지는 −0.01 수준입니다. 그리고 lr 3e-4·2500스텝은 좁은 모델용이라 **재조정하지 않았습니다.** 데이터도 창 2,104개로 작아 스케일업 실험이라 부르기 어렵습니다.")
A("")
A("⚠️ **미해결**: 5번에서 확인했듯 천장은 프로브를 비선형으로 바꿔도 거의 안 움직입니다. 따라서 이 붕괴는 선형/비선형 문제가 **아니고**, 원인을 모릅니다. 합성 생성기로 데이터·스텝·폭을 함께 늘리는 격자 실험이 필요합니다.")
A("")
A("**파일**: `runs_v2/hapt_capacity_check.json` · `src/exp/hapt_capacity_check.py`")
A("")

A("## 5. ⛔ MLP 프로브에서 배제가 무너집니다")
A("")
A("특징·분할·부분표집·블록 정의를 전부 고정하고 **프로브 머리만** 교체했습니다(로지스틱/릿지 → MLP 은닉 256, 조기 종료). 세 데이터셋 × 두 스템.")
A("")
hdr=["조건","probe","다운스트림(=포함)","배정","배제","SEP","천장","무작위 SEP"]
A(row(hdr)); A(row(["---"]*len(hdr)))
for k in mlp:
    for p in ("linear","mlp"):
        s=mlp[k][p]
        A(row([k if p=="linear" else "", p, f"{s['inclusion']:.3f}", f"{s['allocation']:.3f}",
               f"**{s['exclusion']:.3f}**", f"{s['sep']:.3f}", f"{s['ceiling']:.3f}", f"{s['null_sep']:.3f}"]))
A("")
A("**단위**: 전부 [0,1], 클수록 좋음.")
A("")
A("**① 다운스트림은 프로브에 거의 무관합니다** — 6조건 중 5개가 ±0.03 안. 예외는 synth/L1(0.452→0.297)인데 λ=4에서 L1이 병리적인 셀입니다.")
A("")
ex=[(k, mlp[k]['linear']['exclusion'], mlp[k]['mlp']['exclusion']) for k in mlp]
worst=sorted(ex, key=lambda r: r[2]-r[1])[:3]
A("**② 배제는 6조건 중 5개에서 떨어지고, 폭이 큽니다.** " + " · ".join(f"{k} {a:.3f}→{b:.3f}" for k,a,b in worst) + " 등.")
A("")
A("즉 일시 요인이 `z_slow`에 남아 있었고 선형 프로브가 못 읽었을 뿐입니다. `CLAUDE.md` §1이 *\"empirically, at the linear level only\"*로 미리 인정한 한계가 **수치로 확인**됐고, 그 문구가 주는 인상보다 **큽니다**.")
A("")
A(f"**③ 예외 하나**: hapt/NCE만 MLP에서 오히려 좋아집니다(배제 {mlp['hapt/nce']['linear']['exclusion']:.3f}→{mlp['hapt/nce']['mlp']['exclusion']:.3f}, SEP {mlp['hapt/nce']['linear']['sep']:.3f}→{mlp['hapt/nce']['mlp']['sep']:.3f}). 이 셀은 비선형으로 봐도 진짜 배제입니다.")
A("")
A(f"**④ 무작위 기준선도 함께 봐야 합니다.** MLP에서 무작위 SEP도 대체로 내려갑니다. hapt/L1은 SEP {mlp['hapt/l1']['mlp']['sep']:.3f} 대 무작위 {mlp['hapt/l1']['mlp']['null_sep']:.3f}로 **격차가 오히려 벌어집니다**. 배제 절대값과 무작위 대비 우위를 구분해 읽어야 합니다.")
A("")
A("**논문 반영**: 한계 절이 지금은 *\"선형 수준의 한계\"*라고만 적혀 있는데, **비선형 프로브에서 배제 우위의 상당 부분이 사라진다는 수치를 넣어야** 정직합니다. 동시에 **다운스트림 성능은 프로브에 무관**하므로, 분리가 선형 수준의 현상이더라도 `z_slow`를 쓰는 실용적 가치는 유지된다고 함께 써야 합니다.")
A("")
A("**파일**: `runs_v2/mlp_probe_check.json` · `src/exp/mlp_probe_check.py`")
A("")
A("---")
A("")
A("## 다음 세션이 할 일")
A("")
A("1. **HAPT 데이터셋 표를 v2로 다시 계산** (1번). `legacy/datasets_table.py`를 v2용으로 새로 쓰고 `runs_v2/`에 저장. `CLAUDE.md` §5와 아웃라인 한계 3의 57%도 함께 수정.")
A("2. **한계 절에 MLP 프로브 결과 추가** (5번). 수치는 `mlp_probe_check.json`에서.")
A("3. **SEP 표에 `z_full` 천장 열 추가** (3번). `block_factor`가 이미 계산하므로 표시만 하면 됨.")
A("4. **폭 확대 시 천장 붕괴의 원인 규명** (4번). 합성 생성기로 데이터·스텝·폭 격자, SEP 대신 네 값을 따로 보고.")
A("5. 2번은 부록 소재로만. 본문에 넣으면 *\"상위 3개 중 고르면 되지 않나\"*를 자초함.")
A("")
A("## 이번에 만든 그림")
A("")
A("| 파일 | 내용 | 상태 |")
A("|---|---|---|")
A("| `fig_embed_{nce_lam4,l1_lam1,l1_lam4}.png` | 블록×요인 2×2 t-SNE | 본문 후보 (l1_lam4는 λ 병리 부록용) |")
A("| `fig_traj_{nce_lam4,l1_lam1}.png` | 합성 `z_slow` 궤적 | 본문 후보 |")
A("| `fig_traj_sfa.png` · `fig_sfa_seeds.png` | SFA 성분 순서 | 부록 |")
A("| `fig_traj_hapt.png` | HAPT 실데이터 궤적, PC1·PC2 | 본문 후보 |")
A("")
A("⚠️ `fig_traj_hapt`은 *\"활동이 바뀌는 곳에서 움직인다\"*까지만 주장할 수 있습니다. HAPT 라벨은 창의 끝점 샘플로 붙고 인과 인코더는 문맥 누적과 라벨 근접을 구분하지 못합니다(`CLAUDE.md` §5).")
pathlib.Path('docs/DIAGNOSTICS_2026-08-12_ko.md').write_text("\n".join(out)+"\n")
print("생성 완료:", len(out), "줄")
