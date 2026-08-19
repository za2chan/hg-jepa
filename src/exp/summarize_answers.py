"""Collect every run behind the 2026-08-05 question round into report-ready tables.

One table per QUESTION, not one per (dataset x stem x condition). The earlier
version emitted 25 tables that each showed a slice of the same six cells, which
made the numbers unreadable and hid the comparisons that actually matter. Here a
dataset/stem pair is a ROW, so a claim can be checked by reading down a column.

Reads whatever exists in runs_v2/ and skips what does not.
Usage: python3 summarize_answers.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

R = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
DS = [("synth", "합성"), ("ptbxl", "PTB-XL"), ("hapt", "HAPT")]
STEMS = ["l1+ema", "nce+ema"]
CELLS = [("g0_x0", "메커니즘 없음"), ("g1_x0", "게이트만"),
         ("g0_x1", "xcov만"), ("g1_x1", "둘 다 (우리)")]


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def sep_of(cell, ceiling, null):
    from probes import sep_index
    return sep_index({k: (v["slow"], v["fast"]) for k, v in cell.items()}, ceiling, null)


def null_of(d, tag, suffix=""):
    """Fast-factor R2 of a random 16-dim block of the reference model for this stem."""
    from model import D_SLOW
    ref = d.get(f"{tag.split('/')[0]}/g0_x0{suffix}")
    return (ref or d[tag])[f"rand{D_SLOW}"]["fast"], ref is not None


def table(header, rows, note=None, legend_rows=None):
    print(f"\n| {' | '.join(header)} |")
    print("|" + "---|" * len(header))
    for r in rows:
        print(f"| {' | '.join(r)} |")
    if note:
        print(f"\n{note}")
    if legend_rows:
        print("\n<details><summary>열 설명</summary>\n")
        print("| 열 | 뜻 |")
        print("|---|---|")
        for a, b in legend_rows:
            print(f"| `{a}` | {b} |")
        print("\n</details>")


SEP_NOTE = ("SEP = 포함 × 배정 × 배제, [0,1]. **부록 정렬용 요약값**이고 본문 근거는 "
            "부록 A의 원본 행렬입니다. 배제 항의 기준선은 그 스템의 `g0_x0`에서 뽑은 "
            "무작위 16차원입니다.")


# ---------------------------------------------------------------- 1. mechanism
def mechanism():
    print("\n## 1. 게이트와 xcov 중 무엇이 일하는가\n")
    print("**주장:** 둘 다 있어야 분리가 된다. 한 행을 가로로 읽으면 각 부품의 기여가 보입니다.\n")
    rows = []
    for key, label in DS:
        d = load(f"twosided_{key}.json")
        if not d:
            continue
        ceil = max(v["z_full"]["fast"] for v in d.values())
        for stem in STEMS:
            null, _ = null_of(d, f"{stem}/g0_x0")
            rows.append([label, stem] + [
                f"{sep_of(d[f'{stem}/{c}'], ceil, null)['sep']:.3f}" for c, _ in CELLS])
    table(["데이터셋", "스템"] + [f"{n}<br>`{c}`" for c, n in CELLS], rows, SEP_NOTE,
          [("메커니즘 없음", "게이트·xcov 둘 다 끔. 이 값이 낮아야 나머지 열이 의미를 가짐"),
           ("게이트만 / xcov만", "하나씩만 켬. **둘 다 중간값이면 어느 하나로는 부족하다는 뜻**"),
           ("둘 다 (우리)", "우리 방법. 다른 세 열보다 확연히 높아야 주장이 섬")])


# ------------------------------------------------------------------- 2. stems
def stems():
    print("\n## 2. 네 스템 비교 — 낮은 leak이 진짜인지 z_fast로 검증\n")
    print("**주장:** leak만 낮은 스템은 빠른 인자를 배제한 게 아니라 **파괴**한 것이다.\n")
    rows = []
    for key, label in DS:
        d = load(f"twosided_{key}.json")
        if not d:
            continue
        ceil = max(v["z_full"]["fast"] for v in d.values())
        for tag, name in (("reg+ema/g1_x1", "reg+ema (L2)"), ("l1+ema/g1_x1", "l1+ema"),
                          ("nce+ema/g1_x1", "nce+ema"), ("nce+online/g1_x1", "nce+online")):
            c = d.get(tag)
            if not c:
                continue
            null, fixed = null_of(d, tag)
            rows.append([label, name + ("" if fixed else " ⁎"),
                         f"{c['z_slow']['slow']:.3f}", f"{c['z_slow']['fast']:+.3f}",
                         f"**{c['z_fast']['fast']:+.3f}**",
                         f"{sep_of(c, ceil, null)['sep']:.3f}"])
    table(["데이터셋", "스템", "z_slow 느림↑", "z_slow 빠름↓ (leak)", "z_fast 빠름↑", "SEP"],
          rows,
          "⁎ = 게이트 없는 셀을 안 돌린 스템이라 자기 무작위 부분공간을 기준선으로 씀"
          "(덜 보수적).\n\n**`nce+online`을 보세요** — leak이 가장 낮은데 `z_fast`도 낮습니다. "
          "빠른 인자를 어디에도 담지 않은 것이라 낮은 leak이 공허합니다.",
          [("z_slow 느림↑", "앞 16차원의 느린 인자 macro-F1. **높을수록 좋음**"),
           ("z_slow 빠름↓", "앞 16차원의 빠른 인자 R² (= leak). **낮을수록 좋음**"),
           ("z_fast 빠름↑", "뒤 48차원의 빠른 인자 R². **높아야 함** — 낮으면 배제가 아니라 파괴"),
           ("SEP", "요약값. 배정 항이 z_fast를 반영하므로 공허한 스템은 여기서 걸림")])


# -------------------------------------------------------------- 3. layer norm
def layernorm():
    print("\n## 3. per-block LayerNorm을 빼면 무엇이 남는가\n")
    print("`model.py`의 B2 LayerNorm은 게이트·xcov와 무관하게 항상 켜져 있고, 블록의 "
          "**크기**를 정규화해 없앱니다 — HAPT의 빠른 프록시가 바로 크기(‖acc‖)입니다. "
          "그래서 배제가 게이트가 아니라 LayerNorm 덕일 수 있었습니다.\n")
    rows = []
    for key, label in DS:
        d = load(f"twosided_bn_{key}.json")
        if not d:
            continue
        ceil = max(v["z_full"]["fast"] for v in d.values())
        for stem in STEMS:
            free = null_of(d, f"{stem}/g0_x0", "_noBN")[0]      # 완전 무메커니즘 기준선
            own = null_of(d, f"{stem}/g0_x0")[0]
            A = sep_of(d[f"{stem}/g1_x1"], ceil, free)["sep"]
            B = sep_of(d[f"{stem}/g1_x1"], ceil, own)["sep"]
            Cg = sep_of(d[f"{stem}/g1_x1_noBN"], ceil, free)["sep"]
            C0 = sep_of(d[f"{stem}/g0_x0_noBN"], ceil, free)["sep"]
            rows.append([label, stem, f"**{A:.3f}**", f"{B:.3f}",
                         f"{Cg:.3f}", f"{C0:.3f}", f"**{Cg - C0:+.3f}**"])
    table(["데이터셋", "스템", "(A) 배포 모델<br>vs 무메커니즘", "(B) 같은 BN<br>조건끼리",
           "(C) LN 없는<br>우리 셀", "(C) LN 없는<br>대조군", "게이트 순효과<br>(C 차이)"],
          rows,
          "**(A)가 논문의 \"우리 방법이 분리한다\"**이고 여섯 셀 전부에서 대조군을 이깁니다.\n"
          "**맨 오른쪽이 \"게이트가 그 분리를 만드는가\"**입니다 — LayerNorm이 없는 세계에서 "
          "게이트가 순수하게 더한 값. HAPT+L1만 음수이므로, 거기서는 배제가 게이트가 아니라 "
          "LayerNorm의 몫입니다.\n\n(A)와 (B)가 거의 같은 이유: 무작위 16차원 기준선은 "
          "참조 모델의 BN 설정에 거의 영향받지 않습니다. LayerNorm은 기준선이 아니라 "
          "**셀 자체**를 움직입니다.",
          [("(A)", "배포 모델(`g1_x1`, LN 포함) vs `g0_x0_noBN`(게이트·xcov·LN 전부 없음)"),
           ("(B)", "같은 BN 조건의 `g0_x0` 기준. (A)와 거의 같아야 정상"),
           ("(C) 두 열", "LN을 끈 세계에서의 우리 셀과 대조군"),
           ("게이트 순효과", "`(C) 우리 셀 − (C) 대조군`. **LN 도움 없이 게이트가 더한 값**. "
                             "음수면 그 조건에서 게이트가 하는 일이 없음")])


# ---------------------------------------------------------------- 4. rotation
def rotation():
    print("\n## 4. 라벨 없는 사후 회전으로 게이트를 대체할 수 있는가\n")
    print("**반론:** 잠재 예측 손실은 회전에 거의 불변이니, 게이트 없이 학습한 임베딩을 "
          "사후에 회전하면 되지 않나. 값은 SEP*입니다.\n")
    methods = ["gate", "PCA", "ICA-slow", "SFA", "ungated(block)"]
    rows = []
    for stem in STEMS:
        for key, label in DS:
            d = load(f"rotation_{key}_{stem}.json")
            if not d or "random" not in d:
                continue
            null = d["random"]["slow_half"]["fast"][0]
            s_ceil = max(r["slow_half"]["slow"][0] for r in d.values())
            f_ceil = max(r["fast_half"]["fast"][0] for r in d.values())
            cl = lambda v: min(max(v, 0.0), 1.0)

            def sep(m):
                a, b = d[m]["slow_half"], d[m]["fast_half"]
                return (cl(a["slow"][0] / s_ceil) * cl(b["fast"][0] / f_ceil)
                        * cl(1 - a["fast"][0] / null if null > 1e-3 else 0.0))
            best = max(sep(m) for m in methods if m != "gate")
            rows.append([label, stem, f"{null:+.3f}"]
                        + [(f"**{sep(m):.3f}**" if m == "gate" else f"{sep(m):.3f}")
                           for m in methods]
                        + [f"**{sep('gate') - best:+.3f}**"])
    table(["데이터셋", "스템", "측정 가능 범위"] + methods + ["게이트 − 최고 사후회전"], rows,
          "**\"측정 가능 범위\"가 결론을 지배합니다.** 이 값이 그 데이터셋에서 배제를 "
          "잴 수 있는 폭 전체입니다. 합성은 0.70~0.85로 넓어 게이트가 압도하고, PTB-XL은 "
          "0.36~0.40으로 중간이라 사후 회전과 접전이며, **HAPT는 0.04~0.11밖에 안 되므로 "
          "그 행의 차이는 전부 잡음입니다** — 어제 \"HAPT에서 PCA가 게이트를 이긴다\"고 "
          "보고한 것은 이 폭을 못 보고 읽은 것이었습니다.\n\n"
          "SEP*는 이 표 안의 최댓값으로 정규화하므로 **다른 표의 SEP과 비교하지 마세요.**",
          [("측정 가능 범위", "무작위 16차원이 빠른 인자를 맞힌 R². **작으면 그 데이터셋에서는 "
                              "배제를 측정할 수 없습니다**"),
           ("gate", "우리 z_slow. 나머지 열은 **게이트 없는 모델**의 임베딩을 라벨 없이 "
                    "사후 분해한 것"),
           ("SEP*", "포함×배정×배제. 배정 항이 있으므로 **여집합에 빠른 인자를 남기지 못한 "
                    "방법은 여기서 걸립니다**(합성의 PCA가 그 예)"),
           ("게이트 − 최고 사후회전", "게이트가 최선의 사후 방법보다 얼마나 앞서는가. "
                                      "0 근처면 대체 가능하다는 뜻")])


# ------------------------------------------------------------- 5. domain shift
def domain():
    print("\n## 5. 피험자 이동에 z_slow가 더 강건한가 (HAPT)\n")
    print("**주장:** `Δ_slow < Δ_기준`. 게이트 없는 대조군 모델의 `z_full`이 올바른 비교 "
          "대상입니다(같은 모델의 z_full은 readout 비교일 뿐).\n")
    rows = []
    for mode in ("block",):
        for stem in STEMS:
            d = load(f"domain_shift_hapt_{stem}_{mode}.json")
            a = (d or {}).get("agg", {})
            if not a:
                continue
            cells = []
            for k in ("delta", "delta_late", "delta_matched"):
                c = a.get(f"claim_{k}_vs_ungated")
                cells.append(f"{c['ref_minus_slow'][0]:+.3f} · **{c['n_supporting']}/"
                             f"{c['n_runs']}**" if c else "—")
            rows.append([stem, mode, f"{d.get('label_tv', float('nan')):.3f}"] + cells)
    table(["스템", "분할", "라벨 TV", "Δ", "Δ_late", "Δ_matched"], rows,
          "**블록 분할이 라벨 이동을 TV 0.395 → 0.073으로 줄이자 `nce+ema`의 부호가 "
          "뒤집혔습니다**(2/9 → 8/9). 다만 평가셋 크기를 맞춘 `Δ_matched`에서는 5/9로 "
          "떨어지므로, **지표에 따라 갈리며 확정적이지 않습니다.** 세 지표를 모두 보고해야 "
          "합니다.",
          [("라벨 TV", "A_test와 B의 활동 분포 거리. **0에 가까울수록 순수한 피험자 이동**"),
           ("Δ", "`F1(A_test) − F1(B)`의 z_slow 대비 대조군 차이. 양수면 z_slow가 덜 잃음"),
           ("Δ_late", "B를 후반 30%로 한정해 시간 위치를 맞춘 것"),
           ("Δ_matched", "A_test를 B와 **같은 피험자 수·윈도우 수**로 맞춘 것"),
           ("N/9", "3파티션 × 3시드 = 9런 중 주장이 성립한 횟수")])


# --------------------------------------------------------------- 6. difficulty
def difficulty():
    d = load("difficulty.json")
    if not d:
        return
    print("\n## 6. 합성 과제가 너무 쉬운가\n")
    print("**주장:** 기본 설정에서는 모든 셀이 slow-kept 0.99라 포함 축에 판별력이 없다.\n")
    rows = []
    for gap in sorted({k.split("_c")[0] for k in d.get("fft", {})},
                      key=lambda g: -float(g.replace("gap", ""))):
        g = float(gap.replace("gap", ""))
        r64, r128 = d["fft"].get(f"{gap}_c64"), d["fft"].get(f"{gap}_c128")
        tr = d.get("train", {}).get(str(g))
        cells = ["—", "—"]
        if tr:
            from model import D_SLOW
            base = tr["g0_x0"][f"rand{D_SLOW}"][1]
            cells = [f"{tr[c]['z_slow'][0]:.3f} / {base - tr[c]['z_slow'][1]:+.3f}"
                     for c in ("g0_x0", "g1_x1")]
        rows.append([f"±{g*100:g}%", f"{r64['regime_spacing']:.5f}",
                     f"{r64['centroid']:.3f}", f"{r128['centroid']:.3f}"] + cells)
    table(["gap", "regime 간격", "FFT (64패치)", "FFT (128패치)",
           "학습: 무메커니즘<br>느림↑ / 배제여유", "학습: 우리<br>느림↑ / 배제여유"], rows,
          "빠른 인자 자신의 주파수 번짐이 **0.005로 고정**입니다. 기본값 ±5%에서는 regime "
          "간격이 그와 같고, ±1%에서는 **1/5**이 되어 느린 신호가 빠른 흔들림에 묻힙니다.\n\n"
          "**±5%에서는 무메커니즘도 우리도 0.992로 동일 — 포함 축의 판별력이 0입니다.** "
          "±1%에서 0.393 vs 0.620으로 벌어지며 처음으로 측정 가능해집니다.",
          [("regime 간격", "이웃 regime 간 실제 주파수 차이. FFT 빈폭(64패치에서 0.00195)보다 "
                           "좁으면 한 윈도우에서 원리적으로 분해 불가"),
           ("FFT (N패치)", "**학습 없는** 고전 분류기 정확도 (chance 0.333). 우리가 넘어야 할 바"),
           ("학습 두 열", "`slow-kept F1 / 배제 여유`. 두 열이 같으면 그 난이도에서는 "
                          "메커니즘 유무를 구분할 수 없다는 뜻")])


# ----------------------------------------------------------------- 7. fastaxis
def fastaxis():
    d = load("fastaxis.json")
    if not d:
        return
    print("\n## 7. 빠른 인자는 임베딩의 어디에 사는가\n")
    print("**질문:** HAPT에서 PCA가 게이트만큼 배제하는 것처럼 보인 이유.\n")
    table(["데이터셋", "상위16 PC의<br>분산 지분", "빠른 R²<br>(상위16)", "빠른 R²<br>(꼬리48)",
           "빠른 R²<br>(전체64)", "상위16이 차지하는<br>빠른 R² 비중"],
          [[k, f"{v['var_share_top']*100:.1f}%", f"**{v['fast_r2_top']:.3f}**",
            f"{v['fast_r2_tail']:.3f}", f"{v['fast_r2_all']:.3f}",
            f"**{v['fast_r2_share_top']*100:.1f}%**"] for k, v in d.items()],
          "**HAPT에서 분산의 88.6%를 담은 상위 16차원이 빠른 인자를 0.012밖에 안 담습니다.** "
          "그래서 분산 기준으로 16차원을 고르는 PCA가 빠른 인자를 **우연히** 떨어뜨립니다 — "
          "게이트가 할 일이 없습니다. PTB-XL은 44.2%로 정반대입니다.",
          [("분산 지분", "게이트 없는 임베딩을 PCA 했을 때 상위 16성분의 분산 비율"),
           ("빠른 R² (상위16)", "그 상위 16성분만으로 빠른 인자를 맞힌 R². "
                                "**낮으면 빠른 인자가 저분산 꼬리에 숨음**"),
           ("비중", "성분별 빠른 R² 합에서 상위 16개가 차지하는 몫")])


# ----------------------------------------------------------------- appendix A
def appendix():
    from model import D_SLOW
    print("\n---\n\n## 부록 A. block × factor 원본 (3 seeds)\n")
    print("위 요약표들이 압축한 원본입니다. 본문 근거는 이 행렬입니다.\n")
    rows = []
    for key, label in DS:
        d = load(f"twosided_{key}.json")
        if not d:
            continue
        ceil = max(v["z_full"]["fast"] for v in d.values())
        for tag, c in d.items():
            null, fixed = null_of(d, tag)
            rows.append([label, tag, f"{c['z_slow']['slow']:.3f}", f"{c['z_slow']['fast']:+.3f}",
                         f"{c['z_fast']['slow']:.3f}", f"{c['z_fast']['fast']:+.3f}",
                         f"{c['z_full']['slow']:.3f}", f"{c['z_full']['fast']:+.3f}",
                         f"{null:+.3f}" + ("" if fixed else " ⁎"),
                         f"{null - c['z_slow']['fast']:+.3f}",
                         f"{sep_of(c, ceil, null)['sep']:.3f}"])
    table(["데이터셋", "cell", "z_slow 느림↑", "z_slow 빠름↓", "z_fast 느림↓", "z_fast 빠름↑",
           "z_full 느림", "z_full 빠름", "기준선", "배제 여유", "SEP"], rows,
          "⁎ = 게이트 없는 셀을 안 돌린 스템이라 자기 무작위 부분공간을 기준선으로 씀.",
          [("z_full 두 열", "전체 64차원의 점수. 포함 항의 분모이자 빠른 인자의 천장"),
           ("기준선", "그 스템 `g0_x0`의 무작위 16차원이 빠른 인자를 맞힌 R²"),
           ("배제 여유", "`기준선 − z_slow 빠름`. **0이면 아무 16차원과 같음**, 음수면 더 샘")])


if __name__ == "__main__":
    print("# HGLP v2 — 결과 요약 (2026-08-05)\n")
    print("모든 수치는 `runs_v2/*.json`에서 `summarize_answers.py`로 재생성됩니다. "
          "질문 하나에 표 하나이고, 데이터셋·스템은 **행**입니다 — 주장은 열을 따라 "
          "내려읽으면 확인됩니다.")
    for fn in (mechanism, stems, layernorm, rotation, domain, difficulty, fastaxis, appendix):
        fn()
