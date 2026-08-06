"""Collect every run behind the 2026-08-05 question round into report-ready tables.

Reads whatever exists in runs_v2/ and skips what does not, so it can be run while
the chain is still going. Prints markdown, because that is what the report needs.

Usage: python3 summarize_answers.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

R = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
DS = [("synth", "Synthetic"), ("ptbxl", "PTB-XL"), ("hapt", "HAPT")]


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def sep_of(cell, ceiling, null=None):
    from probes import sep_index
    return sep_index({k: (v["slow"], v["fast"]) for k, v in cell.items()}, ceiling, null)


def ungated_null(d, tag):
    """The g0_x0 cell of the same stem: one fixed yardstick for the comparison.

    reg+ema and nce+online were only run as the full-mechanism cell, so they have
    no ungated counterpart; they fall back to their own random subspace. That is
    the less conservative form, so those two rows are marked in the table."""
    from model import D_SLOW
    stem = tag.split("/")[0]
    c = d.get(f"{stem}/g0_x0")
    return (c or d[tag])[f"rand{D_SLOW}"]["fast"], c is not None


def legend(*rows):
    """Print a one-line meaning for every column, directly under its table."""
    print("\n<details><summary>열 설명</summary>\n")
    print("| 열 | 뜻 |")
    print("|---|---|")
    for name, meaning in rows:
        print(f"| `{name}` | {meaning} |")
    print("\n</details>")


BF_LEGEND = (
    ("cell", "스템 / 메커니즘 조합. `g1_x1` = 게이트·xcov 둘 다 켬 = **우리 방법**, "
             "`g0_x0` = 둘 다 끔"),
    ("z_slow 느림↑", "앞 16차원에서 **느린 인자**(합성=regime, HAPT=활동, PTB-XL=진단)를 "
                     "선형 프로브로 맞힌 macro-F1. **높을수록 좋음**"),
    ("z_slow 빠름↓", "앞 16차원에서 **빠른 인자**(합성=u, HAPT=‖acc‖, PTB-XL=ECG 전압)를 "
                     "맞힌 R². 이게 leak. **낮을수록 좋음**"),
    ("z_fast 느림↓", "뒤 48차원에서 느린 인자 macro-F1. 분리가 됐다면 낮아야 하지만, "
                     "설계상 완전히 0일 필요는 없음"),
    ("z_fast 빠름↑", "뒤 48차원에서 빠른 인자 R². **높아야 함** — 낮으면 빠른 인자를 "
                     "배제한 게 아니라 **파괴**한 것"),
    ("기준선(ungated)", "게이트 없는 모델의 **무작위 16차원**이 빠른 인자를 맞힌 R². "
                        "'아무 16차원이나 뽑으면 이 정도'라는 비교 잣대"),
    ("배제 여유", "`기준선 − z_slow 빠름`. **0이면 아무 16차원과 다를 바 없음**(메커니즘이 "
                  "한 일 없음), 클수록 진짜 배제, 음수면 오히려 더 샘"),
    ("(자기 rand16)", "그 셀 **자신의** 임베딩에서 뽑은 무작위 16차원 값. 참고용"),
    ("SEP", "포함×배정×배제, [0,1]. 행렬을 한 줄로 줄인 **부록 정렬용** 보조값"),
)


def block_factor_tables():
    from model import D_SLOW
    for key, label in DS:
        d = load(f"twosided_{key}.json")
        if not d:
            print(f"\n### {label} — block x factor: NOT YET AVAILABLE\n"); continue
        ceiling = max(v["z_full"]["fast"] for v in d.values())
        print(f"\n### {label} — block × factor (3 seeds; fast-factor ceiling {ceiling:.3f})\n")
        # The null MUST be the same one sep_index uses, or the margin column and the
        # SEP column in the same row disagree (they did, until 2026-08-06).
        print("| cell | z_slow slow↑ | z_slow fast↓ | z_fast slow↓ | z_fast fast↑ "
              f"| 기준선(ungated) | 배제 여유 | (자기 rand{D_SLOW}) | SEP |")
        print("|---|---|---|---|---|---|---|---|---|")
        for tag, c in d.items():
            null, fixed = ungated_null(d, tag)
            s = sep_of(c, ceiling, null)
            print(f"| {tag} | {c['z_slow']['slow']:.3f} | {c['z_slow']['fast']:+.3f} "
                  f"| {c['z_fast']['slow']:.3f} | {c['z_fast']['fast']:+.3f} "
                  f"| {null:+.3f}{'' if fixed else ' ⁎'} "
                  f"| {null - c['z_slow']['fast']:+.3f} "
                  f"| {c[f'rand{D_SLOW}']['fast']:+.3f} | **{s['sep']:.3f}** |")
        print("\n⁎ = 그 스템은 게이트 없는 셀을 안 돌려서 자기 무작위 부분공간을 "
              "기준선으로 씀(덜 보수적).")
        legend(*BF_LEGEND)


BN_HDR = ("| cell | z_slow 느림↑ | z_slow 빠름↓ | z_fast 빠름↑ | 기준선 | 배제 여유 | SEP |\n"
          "|---|---|---|---|---|---|---|")


def _bn_row(d, tag, c, ref_suffix, ceiling):
    """One LayerNorm-ablation row. `ref_suffix` picks the g0_x0 the null comes from:
    None = the cell's own BN condition, "_noBN" = always the LN-free control."""
    from model import D_SLOW
    suf = ("_noBN" if tag.endswith("_noBN") else "") if ref_suffix is None else ref_suffix
    ref = d.get(f"{tag.split('/')[0]}/g0_x0{suf}")
    null = (ref or c)[f"rand{D_SLOW}"]["fast"]
    s = sep_of(c, ceiling, null)
    return (f"| {tag} | {c['z_slow']['slow']:.3f} | {c['z_slow']['fast']:+.3f} "
            f"| {c['z_fast']['fast']:+.3f} | {null:+.3f} "
            f"| {null - c['z_slow']['fast']:+.3f} | **{s['sep']:.3f}** |")


def blocknorm_table():
    any_ = False
    for key, label in DS:
        d = load(f"twosided_bn_{key}.json")
        if not d:
            continue
        any_ = True
        ceiling = max(v["z_full"]["fast"] for v in d.values())
        # (A) the headline: the model we actually ship (gate + xcov + per-block LN)
        #     against the ONLY genuinely mechanism-free control, g0_x0 with LN off.
        #     Tables (B) and (C) decompose this; neither of them states it.
        print(f"\n### {label} — 우리 모델 vs 진짜 무메커니즘 대조군 (3 seeds) ★\n")
        print("배포하는 모델(`g1_x1`, per-block LN 포함)을 **메커니즘이 하나도 없는** "
              "`g0_x0_noBN`의 무작위 16차원과 비교합니다. 논문의 \"우리 방법이 분리한다\"는 "
              "이 표로 주장합니다. 아래 (1)(2)는 그 성과를 부품별로 분해한 것입니다.\n")
        print(BN_HDR)
        for tag, c in d.items():
            if tag.endswith("/g1_x1"):
                print(_bn_row(d, tag, c, "_noBN", ceiling))
        legend(("cell", "`g1_x1` = 게이트·xcov 켬 + per-block LN 켬 = **배포 모델**"),
               *[r for r in BF_LEGEND if r[0] in
                 ("z_slow 느림↑", "z_slow 빠름↓", "z_fast 빠름↑", "배제 여유", "SEP")],
               ("기준선", "**`g0_x0_noBN`**의 무작위 16차원 — 게이트도 xcov도 per-block LN도 "
                          "없는, 유일하게 완전한 무메커니즘 기준"))

        # (1) each condition against its OWN reference — what the gate adds on top of
        #     whatever LayerNorm already did in that condition.
        print(f"\n### {label} — per-block LayerNorm은 그 자체로 메커니즘인가 (3 seeds)\n")
        print("각 셀을 **같은 BN 설정의 g0_x0**과 비교합니다.\n")
        print(BN_HDR)
        for tag, c in d.items():
            print(_bn_row(d, tag, c, None, ceiling))
        legend(("cell", "`_noBN` 접미사 = per-block LayerNorm을 끈 셀 (D_Z 전체에 "
                        "LayerNorm 하나만). 접미사 없으면 켜진 셀"),
               *[r for r in BF_LEGEND if r[0] in
                 ("z_slow 느림↑", "z_slow 빠름↓", "z_fast 빠름↑", "배제 여유", "SEP")],
               ("기준선", "**같은 BN 설정의** `g0_x0`에서 뽑은 무작위 16차원 값. "
                          "BN 켠 셀은 BN 켠 기준선, 끈 셀은 끈 기준선과 비교"))

        # (2) the LN-free world only: no row here has per-block LN, so no claim read
        #     off this table can be an artefact of it.
        print(f"\n### {label} — LayerNorm 효과를 뺀 표 (BN 없는 셀만)\n")
        print("`blocknorm=False`는 D_Z 전체에 LayerNorm 하나만 걸어 모든 좌표를 동등하게 "
              "대합니다. 이 표는 그 조건의 셀만 담고 기준선도 BN 없는 g0_x0이라, "
              "**LN 교란이 원리적으로 없습니다.**\n")
        print(BN_HDR)
        for tag, c in d.items():
            if tag.endswith("_noBN"):
                print(_bn_row(d, tag, c, "_noBN", ceiling))
        legend(("cell", "전부 `_noBN` — per-block LayerNorm이 없는 셀만 모은 표"),
               *[r for r in BF_LEGEND if r[0] in
                 ("z_slow 느림↑", "z_slow 빠름↓", "z_fast 빠름↑", "배제 여유", "SEP")],
               ("기준선", "BN 없는 `g0_x0`의 무작위 16차원. **표 전체가 LN 없는 조건**"))
    if not any_:
        print("\n### per-block LayerNorm ablation: NOT YET AVAILABLE\n")


def rotation_tables():
    for stem in ("nce+ema", "l1+ema"):
        for key, label in DS:
            d = load(f"rotation_{key}_{stem}.json")
            if not d or "random" not in d:          # pre-two-sided file, being rewritten
                print(f"\n### {label} / {stem} — post-hoc rotation: NOT YET AVAILABLE\n")
                continue
            null = d["random"]["slow_half"]["fast"][0]
            print(f"\n### {label} / {stem} — 사후 회전, 양방향 (3 seeds; "
                  f"무작위 16차원 기준선 {null:+.3f})\n")
            s_ceil = max(r["slow_half"]["slow"][0] for r in d.values())
            f_ceil = max(r["fast_half"]["fast"][0] for r in d.values())
            print("| 방법 | 느린절반: 느림↑ | 느린절반: 빠름↓ | 기준선 대비 배제 "
                  "| 여집합: 빠름↑ | SEP* |")
            print("|---|---|---|---|---|---|")
            for m, r in d.items():
                a, b = r["slow_half"], r["fast_half"]
                cl = lambda v: min(max(v, 0.0), 1.0)
                sep = (cl(a["slow"][0] / s_ceil) * cl(b["fast"][0] / f_ceil)
                       * cl(1 - a["fast"][0] / null if null > 1e-3 else 0.0))
                print(f"| {m} | {a['slow'][0]:.3f} | {a['fast'][0]:+.3f} "
                      f"| {null - a['fast'][0]:+.3f} | {b['fast'][0]:.3f} "
                      f"| **{sep:.3f}** |")
            print("\nSEP* — 사후 회전에는 z_full 행이 없어서 포함·배정의 분모를 "
                  "**이 표 안의 최댓값**으로 씁니다. block × factor 표의 SEP과 분모가 "
                  "다르므로 표를 가로질러 비교하지 마세요.")
            legend(("방법", "16차원 '느린 부분공간'을 고르는 방식. `gate` = 우리 모델의 "
                            "z_slow, 나머지는 **게이트 없는 모델**의 임베딩을 라벨 없이 "
                            "사후 분해한 것. `random` = 무작위 회전(기준선)"),
                   ("느린절반: 느림↑", "그 방식이 고른 16차원에서 느린 인자 macro-F1"),
                   ("느린절반: 빠름↓", "같은 16차원에서 빠른 인자 R² (leak)"),
                   ("기준선 대비 배제", "`random의 빠름 − 이 방법의 빠름`. 0 이하면 "
                                        "**아무것도 배제하지 못한 것**"),
                   ("여집합: 빠름↑", "나머지 48차원에서 빠른 인자 R². 낮으면 빠른 인자를 "
                                     "옮긴 게 아니라 **버린 것**"),
                   ("SEP*", "이 표 안의 최댓값으로 정규화한 요약값. 표 사이 비교 불가"))


def domain_tables():
    for mode in ("block", "time"):
        for stem in ("nce+ema", "l1+ema"):
            d = load(f"domain_shift_hapt_{stem}_{mode}.json")
            a = (d or {}).get("agg", {})
            if not a:
                print(f"\n### HAPT / {stem} / {mode} — NOT AVAILABLE\n"); continue
            tv = d.get("label_tv")
            print(f"\n### HAPT / {stem} / {mode} 분할 — 도메인 이동 (3 파티션 × 3 시드 = 9런"
                  + (f"; 라벨 TV {tv:.3f})\n" if tv is not None else ")\n"))
            has_m = "delta_matched" in a.get("z_slow", {})
            print("| 블록 | in F1 | ood F1 | Δ | Δ_late | Δ_fast |"
                  + (" **Δ_matched** |" if has_m else ""))
            print("|---|---|---|---|---|---|" + ("---|" if has_m else ""))
            for b in [k for k in a if not k.startswith("claim_")]:
                v = a[b]
                row = (f"| {b} | {v['in_f1'][0]:.3f} | {v['ood_f1'][0]:.3f} "
                       f"| {v['delta'][0]:+.3f}±{v['delta'][1]:.3f} "
                       f"| {v['delta_late'][0]:+.3f}±{v['delta_late'][1]:.3f} "
                       f"| {v['delta_fast'][0]:+.3f} |")
                if has_m:
                    m = v.get("delta_matched", [float("nan")] * 2)
                    row += f" {m[0]:+.3f}±{m[1]:.3f} |"
                print(row)
            print()
            for k in sorted(a):
                if k.startswith("claim_"):
                    c = a[k]
                    print(f"- `{k}`: {c['ref_minus_slow'][0]:+.3f}±{c['ref_minus_slow'][1]:.3f}, "
                          f"**{c['n_supporting']}/{c['n_runs']}** 런")
            legend(("블록", "채점한 표현. `ungated_` 접두사 = **게이트 없는 대조군 모델**의 "
                            "같은 블록"),
                   ("in F1", "A_test(학습에 쓴 24명의 안 본 윈도우)에서의 활동 macro-F1"),
                   ("ood F1", "B(완전히 안 본 6명)에서의 macro-F1"),
                   ("Δ", "`in − ood`. 도메인 이동으로 잃은 양. **작을수록 강건**"),
                   ("Δ_late", "B를 후반 30%로 한정해 시간 위치를 맞춘 Δ"),
                   ("Δ_fast", "같은 이동에서 **빠른 인자** R²가 떨어진 양. Δ만큼 크면 "
                              "두 구조가 함께 이동한 것"),
                   ("Δ_matched", "A_test를 B와 **같은 피험자 수·윈도우 수**로 맞춘 Δ. "
                                 "평가셋 크기 효과를 제거"),
                   ("claim_…", "`Δ_기준 − Δ_slow`. 양수면 z_slow가 덜 잃은 것. "
                               "9런 중 몇 번 성립했는지 병기"))


def difficulty_table():
    d = load("difficulty.json")
    if not d:
        print("\n### 난이도: NOT YET AVAILABLE\n"); return
    if "fft" in d:
        print("\n### 학습 없는 FFT 베이스라인 (chance 0.333)\n")
        print("| gap | 문맥 | FFT 빈폭 | regime 간격 | 빠른인자 번짐 | centroid | peak |")
        print("|---|---|---|---|---|---|---|")
        for k, v in d["fft"].items():
            gap, c = k.replace("gap", "").split("_c")
            print(f"| ±{float(gap)*100:g}% | {c} | {v['bin_width']:.5f} "
                  f"| {v['regime_spacing']:.5f} | {v['smear']:.5f} "
                  f"| **{v['centroid']:.3f}** | {v['peak']:.3f} |")
        legend(("gap", "regime 간 반송파 주파수 간격 (±%)"),
               ("문맥", "관측에 쓴 패치 수 (1패치 = 8샘플)"),
               ("FFT 빈폭", "그 문맥 길이에서 FFT가 구분할 수 있는 최소 주파수 간격. "
                            "regime 간격이 이보다 좁으면 **원리적으로 분해 불가**"),
               ("regime 간격", "이웃 regime 사이의 실제 주파수 차이"),
               ("빠른인자 번짐", "빠른 인자 u가 순간 주파수를 흔드는 폭. regime 간격이 "
                                 "이보다 작으면 **느린 신호가 빠른 흔들림에 묻힘**"),
               ("centroid / peak", "학습 없는 고전 분류기 두 종류의 정확도 (chance 0.333). "
                                   "우리가 넘어야 할 바"))
    if "train" in d:
        from model import D_SLOW
        print("\n### 난이도별 block × factor (nce+ema, seed 0)\n")
        from probes import sep_index
        print(f"| gap | cell | z_slow 느림↑ | z_slow 빠름↓ | z_fast 빠름↑ | 기준선 "
              "| 배제 여유 | SEP |")
        print("|---|---|---|---|---|---|---|---|")
        for gap, cells in d["train"].items():
            ceil = max(c["z_full"][1] for c in cells.values())
            base = cells.get("g0_x0", {}).get(f"rand{D_SLOW}")
            for tag, c in cells.items():
                null = base[1] if base else c[f"rand{D_SLOW}"][1]
                sp = sep_index({k: tuple(v) for k, v in c.items()}, ceil, null)
                print(f"| ±{float(gap)*100:g}% | {tag} | {c['z_slow'][0]:.3f} "
                      f"| {c['z_slow'][1]:+.3f} | {c['z_fast'][1]:+.3f} | {null:+.3f} "
                      f"| {null - c['z_slow'][1]:+.3f} | **{sp['sep']:.3f}** |")
        legend(("gap", "regime 간 반송파 주파수 간격. ±5%가 기본, ±1%가 어려운 설정"),
               *[r for r in BF_LEGEND if r[0] in
                 ("cell", "z_slow 느림↑", "z_slow 빠름↓", "z_fast 빠름↑", "배제 여유", "SEP")],
               ("기준선", "그 난이도의 `g0_x0`에서 뽑은 무작위 16차원 값"))


def fastaxis_table():
    d = load("fastaxis.json")
    if not d:
        print("\n### 빠른 인자의 위치: NOT YET AVAILABLE\n"); return
    from model import D_SLOW
    print(f"\n### 빠른 인자가 저분산 방향에 있는가 (게이트 없는 임베딩)\n")
    print(f"| 데이터셋 | 상위{D_SLOW} PC의 분산 지분 | 빠른 R² (상위{D_SLOW}) "
          f"| 빠른 R² (꼬리 48) | 빠른 R² (전체 64) |")
    print("|---|---|---|---|---|")
    for k, v in d.items():
        print(f"| {k} | {v['var_share_top']*100:.1f}% | {v['fast_r2_top']:.3f} "
              f"| {v['fast_r2_tail']:.3f} | {v['fast_r2_all']:.3f} |")
    legend(("상위16 PC의 분산 지분", "게이트 없는 임베딩을 PCA 했을 때 상위 16성분이 "
                                     "차지하는 분산 비율"),
           ("빠른 R² (상위16)", "그 상위 16성분만으로 빠른 인자를 맞힌 R². "
                                "**낮으면 빠른 인자가 저분산 꼬리에 숨어 있다는 뜻**"),
           ("빠른 R² (꼬리 48)", "나머지 48성분으로 맞힌 R²"),
           ("빠른 R² (전체 64)", "전체 임베딩으로 맞힌 R² (천장)"))


if __name__ == "__main__":
    print("# 2026-08-05 질문 라운드 — 집계된 결과")
    for fn in (block_factor_tables, blocknorm_table, rotation_tables,
               domain_tables, difficulty_table, fastaxis_table):
        fn()
