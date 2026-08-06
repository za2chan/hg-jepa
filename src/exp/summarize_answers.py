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
        print("\n⁎ = no ungated counterpart was run for this stem; its own random "
              "subspace is used, which is the less conservative choice.")


def blocknorm_table():
    from model import D_SLOW
    any_ = False
    for key, label in DS:
        d = load(f"twosided_bn_{key}.json")
        if not d:
            continue
        any_ = True
        print(f"\n### {label} — per-block LayerNorm은 그 자체로 메커니즘인가 (3 seeds)\n")
        print(f"| cell | z_slow fast↓ | 기준선(ungated) | 배제 여유 | (자기 rand{D_SLOW}) "
              "| z_fast fast↑ |")
        print("|---|---|---|---|---|---|")
        for tag, c in d.items():
            # each BN setting has its own ungated reference: g0_x0 with the same BN flag
            stem = tag.split("/")[0]
            suffix = "_noBN" if tag.endswith("_noBN") else ""
            ref = d.get(f"{stem}/g0_x0{suffix}")
            null = ref[f"rand{D_SLOW}"]["fast"] if ref else c[f"rand{D_SLOW}"]["fast"]
            print(f"| {tag} | {c['z_slow']['fast']:+.3f} | {null:+.3f} "
                  f"| {null - c['z_slow']['fast']:+.3f} "
                  f"| {c[f'rand{D_SLOW}']['fast']:+.3f} | {c['z_fast']['fast']:+.3f} |")
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
            print("| 방법 | 느린절반: 느림↑ | 느린절반: 빠름↓ | 기준선 대비 배제 "
                  "| 여집합: 빠름↑ |")
            print("|---|---|---|---|---|")
            for m, r in d.items():
                a, b = r["slow_half"], r["fast_half"]
                print(f"| {m} | {a['slow'][0]:.3f} | {a['fast'][0]:+.3f} "
                      f"| {null - a['fast'][0]:+.3f} | {b['fast'][0]:.3f} |")


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
    if "train" in d:
        from model import D_SLOW
        print("\n### 난이도별 block × factor (nce+ema, seed 0)\n")
        print(f"| gap | cell | z_slow slow↑ | z_slow fast↓ | rand{D_SLOW} fast | 배제 여유 |")
        print("|---|---|---|---|---|---|")
        for gap, cells in d["train"].items():
            for tag, c in cells.items():
                null = c[f"rand{D_SLOW}"][1]
                print(f"| ±{float(gap)*100:g}% | {tag} | {c['z_slow'][0]:.3f} "
                      f"| {c['z_slow'][1]:+.3f} | {null:+.3f} | {null - c['z_slow'][1]:+.3f} |")


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


if __name__ == "__main__":
    print("# 2026-08-05 질문 라운드 — 집계된 결과")
    for fn in (block_factor_tables, blocknorm_table, rotation_tables,
               domain_tables, difficulty_table, fastaxis_table):
        fn()
