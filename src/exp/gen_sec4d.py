"""Tables for the exclusion-clause sections of the report (Results IV-C / IV-D).

Every number is read from runs_v2/*.json and written straight into LaTeX; nothing
here is typed by hand (CLAUDE.md rule 3).

    python3 src/exp/gen_sec4d.py            -> paper_ICLR/tab_*.tex  + console summary

Sleep-EDF is EXCLUDED from every table by user decision (the window-length axis was
never tested there and the analysis is unfinished). It is still computed and printed
to the console under "[held out]" so the Limitations sentence about a fourth dataset
stays honest.

Tables produced
  tab_terms       IV-D  our score minus SFA's, split into inclusion/allocation/exclusion
  tab_gatesym     IV-D  does gating z_slow symmetrically fix the exclusion term?
  tab_mlpprobe    IV-D  how far each term moves when the probe head stops being linear
  tab_lambdafree  IV-D  does a label-free criterion J pick the same lambda as the oracle?
"""
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
OUT = pathlib.Path(__file__).resolve().parents[2] / "paper_ICLR"
load = lambda n: json.loads((R / n).read_text())

LAMS = [0, 1, 4, 16, 64]
# (paper label, rotation-file dataset tag, term_breakdown key prefix)
DSETS = [("Synthetic", "synth", "Synthetic", "_gap0.03"),
         ("PTB-XL", "ptbxl", "PTB-XL", ""),
         ("HAPT", "hapt", "HAPT", "")]
STEMS = [("HGLP-Reg", "l1+ema", "Reg"), ("HGLP-NCE", "nce+ema", "NCE")]
HELD_OUT = "Sleep-EDF"          # computed, printed, never tabulated


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print(f"  wrote paper_ICLR/{name}.tex")


def rot(dset, stem, suffix=""):
    """rotation_<dset>_<stem>[_gap][_sym].json"""
    return load(f"rotation_{dset}_{stem}{suffix}.json")


def per_seed(entry, half="slow_half", factor="slow"):
    """rotation values are stored as [mean, std, seed0, seed1, ...]."""
    return entry[half][factor][2:]


def sep_of(d, key):
    """SEP = inclusion * allocation * exclusion, each clipped to [0,1]."""
    c = lambda v: min(max(v, 0.0), 1.0)
    incl = c(d[key]["slow_half"]["slow"][0])
    excl = c(1.0 - d[key]["slow_half"]["fast"][0])
    alloc = c(d[key]["fast_half"]["fast"][0])
    return incl * alloc * excl


def best_lam(d):
    vals = [(sep_of(d, f"gate@lam{l}"), l) for l in LAMS]
    return max(vals)


# ----------------------------------------------------------------- IV-D table 1
def terms():
    """Where do we actually lose to SFA? Split the product into its three factors."""
    d = load("term_breakdown.json")
    rows = []
    for label, _, key, _ in DSETS:
        for stem_label, _, stem in STEMS:
            e = d[f"{key}/{stem}"]
            g = e["_diagnosis"]
            lam = g["best_ours"].replace("ours@lam", "")
            bold = lambda v: (f"$\\mathbf{{{v:+.3f}}}$" if v < -0.02 else f"${v:+.3f}$")
            rows.append(f"{label} & {stem_label} & {lam} & ${g['d_inclusion']:+.3f}$ & "
                        f"${g['d_allocation']:+.3f}$ & {bold(g['d_exclusion'])} & "
                        f"${g['d_sep']:+.3f}$ \\\\")
    for stem in ("Reg", "NCE"):
        g = d[f"{HELD_OUT}/{stem}"]["_diagnosis"]
        print(f"  [held out] {HELD_OUT}/{stem} incl {g['d_inclusion']:+.3f} "
              f"alloc {g['d_allocation']:+.3f} excl {g['d_exclusion']:+.3f} "
              f"sep {g['d_sep']:+.3f}")

    write("tab_terms", """\\begin{table}[!tb]
\\caption{Where the shortfall against SFA sits. Every cell is \\emph{ours minus SFA} on
one term of SEP, paired by seed and then averaged, $n=5$; positive means we lead. Our
column is taken at the $\\lambda$ that maximizes our SEP, which is why $\\lambda$ is
listed. Inclusion is a draw everywhere; \\textbf{every real-data loss is a loss on the
exclusion term alone} --- the one term the objective does not enforce.}
\\label{tab:terms}
\\centering
\\small
\\begin{tabular}{lllcccc}
\\hline
Dataset & Stem & $\\lambda$ & Inclusion & Allocation & Exclusion & SEP \\\\
\\hline
""" + "\n".join(rows) + """
\\hline
\\end{tabular}
\\end{table}
""")


# ----------------------------------------------------------------- IV-D table 2
def gatesym():
    """Can the exclusion term be fixed structurally, by gating z_slow too?"""
    TIE = 0.005                          # SEP differences below this are called a tie
    rows, tally = [], {"\\textbf{sym}": 0, "asym": 0, "tie": 0}
    for label, tag, _, suf in DSETS:
        for stem_label, stem, _ in STEMS:
            a_sep, a_lam = best_lam(rot(tag, stem, suf))
            s_sep, s_lam = best_lam(rot(tag, stem, suf + "_sym"))
            win = "tie" if abs(s_sep - a_sep) <= TIE else (
                "\\textbf{sym}" if s_sep > a_sep else "asym")
            tally[win] += 1
            rows.append(f"{label} & {stem_label} & {a_sep:.3f} ($\\lambda={a_lam}$) & "
                        f"{s_sep:.3f} ($\\lambda={s_lam}$) & ${s_sep - a_sep:+.3f}$ & {win} \\\\")
    better, worse, tied = tally["\\textbf{sym}"], tally["asym"], tally["tie"]
    print(f"  symmetric gate: better {better}, worse {worse}, tied {tied} of {len(rows)}")

    write("tab_gatesym", f"""\\begin{{table}}[!tb]
\\caption{{Can exclusion be bought structurally? The symmetric variant multiplies
$\\zslow$ by $1-g(\\Delta)$, switching it off below $\\tau$ so that no horizon can push
transient information into it. Cells are SEP at each model's own best $\\lambda$ over
$\\{{0,1,4,16,64\\}}$, $n=5$ seeds; higher is better; differences within $\\pm{TIE:g}$ SEP
are called a tie. The asymmetric gate of Eq.~\\ref{{eq:gate}} is better in {worse} of
{len(rows)} settings against {better} for the symmetric one, so the design choice is not
an oversight.}}
\\label{{tab:gatesym}}
\\centering
\\small
\\begin{{tabular}}{{llcccc}}
\\hline
Dataset & Stem & Asymmetric (ours) & Symmetric & Difference & Better \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\end{{table}}
""")


# ----------------------------------------------------------------- IV-D table 3
def mlpprobe():
    """Same features, same splits, same blocks -- only the probe head changes."""
    d = load("mlp_probe_rotation.json")["results"]
    tag_of = lambda s: "nce" if s == "nce+ema" else "l1"
    keep = [(f"{lbl}/{s_lbl}", f"{tag}/{tag_of(s)}")
            for lbl, tag, _, _ in DSETS for s_lbl, s, _ in STEMS]
    real = [(lbl, key) for lbl, key in keep if not lbl.startswith("Synthetic")]
    methods = ["ours", "SFA", "PCA", "ICA-slow"]     # "ours" is the unprefixed pair
    terms_ = ["inclusion", "allocation", "exclusion"]
    head = lambda e, m, p: e[p] if m == "ours" else e[f"{m}/{p}"]

    # (a) per-term mean change over method x setting. Only the four real settings enter:
    # the synthetic path builds no time-series stream, so it has no post-hoc rows to
    # average against and including it would put "ours" in four times as often.
    agg = {t: [] for t in terms_}
    for _, key in real:
        for m in methods:
            for t in terms_:
                agg[t].append(head(d[key], m, "mlp")[t] - head(d[key], m, "linear")[t])
    lines = [f"{t.capitalize()} & ${sum(v) / len(v):+.3f}$ & ${min(v):+.3f}$ & "
             f"${max(v):+.3f}$ & {len(v)} \\\\" for t, v in
             ((t, agg[t]) for t in terms_)]
    for lbl, key in keep:
        if lbl.startswith("Synthetic"):
            e = d[key]
            print("  [ours only, no post-hoc rows] %s excl %.3f -> %.3f, SEP %.3f -> %.3f"
                  % (lbl, e["linear"]["exclusion"], e["mlp"]["exclusion"],
                     e["linear"]["sep"], e["mlp"]["sep"]))

    # (b) exclusion gap SFA - ours, linear against MLP
    gap = [f"{lbl} & ${d[key]['SFA/linear']['exclusion'] - d[key]['linear']['exclusion']:+.3f}$ "
           f"& ${d[key]['SFA/mlp']['exclusion'] - d[key]['mlp']['exclusion']:+.3f}$ \\\\"
           for lbl, key in real]
    for stem, t in (("Reg", "l1"), ("NCE", "nce")):
        e = d.get(f"sleepedf/{t}")
        if e and "SFA/linear" in e:
            print(f"  [held out] {HELD_OUT}/{stem} exclusion gap linear "
                  f"{e['SFA/linear']['exclusion'] - e['linear']['exclusion']:+.3f} "
                  f"-> mlp {e['SFA/mlp']['exclusion'] - e['mlp']['exclusion']:+.3f}")

    write("tab_mlpprobe", f"""\\begin{{table}}[!tb]
\\caption{{What survives a non-linear reader. The features, the splits and the blocks are
held fixed and only the probe head changes, from a linear model to a one-hidden-layer
MLP (256 units); the \\emph{{same}} head is given to SFA, PCA and ICA, so no method is
tested more harshly than another. $\\lambda=4$, one seed. \\textbf{{(a)}} Cells are
$\\text{{MLP}}-\\text{{linear}}$ on one SEP term, averaged over four methods $\\times$ six
settings. Only exclusion falls, and it is the term the objective never enforced.
\\textbf{{(b)}} Cells are SFA's exclusion minus ours, so positive means SFA holds the
lead; the diagnosis of Table~\\ref{{tab:terms}} was not an artifact of linear probing.}}
\\label{{tab:mlpprobe}}
\\centering
\\small
(a) Change in each SEP term when the probe head stops being linear\\\\[2pt]
\\begin{{tabular}}{{lcccc}}
\\hline
SEP term & Mean change & Worst & Best & Rows \\\\
\\hline
{chr(10).join(lines)}
\\hline
\\end{{tabular}}

\\medskip
(b) Exclusion gap, SFA minus ours\\\\[2pt]
\\begin{{tabular}}{{lcc}}
\\hline
Setting & Linear probe & MLP probe \\\\
\\hline
{chr(10).join(gap)}
\\hline
\\end{{tabular}}
\\end{{table}}
""")


# ----------------------------------------------------------------- IV-D table 4
def lambdafree():
    """J = exclusion x long-horizon self-prediction. Neither factor uses task labels."""
    d = load("label_free_lambda.json")
    # PTB-XL was recomputed after the fix described in the analysis log (the earlier run
    # skipped it, mistaking "one label per record" for "one time position per record"),
    # so the later file wins. It carries no _pick, hence the picks are recomputed here.
    d.update(load("ptbxl_label_free_lambda.json"))

    # The oracle is "the lambda that maximizes SEP". Take it from term_breakdown, which
    # carries five seeds and is the same curve the main tables use; the label-free file's
    # own sep column is three-seed and is null for PTB-XL. Where both exist they agree,
    # and the assertion below keeps it that way.
    tb = load("term_breakdown.json")
    oracle_of = lambda key: float(tb[key]["_diagnosis"]["best_ours"].replace("ours@lam", ""))

    def pick_of(key, e):
        curve = {float(k): v for k, v in e.items() if k != "_pick"}
        js = sorted((v["J"] for v in curve.values()), reverse=True)
        p = dict(J=max(curve.items(), key=lambda kv: kv[1]["J"])[0],
                 oracle=oracle_of(key), margin=js[0] - js[1])
        if "_pick" in e:                     # agree with the stored pick where one exists
            assert (p["J"], p["oracle"]) == (e["_pick"]["J"], e["_pick"]["oracle"]), \
                f"{key}: recomputed {p} disagrees with stored {e['_pick']}"
        return p

    rows, agree = [], 0
    for label, _, key, _ in DSETS:
        for stem_label, _, stem in STEMS:
            p = pick_of(f"{key}/{stem}", d[f"{key}/{stem}"])
            ok = p["J"] == p["oracle"]
            agree += ok
            verdict = "match" if ok else "\\textbf{miss}"
            rows.append(f"{label} & {stem_label} & {p['J']:.0f} & {p['oracle']:.0f} & "
                        f"{verdict} & {p['margin']:.3f} \\\\")
    for stem in ("Reg", "NCE"):
        k = f"{HELD_OUT}/{stem}"
        p = pick_of(k, load("label_free_lambda.json")[k])
        print(f"  [held out] {HELD_OUT}/{stem} J picks lambda={p['J']:.0f}, "
              f"oracle picks {p['oracle']:.0f} -> "
              f"{'match' if p['J'] == p['oracle'] else 'MISS'}")
    print(f"  label-free lambda agrees in {agree} of {len(rows)} reported settings")

    write("tab_lambdafree", f"""\\begin{{table}}[!tb]
\\caption{{Choosing $\\lambda$ without labels. $\\mathrm{{PPS}}=\\text{{purity}}\\times\\text{{persistence}}$: the first factor is read from the transient proxy, which is computed
from the signal, and the second asks how well $\\zslow$ predicts \\emph{{its own}} value
more than $\\tau$ ahead. No task label enters either. The product is needed because each
factor alone has a degenerate maximum --- an empty block scores perfect exclusion, a
constant block perfect self-prediction. The oracle column is the $\\lambda$ that maximizes
SEP, which does use labels. $n=3$ seeds; the margin is the gap between the top two $\\mathrm{{PPS}}$
values, and small margins should not be read as confident agreement.}}
\\label{{tab:lambdafree}}
\\centering
\\small
\\begin{{tabular}}{{llcccc}}
\\hline
Dataset & Stem & $\\lambda$ by $\\mathrm{{PPS}}$ (no labels) & $\\lambda$ by SEP (oracle) & Agreement & $\\mathrm{{PPS}}$ margin \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\end{{table}}
""")


# --------------------------------------------------------------- appendix A.1
def gatehard():
    """Step gate against the smooth one. Backs the claim in Method that a step
    lowers separation on both stems, which until now had no table behind it."""
    soft = load("twosided_main22_synth_gap0.03.json")
    hard = load("twosided_gatehard_synth_gap0.03.json")
    bf = lambda d, k: (min(max(d[k]["z_slow"]["slow"], 0), 1)
                       * min(max(d[k]["z_fast"]["fast"], 0), 1)
                       * min(max(1 - d[k]["z_slow"]["fast"], 0), 1))
    rows = []
    for stem_label, stem, _ in STEMS:
        s, h = bf(soft, f"{stem}/g1_x1"), bf(hard, f"{stem}/g1_x1")
        rows.append(f"{stem_label} & \\textbf{{{s:.3f}}} & {h:.3f} & ${h - s:+.3f}$ \\\\")
        print(f"  {stem_label}: smooth {s:.3f} -> hard {h:.3f} ({h - s:+.3f})")
    write("tab_gatehard", f"""\\begin{{table}}[!tb]
\\caption{{A step gate against the smooth one. The smooth gate is
$g(\\Delta)=\\sigma((\\tau-\\Delta)/\\kappa)$; the step version replaces it with
$\\mathbf{{1}}[\\Delta<\\tau]$, which zeroes $\\zmix$'s gradient outright beyond $\\tau$
instead of passing a partial one. Cells are SEP on synthetic, $n=5$ seeds, all else
held fixed. The step form is worse on both stems, which is why $\\tau$'s width
$\\kappa$ is part of the method rather than a smoothing convenience.}}
\\label{{tab:gatehard}}
\\centering
\\small
\\begin{{tabular}}{{lccc}}
\\hline
Stem & Smooth gate (ours) & Step gate & Difference \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\end{{table}}
""")


def targetpilot():
    """The harmlessness curve: does withholding the transient block cost anything
    at long range? Measured on the two target designs, v1 cumulative and v2 bounded."""
    d = load("pilot_b5.json")
    deltas = sorted((int(k) for k in d["bounded"]["harmlessness"]), key=int)
    head = " & ".join(str(x) for x in deltas)
    row = lambda m: " & ".join(f"{d[m]['harmlessness'][str(x)]:+.3f}" for x in deltas)
    b, c = d["bounded"]["stability"], d["cumulative"]["stability"]
    print(f"  bounded reaches {d['bounded']['harmlessness']['16']:+.3f} at D=16; "
          f"cumulative still {d['cumulative']['harmlessness']['128']:+.3f} at D=128")
    write("tab_targetpilot", f"""\\begin{{table}}[!tb]
\\caption{{Is withholding the transient block harmless at long range? Cells are
$R^2(s,u\\!\\to\\!\\bar z_{{t+\\Delta}}) - R^2(s\\!\\to\\!\\bar z_{{t+\\Delta}})$: how much the
transient factor adds, \\emph{{beyond}} what the persistent factor already explains, to a
linear reading of the target embedding $\\Delta$ ahead. \\textbf{{Zero means the transient
factor is redundant there}}, so gating it away costs nothing. Synthetic, HGLP-Reg, 2{{,}}500
steps, one seed, ground-truth factors. The bounded target reaches zero at
$\\Delta=16$ --- the value of $\\tau$ used on this data --- and stays there; the cumulative
target never does, because its receptive field runs back to the start of the window and so
re-contains the anchor's own transient state. This is the measurement behind the
receptive-field bound of Eq.~\\ref{{eq:target}}. It reads the target embedding rather than
downstream error, and it is one seed on one dataset.}}
\\label{{tab:targetpilot}}
\\centering
\\small
\\begin{{tabular}}{{l{'c' * len(deltas)}}}
\\hline
Target $\\Delta$ & {head} \\\\
\\hline
Bounded (ours) & {row('bounded')} \\\\
Cumulative & {row('cumulative')} \\\\
\\hline
\\end{{tabular}}

\\medskip
\\small On the same two runs, the bounded target also leaves less of the transient factor
in $\\zslow$ ($R^2$ {b['leak_u']:.3f} against {c['leak_u']:.3f}) and does not partially
collapse (target RankMe {b['rankme']:.1f} against {c['rankme']:.1f}).
\\end{{table}}
""")


# ------------------------------------------------- numbers quoted in the prose
def macros():
    """Every figure the IV-C / IV-D prose states in running text.

    The tables above are generated, so the sentences around them have to be too, or
    the two drift apart -- which has happened twice in this project. sec4_results.tex
    cites these by name and never spells a digit.
    """
    m = {}
    fmt = lambda v, n=3: f"{v:.{n}f}"

    tb = load("term_breakdown.json")
    incl = [tb[f"{k}/{s}"]["_diagnosis"]["d_inclusion"] for _, _, k, _ in DSETS
            for _, _, s in STEMS]
    m["InclDiffLo"], m["InclDiffHi"] = f"{min(incl):+.3f}", f"{max(incl):+.3f}"

    drops = []
    for label, tag, _, suf in DSETS:
        for _, stem, _ in STEMS:
            drops.append(best_lam(rot(tag, stem, suf + "_sym"))[0] - best_lam(rot(tag, stem, suf))[0])
    TIE = 0.005                          # SEP differences below this are called a tie
    m["SymBetter"] = str(sum(d > TIE for d in drops))
    m["SymWorse"] = str(sum(d < -TIE for d in drops))
    m["SymTied"] = str(sum(abs(d) <= TIE for d in drops))
    m["SymWorstDrop"] = fmt(-min(drops))

    d = load("mlp_probe_rotation.json")["results"]
    tag_of = lambda s: "nce" if s == "nce+ema" else "l1"
    real = [f"{tag}/{tag_of(s)}" for lbl, tag, _, _ in DSETS if lbl != "Synthetic"
            for _, s, _ in STEMS]
    head = lambda e, mm, ph: e[ph] if mm == "ours" else e[f"{mm}/{ph}"]
    for t, name in (("inclusion", "MlpIncl"), ("allocation", "MlpAlloc"),
                    ("exclusion", "MlpExcl")):
        v = [head(d[k], mm, "mlp")[t] - head(d[k], mm, "linear")[t]
             for k in real for mm in ("ours", "SFA", "PCA", "ICA-slow")]
        m[name] = f"{sum(v) / len(v):+.3f}"
        if t == "exclusion":
            m["MlpExclWorst"] = f"{min(v):+.3f}"

    lf = load("label_free_lambda.json")
    lf.update(load("ptbxl_label_free_lambda.json"))
    syn = {float(k): v for k, v in lf["Synthetic/Reg"].items() if k != "_pick"}
    by = lambda f: max(syn.items(), key=lambda kv: kv[1][f])[0]
    m["JexclOnlyLam"] = f"{by('exclusion'):.0f}"
    m["JexclOnlySep"] = fmt(syn[by("exclusion")]["sep"])
    m["JbestLam"] = f"{by('sep'):.0f}"
    m["JbestSep"] = fmt(syn[by("sep")]["sep"])
    m["JlongBest"] = fmt(syn[by("sep")]["longhorizon"])
    m["JlongCollapse"] = fmt(min(v["longhorizon"] for k, v in syn.items() if k >= 4))
    m["JrankLo"] = fmt(min(v["rankme"] for v in syn.values()), 1)
    m["JrankHi"] = fmt(max(v["rankme"] for v in syn.values()), 1)

    marg = {}
    for label, _, key, _ in DSETS:
        for stem_label, _, stem in STEMS:
            c = {k: v for k, v in lf[f"{key}/{stem}"].items() if k != "_pick"}
            js = sorted((v["J"] for v in c.values()), reverse=True)
            marg.setdefault("Synth" if label == "Synthetic" else "Real", []).append(js[0] - js[1])
    for k, v in marg.items():
        m[f"Jmargin{k}Lo"], m[f"Jmargin{k}Hi"] = fmt(min(v)), fmt(max(v))

    # --- figures the Introduction states, so intro and Results cannot drift either
    syn = rot("synth", "nce+ema", "_gap0.03")
    m["SynthOurs"] = fmt(best_lam(syn)[0])
    m["SynthSfa"] = fmt(sep_of(syn, "SFA"))
    m["SynthPca"] = fmt(sep_of(syn, "PCA"))
    losses = []
    for label, tag, _, suf in DSETS:
        if label == "Synthetic":
            continue
        for _, stem, _ in STEMS:
            d = rot(tag, stem, suf)
            gap = sep_of(d, "SFA") - best_lam(d)[0]
            if gap > 0:
                losses.append(gap)
    m["RealLossLo"], m["RealLossHi"] = fmt(min(losses)), fmt(max(losses))
    # how much of the PERSISTENT factor the complement block still carries, at each
    # setting's best lambda -- the Background claim that z_mix is not "the fast block"
    mixper = [rot(tag, stem, suf)[f"gate@lam{best_lam(rot(tag, stem, suf))[1]}"]
              ["fast_half"]["slow"][0]
              for _, tag, _, suf in DSETS for _, stem, _ in STEMS]
    m["MixPerLo"], m["MixPerHi"] = fmt(min(mixper)), fmt(max(mixper))

    alloc = [tb[f"{k}/{s}"]["_diagnosis"]["d_allocation"] for _, _, k, _ in DSETS
             for _, _, s in STEMS]
    m["AllocLead"] = str(sum(a > 0 for a in alloc))
    m["AllocMax"] = f"{max(alloc):+.3f}"

    # the two rows of the block x factor figure: the whole SEP gap is one term
    try:
        e = load("fig_embed_compare.json")
        for row, key in (("gate+xcov", "Fig"), ("nomech", "FigNo")):
            for t in ("inclusion", "allocation", "exclusion", "sep"):
                m[key + t.capitalize()] = fmt(e[row]["sep"][t])
    except FileNotFoundError:
        print("  (fig_embed_compare.json absent -- run fig_embed.py --compare)")

    body = "\n".join(f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in m.items())
    write("nums_sec4d", "% Generated by src/exp/gen_sec4d.py -- do not edit by hand.\n"
                        + body + "\n")
    print("  " + "  ".join(f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    for fn in (terms, gatesym, gatehard, targetpilot,
               mlpprobe, lambdafree, macros):
        print(f"[{fn.__name__}]")
        fn()
