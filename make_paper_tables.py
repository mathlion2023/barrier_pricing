"""Generate every table and in-text number of JAM_Paper_revised.tex from
joint_convergence_study_results.json (written by joint_convergence_study.py).

Run from this directory:  python make_paper_tables.py
Writes tables/*.tex and tables/numbers.tex. No number in the paper's
results section is typed by hand.
"""
from __future__ import annotations

import json
import os

import numpy as np

SRC = "joint_convergence_study_results.json"
OUT = "tables"
os.makedirs(OUT, exist_ok=True)
R = json.load(open(SRC, encoding="utf-8")) if os.path.exists(SRC) else {}
NUM: dict[str, str] = {}


def w(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)


def f5(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.4f}"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return "$" + m + r"\times10^{" + str(int(e)) + "}$"


def pct(x, d=2):
    return "$" + f"{x:.{d}f}" + r"\%$"


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a - b) / b


# ---------------------------------------------------------------- stability
def t_stability():
    S = R.get("stability", {})
    rows, cmax = [], 0.0
    for key, d in S.items():
        label, n, bc = key.split("|")
        if bc != "outflow":
            continue
        c = (d["max_step_norm_2"] - 1.0) / d["dt"]
        cmax = max(cmax, c)
        rows.append(f"{label} & {n} & {d['max_spectral_abscissa']:+.4f} & {d['max_step_norm_2']:.4f} & "
                    f"{d['max_expm_step_norm_2']:.4f} & {d['min_thomas_pivot']:.2f} & {c:.2f} \\\\")
    NUM["StabCmax"] = f"{np.ceil(cmax * 10) / 10:.1f}" if rows else "--"
    w("tab_stability.tex",
      "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
      "Regime & $T/\\Dt$ & $\\max\\operatorname{Re}\\lambda(L_k)$ & $\\max\\|(I-\\Dt L_k)^{-1}\\|_2$ & "
      "$\\max\\|e^{\\Dt L_k}\\|_2$ & min pivot & $c$ \\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------- ADI validation
def t_adivalid():
    A = R.get("adi_validation")
    if not A:
        w("tab_adivalid.tex", "\\emph{(ADI validation pending)}\n")
        return
    ev, eb = np.array(A["vanilla_exact"]), np.array(A["maturity_barrier_exact"])
    rows, prev = [], None
    for g in A["vanilla"]:
        e1 = np.max(np.abs(np.array(A["vanilla"][g]) - ev))
        e2 = np.max(np.abs(np.array(A["maturity_barrier"][g]) - eb))
        gg = g.strip("()").replace(" ", "")
        ratio = "" if prev is None else f"{prev[0] / e1:.1f} & {prev[1] / e2:.1f}"
        rows.append(f"$({gg})$ & {e1:.1e} & {e2:.1e} & {ratio if ratio else '-- & --'} \\\\")
        prev = (e1, e2)
    w("tab_adivalid.tex",
      "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n$(n_S,n_v,n_t)$ & vanilla error & barrier error & "
      "vanilla ratio & barrier ratio \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------- benchmark tables
MOLKEY = "mol_250_250_400"


def bench_block(rec, caption, label, with_mc, with_ckm=True):
    spots = rec["spots"]
    J = np.array(rec["joint"]["prices"])
    Je = np.array(rec["joint"]["err"])
    cols = [("Joint", J), ("Joint error est.", None)]
    lines = []
    head = "$S_0$ & Joint & est.\\ error"
    if "adi" in rec:
        A = np.array(rec["adi"]["hi"])
        head += " & ADI & ADI bar & rel.\\ diff."
    if MOLKEY in rec:
        head += " & MOL"
    if with_mc and "mc" in rec:
        head += " & MC ($\\pm$95\\%)"
    if "frozen" in rec:
        head += " & Frozen"
    if with_ckm and "ckm" in rec:
        head += " & CKM"
    ncol = head.count("&") + 1
    for i, s in enumerate(spots):
        row = f"{s:.0f} & {J[i]:.4f} & {Je[i]:.0e}"
        if "adi" in rec:
            a = rec["adi"]["hi"][i]
            row += f" & {a:.4f} & {rec['adi']['bar'][i]:.0e} & " + ("$" + f"{100 * (J[i] - a) / a:+.3f}" + r"\%$" if a > 1e-3 else "--")
        if MOLKEY in rec:
            row += f" & {rec[MOLKEY]['prices'][i]:.4f}"
        if with_mc and "mc" in rec:
            row += f" & {rec['mc']['prices'][i]:.4f} ($\\pm${rec['mc']['ci95'][i]:.4f})"
        if "frozen" in rec:
            fz = rec["frozen"][i]
            row += " & " + ("--" if fz is None or not np.isfinite(fz) else f"{fz:.4f}")
        if with_ckm and "ckm" in rec:
            ck = rec["ckm"][i]
            row += " & " + ("--" if ck is None or (isinstance(ck, float) and not np.isfinite(ck)) else f"{ck:.4f}")
        lines.append(row + " \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}" + "r" * ncol + "@{}}\n\\toprule\n" + head +
            " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def t_benchmarks():
    B = R.get("benchmarks", {})
    blocks, maxrel, maxfz = [], 0.0, 0.0
    spec = [("eur_cont", "European up-and-out call, continuous monitoring.", "tab:eurcont", True),
            ("eur_disc", "European up-and-out call, monitoring at $T/2$ and $T$.", "tab:eurdisc", True),
            ("am_cont", "American up-and-out call, continuous monitoring.", "tab:amcont", False),
            ("am_disc", "American up-and-out call, monitoring at $T/3$, $2T/3$ and $T$.", "tab:amdisc", False)]
    for key, cap, lab, mc in spec:
        if key not in B:
            blocks.append(f"\\begin{{table}}[t]\\centering\\caption{{{cap} (pending)}}\\label{{{lab}}}\\end{{table}}\n")
            continue
        rec = B[key]
        blocks.append(bench_block(rec, cap + " Joint: extrapolated price and its error estimate. "
                                  "ADI: $(400,200,800)$ grid and difference from $(200,100,400)$.", lab, mc))
        if "adi" in rec:
            maxrel = max(maxrel, float(np.max(np.abs(rel(rec["joint"]["prices"], rec["adi"]["hi"])))))
            if "frozen" in rec:
                fz = np.array(rec["frozen"], float)
                ok = np.isfinite(fz)
                if ok.any():
                    maxfz = max(maxfz, float(np.max(np.abs(rel(fz[ok], np.array(rec["adi"]["hi"])[ok])))))
    NUM["JointAdiMaxBase"] = pct(100 * maxrel, 3) if maxrel else "--"
    NUM["FrozenBaseMaxErr"] = pct(100 * maxfz, 1) if maxfz else "--"
    w("tab_benchmarks.tex", "\n".join(blocks))


def t_tight():
    B = R.get("benchmarks", {})
    lines, jmax, fz = [], 0.0, {}
    for Hl in (105, 102):
        rec = B.get(f"tight_H{Hl}")
        if not rec:
            continue
        for i, s in enumerate(rec["spots"]):
            a = rec["adi"]["hi"][i]
            j = rec["joint"]["prices"][i]
            m = rec[MOLKEY]["prices"][i] if MOLKEY in rec else np.nan
            f = rec["frozen"][i] if "frozen" in rec else np.nan
            jmax = max(jmax, abs(j - a) / a)
            if i == 0:
                fz[Hl] = (f - a) / a
            lines.append(f"{Hl} & {s:.0f} & {Hl - 100:.0f} & {j:.4f} & {a:.4f} & {m:.4f} & {f:.4f} \\\\")
    NUM["TightJointAdiMax"] = pct(100 * jmax, 3) if lines else "--"
    NUM["FrozenTightErrA"] = pct(100 * fz.get(105, np.nan), 0) if 105 in fz else "--"
    NUM["FrozenTightErrB"] = pct(100 * fz.get(102, np.nan), 0) if 102 in fz else "--"
    NUM["FrozenTightMaxErr"] = pct(100 * max(abs(x) for x in fz.values()), 0) if fz else "--"
    w("tab_tight.tex",
      "\\begin{table}[t]\n\\centering\n\\small\n\\caption{Tight barriers, American, continuous monitoring. "
      "The price is bounded by $H-K$.}\n\\label{tab:tight}\n\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n"
      "$H$ & $S_0$ & $H-K$ & Joint & ADI & MOL & Frozen \\\\\n\\midrule\n" + "\n".join(lines) +
      "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def t_stress():
    B = R.get("benchmarks", {})
    lines, jmax, molmax, fzmax = [], 0.0, 0.0, 0.0
    for reg, lab in (("Feller violated", "Feller violated"), ("High vol-of-vol", "High vol.\\ of var.")):
        for case, cl in (("am_cont", "Am.\\ cont."), ("eur_disc", "Eur.\\ disc.")):
            rec = B.get(f"{reg}|{case}")
            if not rec:
                continue
            for i, s in enumerate(rec["spots"]):
                a = rec["adi"]["hi"][i]
                j = rec["joint"]["prices"][i]
                m150 = rec.get("mol_150_150_200", {}).get("prices", [np.nan] * 5)[i]
                m250 = rec.get(MOLKEY, {}).get("prices", [np.nan] * 5)[i]
                f = rec["frozen"][i] if "frozen" in rec else np.nan
                mc = rec["mc"]["prices"][i] if "mc" in rec else np.nan
                if a > 1e-2:
                    jmax = max(jmax, abs(j - a) / a)
                    molmax = max(molmax, abs(m250 - a) / a)
                    if np.isfinite(f):
                        fzmax = max(fzmax, abs(f - a) / a)
                lines.append(f"{lab} & {cl} & {s:.0f} & {j:.4f} & {a:.4f} & {m150:.4f} & {m250:.4f} & "
                             f"{'--' if not np.isfinite(mc) else f'{mc:.4f}'} & {'--' if not np.isfinite(f) else f'{f:.4f}'} \\\\")
    NUM["StressJointAdiMax"] = pct(100 * jmax, 2) if lines else "--"
    NUM["MolStressErr"] = pct(100 * molmax, 0) if lines else "--"
    NUM["FrozenStressErr"] = pct(100 * fzmax, 0) if lines else "--"
    w("tab_stress.tex",
      "\\begin{table}[t]\n\\centering\n\\small\n\\caption{Feller-violated ($\\kappa=1,\\theta=0.04,\\sigma=0.5,v_0=0.04$) "
      "and high volatility-of-variance ($\\kappa=4,\\theta=0.09,\\sigma=0.4,v_0=0.09$) regimes. MOL on two grids "
      "shows divergence under refinement.}\n\\label{tab:stress}\n\\setlength{\\tabcolsep}{4pt}\n"
      "\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
      "Regime & Contract & $S_0$ & Joint & ADI & MOL$_{150}$ & MOL$_{250}$ & MC & Frozen \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def t_jumps():
    B = R.get("benchmarks", {})
    lines = []
    for key, lab in (("jumps_eur_disc", "Eur.\\ disc."), ("jumps_am_cont", "Am.\\ cont.")):
        rec = B.get(key)
        if not rec:
            continue
        for i, s in enumerate(rec["spots"]):
            j = rec["joint"]["prices"][i]
            m = rec.get(MOLKEY, {}).get("prices", [np.nan] * 5)[i]
            mc = f"{rec['mc']['prices'][i]:.4f} ($\\pm${rec['mc']['ci95'][i]:.4f})" if "mc" in rec else "--"
            f = rec["frozen"][i] if "frozen" in rec else np.nan
            lines.append(f"{lab} & {s:.0f} & {j:.4f} & {m:.4f} & {mc} & {'--' if not np.isfinite(f) else f'{f:.4f}'} \\\\")
    w("tab_jumps.tex",
      "\\begin{table}[t]\n\\centering\n\\small\n\\caption{Jumps: $\\lambda=0.5$, $\\gamma=-0.05$, $\\delta=0.1$.}\n"
      "\\label{tab:jumps}\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
      "Contract & $S_0$ & Joint & MOL & MC ($\\pm$95\\%) & Frozen \\\\\n\\midrule\n" + "\n".join(lines) +
      "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ------------------------------------------------------------ convergence
def t_convergence():
    C = R.get("convergence", {}).get("am_cont")
    if not C:
        w("tab_convergence.tex", "\\begin{table}[t]\\centering\\caption{Convergence (pending)}\\label{tab:convergence}\\end{table}\n")
        w("tab_ratchet.tex", "\\begin{table}[t]\\centering\\caption{Ratchet (pending)}\\label{tab:ratchet}\\end{table}\n")
        NUM.setdefault("VmaxBcEffect", "--")
        return
    ref = np.array(C["N=2048"]["prices"])
    order = [("N=512", "$N=512$"), ("N=1024", "$N=1024$"), ("N=2048", "$N=2048$ (reference)"), ("N=4096", "$N=4096$"),
             ("M=25", "$M=25$"), ("M=50", "$M=50$"), ("M=200", "$M=200$"),
             ("vmax=0.5", "$v_{\\max}=0.5$"), ("vmax=2.0", "$v_{\\max}=2$"),
             ("neumann", "Neumann row at $v_{\\max}$"), ("L=14", "truncation $L=14$"),
             ("oversample=1", "no oversampling ($\\vartheta=1$)"),
             ("n0=25", "base level $n=25$"), ("n0=100", "base level $n=100$")]
    lines, eff = [], 0.0
    for k, lab in order:
        if k not in C:
            continue
        p = np.array(C[k]["prices"])
        d = np.max(np.abs(p - ref) / ref)
        if k in ("vmax=0.5", "vmax=2.0", "neumann", "L=14"):
            eff = max(eff, d)
        lines.append(f"{lab} & " + " & ".join(f"{x:.4f}" for x in p) + f" & {d:.1e} & {C[k]['runtime']:.0f} \\\\")
    NUM["VmaxBcEffect"] = sci(eff) if eff else "--"
    w("tab_convergence.tex",
      "\\begin{table}[t]\n\\centering\n\\small\n\\caption{One-at-a-time refinement, American, continuous monitoring. "
      "Reference: $N=2048$, $M=100$, $v_{\\max}=1$, outflow row, $L=10$, $\\vartheta=4$, $n=50$. "
      "Last columns: maximum relative change against the reference and runtime (s).}\n\\label{tab:convergence}\n"
      "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
      "Variant & 80 & 90 & 100 & 110 & 120 & max rel.\\ change & s \\\\\n\\midrule\n" + "\n".join(lines) +
      "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    Rt = R.get("convergence", {}).get("ratchet")
    if Rt:
        rl = [f"{N} & " + " & ".join(f"{x:.4f}" for x in row) + " \\\\" for N, row in Rt["rows"].items()]
        w("tab_ratchet.tex",
          "\\begin{table}[t]\n\\centering\n\\small\n\\caption{Projection ratchet. American call without barrier, "
          f"$S_0=100$, backward Euler, no extrapolation. ADI reference: {Rt['adi']:.4f}.}}\n\\label{{tab:ratchet}}\n"
          "\\begin{tabular}{@{}rrrr@{}}\n\\toprule\n$N$ & $n=100$ & $n=400$ & $n=1600$ \\\\\n\\midrule\n" +
          "\n".join(rl) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def t_greeks():
    G = R.get("greeks")
    if not G:
        w("tab_greeks.tex", "\\begin{table}[t]\\centering\\caption{Greeks (pending)}\\label{tab:greeks}\\end{table}\n")
        return
    sp = G["spots"]
    rows = []
    for nm, lab in (("prices", "price"), ("deltas", "delta"), ("gammas", "gamma")):
        rows.append(f"Joint {lab} & " + " & ".join(f"{x:.4f}" for x in G["joint"][nm]) + " \\\\")
        rows.append(f"ADI {lab} & " + " & ".join(f"{x:.4f}" for x in G["adi"][nm]) + " \\\\")
    bl = " \\\\\n".join(f"{v:.2f} & " + ("--" if not np.isfinite(sj) else f"{sj:.2f}") + " & "
                        + ("--" if not np.isfinite(sa) else f"{sa:.2f}")
                        for v, sj, sa in G["boundary"])
    w("tab_greeks.tex",
      "\\begin{table}[t]\n\\centering\n\\small\n\\caption{Price, delta and gamma (top) and exercise boundary "
      "$S^*(v)$ at $t=0$ (bottom), American, continuous monitoring. For $v\\ge0.05$ neither method finds an "
      "interior boundary below $H$: the joint diagnostic drifts toward $H$ under refinement rather than "
      "stabilising (checked to $n=1600$) and is reported as unresolved; ADI's Lagrange multiplier shows no "
      "exercise activation, consistent with no boundary below the cap.}\n\\label{tab:greeks}\n"
      "\\begin{tabular}{@{}l" + "r" * len(sp) + "@{}}\n\\toprule\n$S_0$ & " + " & ".join(f"{s:.0f}" for s in sp) +
      " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{6pt}\n"
      "\\begin{tabular}{@{}rrr@{}}\n\\toprule\n$v$ & $S^*$ joint & $S^*$ ADI \\\\\n\\midrule\n" + bl +
      " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def t_runtime():
    E = R.get("runtime")
    if not E:
        w("tab_runtime.tex", "\\begin{table}[t]\\centering\\caption{Runtime (pending)}\\label{tab:runtime}\\end{table}\n")
        return
    info = E["info"]
    rows = [f"{r_['label'].replace('_', chr(92) + '_')} & {r_['seconds']:.1f} & {r_['spots']} & "
            f"{r_['seconds'] / r_['spots']:.1f} & {r_['price_S100']:.4f} \\\\"
            for r_ in E["rows"]]
    cap = (f"Wall-clock time, JIT compilation excluded. Python {info['python']}, NumPy {info['numpy']}, "
           f"SciPy {info['scipy']}, Numba {info['numba']}, {info['cpu_count']} logical CPUs ({info['platform']}).")
    w("tab_runtime.tex",
      "\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + cap.replace("_", "\\_") + "}\n\\label{tab:runtime}\n"
      "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nMethod & s per run & spots & s per spot & price ($S_0=100$) \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


for fn in (t_stability, t_adivalid, t_benchmarks, t_tight, t_stress, t_jumps, t_convergence, t_greeks, t_runtime):
    fn()
for k in ("StabCmax", "JointAdiMaxBase", "FrozenBaseMaxErr", "TightJointAdiMax", "FrozenTightErrA",
          "FrozenTightErrB", "FrozenTightMaxErr", "StressJointAdiMax", "MolStressErr", "FrozenStressErr",
          "VmaxBcEffect"):
    NUM.setdefault(k, "--")
w("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in NUM.items()) + "\n")
print("tables written:", sorted(os.listdir(OUT)))
print(NUM)
