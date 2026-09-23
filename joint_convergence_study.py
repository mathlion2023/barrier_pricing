"""Convergence, stability and benchmark study for the joint (X,v) COS/FD
engine (barrier_pricing/cos2d_joint.py), addressing the Q1 referee report
(Q1_Referee_Report_JAM_Paper.pdf), items 2-7, 10 and 11:

  A. matrix-level stability diagnostics of the per-mode operator (item 3)
  B. separate convergence studies in N, dv, dt (levels), v_max, the v_max
     closure, the time integrator and the projection oversampling, with
     observed orders (item 4), and the projection-ratchet diagnostic (4.4)
  C. benchmark table against the independent HV-ADI solver (adi.py), the
     corrected method of lines (mol.py), Monte Carlo (mc.py; European
     only), CKM's published values and the frozen-branch surrogates
     (items 5-7): base tables, tight barriers, Feller-violated and high
     vol-of-vol regimes, jumps
  D. delta/gamma and early-exercise boundary against ADI (item 11)
  E. runtime protocol: hardware/software, JIT warm-up excluded, number of
     spots produced per run (item 10)

Run from this directory:  python joint_convergence_study.py [section ...]
Writes joint_convergence_study_output.txt (human-readable) and
joint_convergence_study_results.json (every number, for the paper tables).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

from barrier_pricing import adi, cos2d, cos2d_american, cos2d_joint as cj, mc, mol
from barrier_pricing.charfunc import cos_vanilla_call
from barrier_pricing.params import HestonParams, JumpParams, NO_JUMPS, OptionSpec

HP = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)
T, K, H, r, q = 0.5, 100.0, 130.0, 0.03, 0.05
SPOTS = [80.0, 90.0, 100.0, 110.0, 120.0]
CKM = {  # Chiarella, Kang & Meyer (2012), finest-grid MOL values
    "eur_cont": {80: 0.9046, 90: 1.8806, 100: 2.5999, 110: 2.4858, 120: 1.4812},
    "eur_disc": {80: 1.0807, 90: 2.5289, 100: 4.1116, 110: 5.0235, 120: 4.8706},
    "am_cont": {80: 1.4012, 90: 3.9364, 100: 8.3003, 110: 14.4033, 120: 21.8219},
    "am_disc": {80: 1.4008, 90: 3.9339, 100: 8.3010, 110: 14.4446, 120: 22.0389},
}
MD = {"eur_cont": None, "eur_disc": [T / 2, 0.0], "am_cont": None, "am_disc": [2 * T / 3, T / 3]}
AM = {"eur_cont": False, "eur_disc": False, "am_cont": True, "am_disc": True}

OUT_TXT = "joint_convergence_study_output.txt"
OUT_JSON = "joint_convergence_study_results.json"
RESULTS: dict = {}
_log = open(OUT_TXT, "a", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _log.write(msg + "\n")
    _log.flush()


def rule(t):
    say(f"\n=== {t} ===")


def spec(S0=100.0, case="am_cont", **kw):
    d = dict(S0=S0, K=K, H=H, r=r, q=q, T=T, american=AM[case], knock_in=False,
             monitor_dates=MD[case])
    d.update(kw)
    return OptionSpec(**d)


def joint(case, hp=HP, jp=NO_JUMPS, spots=SPOTS, **kw):
    kw.setdefault("n_steps", 48 if case.endswith("disc") else 50)
    s = spec(spots[0], case, **{k: kw.pop(k) for k in ("H",) if k in kw})
    res = cj.price_barrier_call(s, hp, jp, spots=spots[1:], **kw)
    return res


def save():
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=1, default=conv)


# ----------------------------------------------------------------------
# A. stability
# ----------------------------------------------------------------------
def section_stability():
    rule("A. Stability diagnostics of the per-mode operator (all modes sampled, boundary rows included)")
    cases = [("base", HP), ("Feller violated", HestonParams(1.0, 0.04, 0.5, -0.5, 0.04)),
             ("high vol-of-vol", HestonParams(4.0, 0.09, 0.4, -0.5, 0.09)),
             ("rho=-0.9", HestonParams(2.0, 0.1, 0.1, -0.9, 0.1))]
    out = {}
    for label, hp in cases:
        for n_steps in (50, 400):
            for bc in ("outflow", "neumann"):
                d = cj.stability_diagnostics(spec(), hp, n_steps=n_steps, N=2048, M=100, vmax_bc=bc)
                say(f"{label:16s} n={n_steps:4d} {bc:8s}: max Re eig(L_k)={d['max_spectral_abscissa']:+.3e}  "
                    f"max||(I-dtL)^-1||_2={d['max_step_norm_2']:.6f}  max||exp(dtL)||_2={d['max_expm_step_norm_2']:.6f}  "
                    f"1/(1+r dt)={d['bound_1_over_1_plus_r_dt']:.6f}  min pivot={d['min_thomas_pivot']:.3e}  u_max={d['u_max']:.0f}")
                out[f"{label}|{n_steps}|{bc}"] = d
    RESULTS["stability"] = out
    save()


# ----------------------------------------------------------------------
# B. convergence studies
# ----------------------------------------------------------------------
def _row(tag, res):
    e = res.extra
    say(f"  {tag:28s} " + " ".join(f"{p:9.5f}" for p in e["prices"])
        + f"   err_est(S=100)={e['err_ests'][SPOTS.index(100.0)] if len(e['prices']) == 5 else e['err_est']:.1e}"
        + f"  {res.runtime:6.1f}s")
    return {"prices": e["prices"], "err": e["err_ests"], "levels": e["level_prices"], "runtime": res.runtime}


def section_convergence():
    rule("B. Separate convergence studies, joint engine (spots 80..120)")
    out = RESULTS.setdefault("convergence", {})  # saved incrementally
    for case in ("am_cont", "eur_disc", "eur_cont"):
        say(f"-- {case}")
        c = {}
        for N in (512, 1024, 2048, 4096):
            c[f"N={N}"] = _row(f"N={N} (M=100)", joint(case, N=N))
        for M in (25, 50, 100, 200):
            c[f"M={M}"] = _row(f"M={M} (N=2048, sinh v-grid)", joint(case, M=M))
        for vm, M in ((0.5, 50), (1.0, 100), (2.0, 200)):
            c[f"vmax={vm}"] = _row(f"v_max={vm} (M/v_max=100)", joint(case, v_max=vm, M=M))
        c["neumann"] = _row("v_max closure Neumann", joint(case, vmax_bc="neumann"))
        c["L=14"] = _row("truncation L=14", joint(case, L=14.0))
        c["oversample=1"] = _row("oversample=1 (DCT interp.)", joint(case, oversample=1))
        for n0 in (25, 50, 100):
            nn = n0 if not case.endswith("disc") else {25: 24, 50: 48, 100: 96}[n0]
            res = joint(case, n_steps=nn)
            c[f"n0={nn}"] = _row(f"base steps n0={nn}", res)
            say(f"      raw levels at S=100: {np.round(res.extra['level_prices'], 5)}  p_hat={res.extra['p_hat']:.2f}")
        out[case] = c
        save()
    say("-- time integrator check (European discrete; expm is exact in time)")
    ex = joint("eur_disc", n_steps=2, time_integrator="expm", extrapolation="none")
    out["eur_disc_expm_exact"] = _row("expm, 2 steps (exact)", ex)
    for n in (8, 16, 48):
        out[f"eur_disc_rbe_{n}"] = _row(f"rbe, {n} steps, no extrap.", joint("eur_disc", n_steps=n, extrapolation="none"))
    say("-- projection ratchet (American, no barrier H=1e4, 'be', no extrapolation): price vs steps n at fixed N")
    rat = {}
    for N in (256, 512, 1024, 2048):
        row = []
        for n in (100, 400, 1600):
            res = cj.price_barrier_call(spec(100.0, "am_cont", H=1e4), HP, NO_JUMPS, n_steps=n, N=N, M=50,
                                        extrapolation="none", barrier_method="zero", time_integrator="be")
            row.append(res.price)
        rat[N] = row
        say(f"  N={N:5d}: n=100 {row[0]:.5f}  n=400 {row[1]:.5f}  n=1600 {row[2]:.5f}")
    ref = adi.price_barrier_call(spec(100.0, "am_cont"), HP, nS=400, nv=200, n_steps=800, no_barrier=True)
    say(f"  ADI American vanilla (400/200/800): {ref.price:.5f}")
    out["ratchet"] = {"rows": rat, "adi": ref.price}
    RESULTS["convergence"] = out
    save()


# ----------------------------------------------------------------------
# C. benchmarks
# ----------------------------------------------------------------------
def _adi_ref(s, hp, spots):
    """ADI at two resolutions. The fine value is the reference and
    |fine - coarse| its error bar (no extrapolation: with American
    exercise the ADI time error is first order, so a fixed-order
    Richardson step would not be justified)."""
    s = OptionSpec(**{**s.__dict__, "S0": spots[0]})  # first spot is S0 (same layout as the joint run)
    lo = adi.price_barrier_call(s, hp, nS=200, nv=100, n_steps=_nt(s, 400), spots=spots[1:])
    hi = adi.price_barrier_call(s, hp, nS=400, nv=200, n_steps=_nt(s, 800), spots=spots[1:])
    return lo.extra["prices"], hi.extra["prices"], np.abs(hi.extra["prices"] - lo.extra["prices"]), lo.runtime + hi.runtime


def _nt(s, n):
    if s.is_continuous():
        return n
    return int(np.ceil(n / 6) * 6)


def _mol(s, hp, jp, spots, nS, M, nt):
    t0 = time.perf_counter()
    p = [mol.price_barrier_call(OptionSpec(**{**s.__dict__, "S0": x}), hp, jp, n_S=nS, M=M,
                                n_time_steps=nt).price for x in spots]
    return np.array(p), time.perf_counter() - t0


def _mc(s, hp, jp, spots, n_paths=1_000_000, n_steps=240, seed=3):
    out, ci = [], []
    for x in spots:
        res = mc.price_barrier_call_mc(OptionSpec(**{**s.__dict__, "S0": x}), hp, jp, n_paths=n_paths,
                                       n_steps=n_steps, seed=seed, greeks=False)
        out.append(res.price)
        ci.append(res.extra["ci95"])
    return np.array(out), np.array(ci)


def _frozen(s, hp, jp, spots):
    p = []
    for x in spots:
        sx = OptionSpec(**{**s.__dict__, "S0": x})
        if sx.american:
            nex = 200 if sx.is_continuous() else 48
            N = 512 if sx.is_continuous() else 256
            p.append(cos2d_american.price_barrier_call(sx, hp, jp, n_exercise=nex, N=N, M=200).price)
        elif not sx.is_continuous():
            p.append(cos2d.price_barrier_call(sx, hp, jp, N=1024, M=200).price)
        else:
            p.append(np.nan)
    return np.array(p)


def _bench(label, s, hp=HP, jp=NO_JUMPS, spots=SPOTS, ckm=None, with_mc=False, with_adi=True,
           mol_grids=((150, 150, 200), (250, 250, 400)), joint_kw=None):
    say(f"-- {label}")
    say("  spots: " + " ".join(f"{x:9.1f}" for x in spots))
    rec = {"spots": spots}
    jr = cj.price_barrier_call(OptionSpec(**{**s.__dict__, "S0": spots[0]}), hp, jp, spots=spots[1:],
                               **({"n_steps": 48 if not s.is_continuous() else 50} | (joint_kw or {})))
    rec["joint"] = {"prices": jr.extra["prices"], "err": jr.extra["err_ests"], "runtime": jr.runtime,
                    "method": jr.extra["barrier_method"], "extrapolation": jr.extra["extrapolation"]}
    say("  joint          " + " ".join(f"{x:9.5f}" for x in jr.extra["prices"]) + f"   ({jr.runtime:.1f}s, one run)")
    say("  joint err est  " + " ".join(f"{x:9.1e}" for x in jr.extra["err_ests"]))
    if with_adi and not jp.active:
        lo, hi, bar, rt = _adi_ref(s, hp, spots)
        rec["adi"] = {"lo": lo, "hi": hi, "bar": bar, "runtime": rt}
        say("  ADI 200/100/.. " + " ".join(f"{x:9.5f}" for x in lo))
        say("  ADI 400/200/.. " + " ".join(f"{x:9.5f}" for x in hi))
        say("  ADI |fine-crs| " + " ".join(f"{x:9.1e}" for x in bar))
        say("  joint-ADI rel  " + " ".join(f"{100 * (a - b) / b:+8.3f}%" if b > 1e-3 else "      n/a" for a, b in zip(jr.extra["prices"], hi)))
    for g in mol_grids:
        p, rt = _mol(s, hp, jp, spots, *g)
        rec[f"mol_{g[0]}_{g[1]}_{g[2]}"] = {"prices": p, "runtime": rt}
        say(f"  MOL {g[0]}/{g[1]}/{g[2]:<4d}" + " ".join(f"{x:9.5f}" for x in p) + f"   ({rt:.1f}s)")
    if with_mc and not s.american:
        p, ci = _mc(s, hp, jp, spots)
        rec["mc"] = {"prices": p, "ci95": ci}
        say("  MC (1e6)       " + " ".join(f"{x:9.4f}" for x in p))
        say("  MC 95% CI +/-  " + " ".join(f"{x:9.4f}" for x in ci))
    fz = _frozen(s, hp, jp, spots)
    if np.isfinite(fz).any():
        rec["frozen"] = fz
        say("  frozen-branch  " + " ".join(f"{x:9.5f}" for x in fz))
    if ckm:
        rec["ckm"] = [ckm.get(int(x), np.nan) for x in spots]
        say("  CKM published  " + " ".join(f"{x:9.4f}" for x in rec["ckm"]))
    return rec


def section_benchmarks():
    rule("C. Benchmarks: joint vs independent ADI vs corrected MOL vs MC vs CKM vs frozen-branch")
    out = RESULTS.setdefault("benchmarks", {})  # saved incrementally by save()
    for case, lab in (("eur_cont", "Table 2: European, continuous"),
                      ("eur_disc", "Table 4: European, discrete (T/2, T)"),
                      ("am_cont", "Table 3: American, continuous"),
                      ("am_disc", "Table 5: American, discrete (T/3, 2T/3, T)")):
        out[case] = _bench(lab, spec(100.0, case), ckm=CKM[case], with_mc=not AM[case])
        save()
    sp3 = [100.0, 95.0, 90.0]
    for Hl in (105.0, 102.0):
        out[f"tight_H{int(Hl)}"] = _bench(f"Tight barrier H={Hl}, American continuous (bound H-K={Hl - K})",
                                           spec(100.0, "am_cont", H=Hl), spots=sp3)
        save()
    for label, hp in (("Feller violated (kappa=1, theta=0.04, sigma=0.5, v0=0.04)", HestonParams(1.0, 0.04, 0.5, -0.5, 0.04)),
                      ("High vol-of-vol (kappa=4, theta=0.09, sigma=0.4, v0=0.09)", HestonParams(4.0, 0.09, 0.4, -0.5, 0.09))):
        for case in ("am_cont", "eur_disc"):
            key = f"{label.split(' (')[0]}|{case}"
            out[key] = _bench(f"{label}, {case}", spec(100.0, case), hp=hp, spots=[80.0, 100.0, 120.0],
                              with_mc=not AM[case], joint_kw={"M": 200})
            save()
    jp = JumpParams(lam=0.5, gam=-0.05, del_=0.1)
    out["jumps_eur_disc"] = _bench("Jumps (lam=0.5, gam=-0.05, del=0.1), European discrete", spec(100.0, "eur_disc"),
                                   jp=jp, spots=[80.0, 100.0, 120.0], with_mc=True, with_adi=False)
    out["jumps_am_cont"] = _bench("Jumps, American continuous", spec(100.0, "am_cont"), jp=jp,
                                  spots=[80.0, 100.0, 120.0], with_adi=False)
    RESULTS["benchmarks"] = out
    save()


# ----------------------------------------------------------------------
# D. Greeks and exercise boundary
# ----------------------------------------------------------------------
def section_greeks():
    rule("D. Delta, gamma and early-exercise boundary vs ADI (American, continuous)")
    spots = [80.0, 90.0, 100.0, 110.0, 115.0, 120.0, 125.0]
    s = spec(100.0, "am_cont")
    jr = cj.price_barrier_call(OptionSpec(**{**s.__dict__, "S0": spots[0]}), HP, NO_JUMPS, spots=spots[1:],
                               return_boundary=True)
    ar = adi.price_barrier_call(OptionSpec(**{**s.__dict__, "S0": spots[0]}), HP, nS=800, nv=200, n_steps=800,
                                spots=spots[1:], return_grid=True)
    say("  spots        " + " ".join(f"{x:9.1f}" for x in spots))
    for name, a, b in (("price", jr.extra["prices"], ar.extra["prices"]),
                       ("delta", jr.extra["deltas"], ar.extra["deltas"]),
                       ("gamma", jr.extra["gammas"], ar.extra["gammas"])):
        say(f"  joint {name:6s} " + " ".join(f"{x:9.5f}" for x in a))
        say(f"  ADI   {name:6s} " + " ".join(f"{x:9.5f}" for x in b))
    vj, Sj = jr.extra["boundary"]["v"], jr.extra["boundary"]["S_star"]
    va, Sa = adi.exercise_boundary(ar, K, H)
    # Resolution-robustness check on the joint boundary: near a continuously
    # monitored barrier, the Bermudan/time-discretization bias in the raw
    # continuation-vs-intrinsic crossing does not vanish at the default
    # n_steps=400 finest level -- it keeps drifting toward H as n_steps grows
    # (checked up to n_steps=1600: S*(v=0.10) moves 128.49 -> 128.92 -> 129.22,
    # never stabilizing), while ADI's Ikonen-Toivanen Lagrange multiplier
    # (exact LCP complementarity, unaffected by this bias) shows no exercise
    # activation at all in that region -- consistent with Broadie-Detemple
    # (1995): when the unconstrained American boundary lies above the cap,
    # there is no interior free boundary. A second, independent single-level
    # run at n_steps=1600 lets us flag rows where the joint diagnostic has not
    # converged rather than report a drifting artifact.
    jr_fine = cj.price_barrier_call(spec(100.0, "am_cont"), HP, NO_JUMPS, extrapolation="none",
                                    n_steps=1600, return_boundary=True)
    vf, Sf = jr_fine.extra["boundary"]["v"], jr_fine.extra["boundary"]["S_star"]
    say("  exercise boundary S*(v) at t=0 (joint: n_steps=400 vs 1600 stability check; ADI: Lagrange multiplier):")
    comp = []
    for vv in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        sj = float(np.interp(vv, vj, Sj))
        sf = float(np.interp(vv, vf, Sf))
        sa = float(np.interp(vv, va, Sa))
        stable = abs(sf - sj) < 0.5
        sj_report = sf if stable else np.nan
        comp.append((vv, sj_report, sa))
        say(f"    v={vv:.2f}: joint(400) {sj:8.3f}  joint(1600) {sf:8.3f}  "
            f"{'stable' if stable else 'DRIFTING (unresolved, reported as --)'}   ADI {sa:8.3f}")
    RESULTS["greeks"] = {"spots": spots, "joint": {k: jr.extra[k] for k in ("prices", "deltas", "gammas")},
                         "adi": {k: ar.extra[k] for k in ("prices", "deltas", "gammas")}, "boundary": comp}
    save()


# ----------------------------------------------------------------------
# E. runtime protocol
# ----------------------------------------------------------------------
def section_runtime():
    rule("E. Runtime protocol")
    import numba
    import scipy
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}
    for k, v in info.items():
        say(f"  {k}: {v}")
    say("  all timings: wall clock, JIT compilation excluded (one warm-up call first), single process;")
    say("  NumPy/SciPy FFT and BLAS may use several threads. 'spots' = number of spot prices returned by one run.")
    s = spec(100.0, "am_cont")
    cj.price_barrier_call(s, HP, NO_JUMPS, n_steps=10, N=256, M=20, extrapolation="none")  # warm-up
    mol.price_barrier_call(s, HP, NO_JUMPS, n_S=40, M=20, n_time_steps=10)
    rows = []
    for label, fn, nspots in (
        ("joint N=2048 M=100 n0=50 (sqrt, 4 levels)", lambda: cj.price_barrier_call(s, HP, NO_JUMPS, spots=SPOTS[:-1]), 5),
        ("joint N=1024 M=50 n0=25 (sqrt, 4 levels)", lambda: cj.price_barrier_call(s, HP, NO_JUMPS, N=1024, M=50, n_steps=25, spots=SPOTS[:-1]), 5),
        ("ADI 200/100/200", lambda: adi.price_barrier_call(s, HP, nS=200, nv=100, n_steps=200, spots=SPOTS[:-1]), 5),
        ("ADI 400/200/400", lambda: adi.price_barrier_call(s, HP, nS=400, nv=200, n_steps=400, spots=SPOTS[:-1]), 5),
        ("MOL 150/150/200 (one spot per run)", lambda: mol.price_barrier_call(s, HP, NO_JUMPS, n_S=150, M=150, n_time_steps=200), 1),
        ("frozen-branch COS n_ex=200 N=512 (one spot)", lambda: cos2d_american.price_barrier_call(s, HP, NO_JUMPS, n_exercise=200, N=512, M=200), 1),
    ):
        t0 = time.perf_counter()
        res = fn()
        dt_ = time.perf_counter() - t0
        rows.append({"label": label, "seconds": dt_, "spots": nspots, "price_S100": res.price})
        say(f"  {label:48s} {dt_:8.2f}s  spots/run={nspots}  price(S=100)={res.price:.5f}")
    RESULTS["runtime"] = {"info": info, "rows": rows}
    save()


def _cos_capped_call(S0, hp, N=4096, L=14.0):
    """Closed-form (1D COS, exact payoff coefficients) price of the
    maturity-only-monitored up-and-out call e^{-rT} E[(S_T-K)^+ 1{S_T<H}]."""
    from barrier_pricing.charfunc import _chi_psi_call, _svjd_cumulants, heston_jump_cf
    c1, c2, c4 = _svjd_cumulants(T, r, q, hp, NO_JUMPS)
    sd = np.sqrt(c2 + np.sqrt(c4))
    a, b = c1 - L * sd, c1 + L * sd
    k = np.arange(N)
    u = k * np.pi / (b - a)
    Uk = 2.0 / (b - a) * _chi_psi_call(a, b, k, 0.0, np.log(H / K))
    term = heston_jump_cf(u, T, r, q, hp) * np.exp(1j * u * (np.log(S0 / K) - a)) * Uk
    term[0] *= 0.5
    return float(np.exp(-r * T) * K * np.real(term.sum()))


def section_adi_validation():
    rule("V. Validation of the ADI benchmark against closed forms")
    spots = [80.0, 100.0, 120.0]
    exact_v = [cos_vanilla_call(s, K, r, q, T, HP) for s in spots]
    exact_b = [_cos_capped_call(s, HP) for s in spots]
    say("  spots                 " + " ".join(f"{x:10.1f}" for x in spots))
    say("  vanilla, closed form  " + " ".join(f"{x:10.5f}" for x in exact_v))
    out = {"spots": spots, "vanilla_exact": exact_v, "maturity_barrier_exact": exact_b, "vanilla": {}, "maturity_barrier": {}}
    for g in ((100, 50, 100), (200, 100, 200), (400, 200, 400), (800, 200, 800)):
        res = adi.price_barrier_call(spec(spots[0], "eur_cont"), HP, nS=g[0], nv=g[1], n_steps=g[2],
                                     no_barrier=True, spots=spots[1:])
        out["vanilla"][str(g)] = res.extra["prices"]
        say(f"  vanilla ADI {g}  " + " ".join(f"{x:10.5f}" for x in res.extra["prices"])
            + "   max abs err " + f"{np.max(np.abs(res.extra['prices'] - exact_v)):.1e}")
    say("  maturity-only barrier, closed form " + " ".join(f"{x:10.5f}" for x in exact_b))
    for g in ((100, 50, 100), (200, 100, 200), (400, 200, 400), (800, 200, 800)):
        res = adi.price_barrier_call(spec(spots[0], "eur_cont", monitor_dates=[0.0]), HP, nS=g[0], nv=g[1],
                                     n_steps=g[2], spots=spots[1:])
        out["maturity_barrier"][str(g)] = res.extra["prices"]
        say(f"  barrier ADI {g}  " + " ".join(f"{x:10.5f}" for x in res.extra["prices"])
            + "   max abs err " + f"{np.max(np.abs(res.extra['prices'] - exact_b)):.1e}")
    RESULTS["adi_validation"] = out
    save()


def section_benchmarks_amcont():
    """Re-run only the continuously monitored American entries of section C
    (the ones that depend on the order of the exercise and barrier steps)."""
    rule("C'. Benchmarks, American continuous entries (exercise before the barrier step)")
    out = RESULTS.setdefault("benchmarks", {})
    out["am_cont"] = _bench("Table 3: American, continuous", spec(100.0, "am_cont"), ckm=CKM["am_cont"])
    save()
    sp3 = [100.0, 95.0, 90.0]
    for Hl in (105.0, 102.0):
        out[f"tight_H{int(Hl)}"] = _bench(f"Tight barrier H={Hl}, American continuous (bound H-K={Hl - K})",
                                           spec(100.0, "am_cont", H=Hl), spots=sp3)
        save()
    for label, hp in (("Feller violated (kappa=1, theta=0.04, sigma=0.5, v0=0.04)", HestonParams(1.0, 0.04, 0.5, -0.5, 0.04)),
                      ("High vol-of-vol (kappa=4, theta=0.09, sigma=0.4, v0=0.09)", HestonParams(4.0, 0.09, 0.4, -0.5, 0.09))):
        key = f"{label.split(' (')[0]}|am_cont"
        out[key] = _bench(f"{label}, am_cont", spec(100.0, "am_cont"), hp=hp, spots=[80.0, 100.0, 120.0],
                          joint_kw={"M": 200})
        save()
    jp = JumpParams(lam=0.5, gam=-0.05, del_=0.1)
    out["jumps_am_cont"] = _bench("Jumps, American continuous", spec(100.0, "am_cont"), jp=jp,
                                  spots=[80.0, 100.0, 120.0], with_adi=False)
    save()


def section_benchmarks_jumps_eur():
    rule("C''. Benchmarks, jumps, European discrete")
    out = RESULTS.setdefault("benchmarks", {})
    jp = JumpParams(lam=0.5, gam=-0.05, del_=0.1)
    out["jumps_eur_disc"] = _bench("Jumps (lam=0.5, gam=-0.05, del=0.1), European discrete", spec(100.0, "eur_disc"),
                                   jp=jp, spots=[80.0, 100.0, 120.0], with_mc=True, with_adi=False)
    save()


SECTIONS = {"adi_validation": section_adi_validation, "benchmarks_amcont": section_benchmarks_amcont,
            "benchmarks_jumps_eur": section_benchmarks_jumps_eur, "stability": section_stability, "convergence": section_convergence,
            "benchmarks": section_benchmarks, "greeks": section_greeks, "runtime": section_runtime}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SECTIONS)
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            RESULTS.update(json.load(f))
    say(f"\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')}  sections={wanted}")
    for w in wanted:
        SECTIONS[w]()
    say("done")
