"""Direct numerical validation of the Sec. 5.4 two-term time-error model
c1*dt^{1/2} + c2*dt for continuously monitored American contracts,
addressing Reviewer 2's major comment #2 on the Nov 2026 JAM decision
letter for manuscript 8327021: Table 10 in the manuscript only reports
sensitivity of the already-extrapolated price to the base level n0, not
the convergence of the raw (unextrapolated) price sequence itself.

For 1-2 representative continuously monitored American cases, this script
prices the contract at a single level (extrapolation="none") for a
doubling sequence of n_steps, with fine, fixed spatial resolution (N, M)
so spatial/projection error does not contaminate the time-error signal.
It reports:
  - the raw unextrapolated price at each level,
  - successive differences and their ratios (should -> sqrt(2) approx
    1.414 while the dt^{1/2} term dominates, moving toward 2 once a run
    of levels is fine enough that the dt term dominates instead),
  - a least-squares fit of C(n) = C_inf + a/sqrt(n) + b/n against the
    finest levels, compared with pure dt^{1/2}-only and dt-only fits,
    reporting the residual norm of each so the two-term model's fit
    quality can be judged directly against the alternatives.

Run from this directory: python dt_expansion_validation.py
Writes dt_expansion_validation_output.txt and .json.
"""
from __future__ import annotations

import json
import time

import numpy as np

from barrier_pricing import cos2d_joint as cj
from barrier_pricing.params import HestonParams, JumpParams, NO_JUMPS, OptionSpec

HP = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)
T, K, H, r, q = 0.5, 100.0, 130.0, 0.03, 0.05
N_FINE, M_FINE = 4096, 150
LEVELS = [50, 100, 200, 400, 800, 1600, 3200]

OUT_TXT = "dt_expansion_validation_output.txt"
OUT_JSON = "dt_expansion_validation_results.json"
_log = open(OUT_TXT, "w", encoding="utf-8")
RESULTS: dict = {}


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _log.write(msg + "\n")
    _log.flush()


def save():
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)


def spec(S0, Hl):
    return OptionSpec(S0=S0, K=K, H=Hl, r=r, q=q, T=T, american=True, knock_in=False,
                       monitor_dates=None)


def fit_two_term(ns, prices):
    """Least-squares fit C(n) = C_inf + a/sqrt(n) + b/n on the finest
    len(ns) points, plus single-term comparisons. Returns dict."""
    ns = np.asarray(ns, dtype=float)
    y = np.asarray(prices, dtype=float)
    x1 = 1.0 / np.sqrt(ns)
    x2 = 1.0 / ns

    def rss(cols):
        Amat = np.column_stack([np.ones_like(ns)] + cols)
        coef, res, *_ = np.linalg.lstsq(Amat, y, rcond=None)
        fitted = Amat @ coef
        resid = y - fitted
        return coef, float(np.sqrt(np.mean(resid**2)))

    coef2, rmse2 = rss([x1, x2])      # two-term: C_inf, a, b
    coef_sqrt, rmse_sqrt = rss([x1])  # sqrt-only
    coef_lin, rmse_lin = rss([x2])    # dt-only
    return {
        "two_term": {"C_inf": coef2[0], "a": coef2[1], "b": coef2[2], "rmse": rmse2},
        "sqrt_only": {"C_inf": coef_sqrt[0], "a": coef_sqrt[1], "rmse": rmse_sqrt},
        "dt_only": {"C_inf": coef_lin[0], "b": coef_lin[1], "rmse": rmse_lin},
    }


def run_case(label, S0, Hl, levels=LEVELS):
    say(f"\n=== {label} (S0={S0}, H={Hl}, N={N_FINE}, M={M_FINE}) ===")
    s = spec(S0, Hl)
    prices, times = [], []
    for n in levels:
        t0 = time.perf_counter()
        res = cj.price_barrier_call(s, HP, NO_JUMPS, n_steps=n, N=N_FINE, M=M_FINE,
                                    extrapolation="none")
        dt_ = time.perf_counter() - t0
        prices.append(res.price)
        times.append(dt_)
        say(f"  n_steps={n:5d}  dt={T/n:.6f}  price={res.price:.6f}  ({dt_:.1f}s)")
    prices = np.array(prices)
    say("  successive differences and ratios (-> sqrt(2)=1.414 if dt^1/2 dominates, -> 2 if dt dominates):")
    diffs = np.abs(np.diff(prices))
    for i in range(len(diffs)):
        ratio = diffs[i] / diffs[i + 1] if i + 1 < len(diffs) else float("nan")
        say(f"    |C({levels[i]})-C({levels[i+1]})| = {diffs[i]:.6f}" +
            (f"   ratio to next = {ratio:.3f}" if i + 1 < len(diffs) else ""))
    fit = fit_two_term(levels, prices)
    say("  least-squares fits (finest extrapolated price C_inf, RMSE over all levels):")
    say(f"    two-term  (a/sqrt(n)+b/n): C_inf={fit['two_term']['C_inf']:.6f}  "
        f"a={fit['two_term']['a']:.5f}  b={fit['two_term']['b']:.5f}  RMSE={fit['two_term']['rmse']:.2e}")
    say(f"    sqrt-only (a/sqrt(n)):     C_inf={fit['sqrt_only']['C_inf']:.6f}  "
        f"a={fit['sqrt_only']['a']:.5f}  RMSE={fit['sqrt_only']['rmse']:.2e}")
    say(f"    dt-only   (b/n):           C_inf={fit['dt_only']['C_inf']:.6f}  "
        f"b={fit['dt_only']['b']:.5f}  RMSE={fit['dt_only']['rmse']:.2e}")
    RESULTS[label] = {"S0": S0, "H": Hl, "N": N_FINE, "M": M_FINE, "levels": levels,
                      "prices": prices.tolist(), "seconds": times, "fit": fit}
    save()


if __name__ == "__main__":
    run_case("base_am_cont_S100", 100.0, 130.0)
    run_case("tight_am_cont_S100_H105", 100.0, 105.0)
    say("\nDONE")
