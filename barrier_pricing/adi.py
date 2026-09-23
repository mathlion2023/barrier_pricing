"""Independent finite-difference benchmark: Hundsdorfer-Verwer ADI for the
two-dimensional Heston PDE in (S, v), with Ikonen-Toivanen operator
splitting for American exercise.

References
    K. J. in 't Hout & S. Foulon (2010), "ADI finite difference schemes for
    option pricing in the Heston model with correlation", Int. J. Numer.
    Anal. Model. 7, 303-320.
    T. Haentjens & K. J. in 't Hout (2015), "ADI schemes for pricing
    American options under the Heston model", Appl. Math. Finance 22,
    207-237.
    S. Ikonen & J. Toivanen (2004), "Operator splitting methods for American
    option pricing", Appl. Math. Lett. 17, 809-814.

This module exists as a *third*, structurally independent method: it
shares no code with ``mol.py`` (line-by-line Riccati sweeps across variance
lines) or with the COS engines (spectral in log-price). It discretizes the
full 2D operator, mixed derivative included, on non-uniform grids with
second-order central differences. Exceptions: S-convection is
second-order upwind where the cell Peclet number exceeds 1 (always at
v = 0, where the S-direction is pure convection), and v-convection is
second-order upwind for v > 1. Jumps in the data at H (terminal payoff and
discrete monitoring dates) are cell-averaged at the node H, which keeps
the scheme second order in S (Pooley, Vetzal & Forsyth 2003); sampling the
jump one-sidedly makes it first order. Time integration is second order
away from the payoff, barrier and exercise non-smoothness. Its purpose is to provide
converged reference prices, in particular for Feller-violated,
high vol-of-vol and tight-barrier regimes.

PDE (tau = time to maturity, no jumps):
    u_tau = 1/2 S^2 v u_SS + rho sigma S v u_Sv + 1/2 sigma^2 v u_vv
            + (r-q) S u_S + kappa (theta - v) u_v - r u.

Grids. S: smooth non-uniform grid on [0, S_max] clustered at the strike
and at the barrier (via a cumulative mesh-density function). v: the
in 't Hout-Foulon sinh grid on [0, v_max], clustered near v = 0. The S grid
contains the barrier H as a node.

Boundary conditions.
* S = 0: u = 0.
* S = S_max: continuous monitoring (S_max = H): u = 0 (European) or
  u = H - K (American: the knock-out time is predictable for continuous
  paths, so the American up-and-out call equals the American call stopped
  at H paying (H-K)^+; this is also CKM's condition C(H) = H - K).
  Discrete monitoring or no barrier (S_max well above H): u = 0 for
  European up-and-out (the option is knocked out at the next monitoring
  date, and maturity is always a monitoring date), u = S_max - K for
  American (immediate exercise above the barrier between monitoring
  dates); vanilla: u = S_max e^{-q tau} - K e^{-r tau}.
* v = 0: no boundary condition. The PDE is imposed in its degenerate form
  u_tau = (r-q) S u_S + kappa theta u_v - r u, with a second-order one-sided
  (forward) difference for u_v. The drift kappa*theta > 0 points into the
  domain (Fichera), independent of the Feller condition.
* v = v_max: homogeneous Neumann u_v = 0 (ghost-point mirror for u_vv;
  convection and mixed terms vanish). ``v_max`` is large by default, and
  sensitivity to it is part of the benchmark checks.

Time stepping. Hundsdorfer-Verwer (theta = 1/2 + sqrt(3)/6), with the
mixed-derivative operator A0 explicit and A1 (S), A2 (v) implicit. After
maturity and after every discrete monitoring date the first
``n_damp`` steps use the Douglas scheme with theta = 1 (Rannacher-type
damping) to prevent the non-smooth data from exciting oscillations.
American exercise: Ikonen-Toivanen splitting (Lagrange multiplier lambda
carried between steps). Discrete monitoring dates must lie on the time grid.
"""
from __future__ import annotations

import time as _time

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import splu

from .params import HestonParams, JumpParams, NO_JUMPS, OptionSpec, PricingResult


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------

def _density_grid(lo: float, hi: float, n: int, centers, widths, weights,
                  must_include=()) -> np.ndarray:
    """Smooth non-uniform grid with n+1 points on [lo, hi], point density
    proportional to 1 + sum_j w_j / sqrt(1 + ((x - c_j)/d_j)^2), with the
    points listed in ``must_include`` moved exactly onto the grid (nearest
    node)."""
    xf = np.linspace(lo, hi, 200001)
    dens = np.ones_like(xf)
    for c, d, w in zip(centers, widths, weights):
        dens += w / np.sqrt(1.0 + ((xf - c) / d) ** 2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xf))])
    cdf /= cdf[-1]
    # Put the required points exactly on nodes without distorting local
    # spacing: map the uniform computational coordinate piecewise-linearly
    # so that each required point sits at the grid index nearest to its
    # own cdf value (the density, and hence the spacing, stays smooth).
    req = sorted(p for p in must_include if lo < p < hi)
    knots_c = [0.0] + [float(np.interp(p, xf, cdf)) for p in req] + [1.0]
    knots_i = [0] + [int(round(c * n)) for c in knots_c[1:-1]] + [n]
    xi = np.interp(np.arange(n + 1), knots_i, knots_c)
    x = np.interp(xi, cdf, xf)
    x[0], x[-1] = lo, hi
    for p, i in zip(req, knots_i[1:-1]):
        x[i] = p
    return x


def _v_grid(v_max: float, m: int, d_frac: float = 1.0 / 500.0) -> np.ndarray:
    d = v_max * d_frac
    eta = np.linspace(0.0, np.arcsinh(v_max / d), m + 1)
    v = d * np.sinh(eta)
    v[0], v[-1] = 0.0, v_max
    return v


# --------------------------------------------------------------------------
# 1D finite-difference stencils on non-uniform grids
# --------------------------------------------------------------------------

def _central_weights(x: np.ndarray):
    """(beta_-1, beta_0, beta_1) for d/dx and (delta_-1, delta_0, delta_1)
    for d2/dx2 at interior nodes (arrays of length n+1, zero at ends)."""
    n1 = len(x)
    bm, b0, bp = (np.zeros(n1) for _ in range(3))
    dm, d0, dp = (np.zeros(n1) for _ in range(3))
    h = np.diff(x)
    hi, hip = h[:-1], h[1:]
    bm[1:-1] = -hip / (hi * (hi + hip))
    b0[1:-1] = (hip - hi) / (hi * hip)
    bp[1:-1] = hi / (hip * (hi + hip))
    dm[1:-1] = 2.0 / (hi * (hi + hip))
    d0[1:-1] = -2.0 / (hi * hip)
    dp[1:-1] = 2.0 / (hip * (hi + hip))
    return (bm, b0, bp), (dm, d0, dp)


def _build_operators(S: np.ndarray, v: np.ndarray, r: float, q: float, hp: HestonParams,
                     dirichlet_top: bool):
    """Sparse A0 (mixed), A1 (S-direction incl. -r/2), A2 (v-direction incl.
    -r/2) on the full node set, unknown index = i + (nS+1)*j (S fastest).
    Rows of Dirichlet nodes (S=0, and S=S_max) are zero, so those values
    stay equal to the initial/boundary data throughout.
    """
    nS1, nv1 = len(S), len(v)
    idx = lambda i, j: i + nS1 * j
    (sbm, sb0, sbp), (sdm, sd0, sdp) = _central_weights(S)
    (vbm, vb0, vbp), (vdm, vd0, vdp) = _central_weights(v)
    kappa, theta, sigma, rho = hp.kappa, hp.theta, hp.sigma, hp.rho

    rows0, cols0, vals0 = [], [], []
    rows1, cols1, vals1 = [], [], []
    rows2, cols2, vals2 = [], [], []

    i_int = np.arange(1, nS1 - 1)
    for j in range(nv1):
        vj = v[j]
        # ---- A1: S-direction, interior S nodes. Central convection where
        # the cell Peclet number |a1| h / (2 a2) <= 1, second-order
        # (three-point) upwind otherwise -- always at v = 0, where the
        # S-direction is pure convection: central differences there are
        # non-dissipative, and with the jump at H in discrete-monitoring data
        # they converged to a visibly wrong limit (0.7% at H = 130; checked
        # against the closed-form maturity-only-monitored price).
        a2 = 0.5 * S[i_int] ** 2 * vj
        a1 = (r - q) * S[i_int]
        hS = np.maximum(S[i_int + 1] - S[i_int], S[i_int] - S[i_int - 1])
        central = np.abs(a1) * hS <= 2.0 * a2
        coef = {o: np.zeros(len(i_int)) for o in (-2, -1, 0, 1, 2)}
        for off, cd, cb in ((-1, sdm, sbm), (0, sd0, sb0), (1, sdp, sbp)):
            coef[off] += a2 * cd[i_int] + np.where(central, a1 * cb[i_int], 0.0)
        coef[0] += -0.5 * r
        ii = i_int
        # backward (a1 < 0) upwind: nodes i-2, i-1, i (first order at i = 1)
        bk = ~central & (a1 < 0)
        two = bk & (ii >= 2)
        one = bk & (ii < 2)
        h1 = np.where(ii >= 2, S[np.maximum(ii - 1, 0)] - S[np.maximum(ii - 2, 0)], 1.0)
        h2 = S[ii] - S[ii - 1]
        coef[-2] += np.where(two, a1 * h2 / (h1 * (h1 + h2)), 0.0)
        coef[-1] += np.where(two, -a1 * (h1 + h2) / (h1 * h2), 0.0) + np.where(one, -a1 / h2, 0.0)
        coef[0] += np.where(two, a1 * (h1 + 2 * h2) / (h2 * (h1 + h2)), 0.0) + np.where(one, a1 / h2, 0.0)
        # forward (a1 >= 0) upwind: nodes i, i+1, i+2 (first order at i = n-1)
        fw = ~central & (a1 >= 0)
        twof = fw & (ii <= nS1 - 3)
        onef = fw & (ii > nS1 - 3)
        g1 = S[ii + 1] - S[ii]
        g2 = np.where(ii <= nS1 - 3, S[np.minimum(ii + 2, nS1 - 1)] - S[ii + 1], 1.0)
        coef[0] += np.where(twof, -a1 * (2 * g1 + g2) / (g1 * (g1 + g2)), 0.0) + np.where(onef, -a1 / g1, 0.0)
        coef[1] += np.where(twof, a1 * (g1 + g2) / (g1 * g2), 0.0) + np.where(onef, a1 / g1, 0.0)
        coef[2] += np.where(twof, -a1 * g1 / (g2 * (g1 + g2)), 0.0)
        for off, c in coef.items():
            ok = (ii + off >= 0) & (ii + off <= nS1 - 1) & (c != 0.0)
            rows1.append(idx(ii[ok], j)); cols1.append(idx(ii[ok] + off, j)); vals1.append(c[ok])
        # ---- A2: v-direction
        rr = idx(i_int, j)
        if j == 0:
            # forward second-order one-sided convection; diffusion vanishes
            h1, h2 = v[1] - v[0], v[2] - v[1]
            al0 = (-2 * h1 - h2) / (h1 * (h1 + h2))
            al1 = (h1 + h2) / (h1 * h2)
            al2 = -h1 / (h2 * (h1 + h2))
            c = kappa * theta
            for jj, w in ((0, c * al0 - 0.5 * r), (1, c * al1), (2, c * al2)):
                rows2.append(rr); cols2.append(idx(i_int, jj)); vals2.append(np.full(len(i_int), w))
        elif j == nv1 - 1:
            # Neumann u_v = 0: ghost mirror for u_vv, no convection
            hm = v[j] - v[j - 1]
            dif = 0.5 * sigma**2 * vj
            rows2.append(rr); cols2.append(idx(i_int, j - 1)); vals2.append(np.full(len(i_int), 2 * dif / hm**2))
            rows2.append(rr); cols2.append(idx(i_int, j)); vals2.append(np.full(len(i_int), -2 * dif / hm**2 - 0.5 * r))
        else:
            dif = 0.5 * sigma**2 * vj
            drift = kappa * (theta - vj)
            if vj > 1.0 and j >= 2:
                # upwind (backward, second order) convection for large v
                h1, h2 = v[j - 1] - v[j - 2], v[j] - v[j - 1]
                g2 = h2 / (h1 * (h1 + h2))
                g1 = -(h1 + h2) / (h1 * h2)
                g0 = (h1 + 2 * h2) / (h2 * (h1 + h2))
                conv = {-2: drift * g2, -1: drift * g1, 0: drift * g0}
            else:
                conv = {-1: drift * vbm[j], 0: drift * vb0[j], 1: drift * vbp[j]}
            diff = {-1: dif * vdm[j], 0: dif * vd0[j], 1: dif * vdp[j]}
            for off in sorted(set(conv) | set(diff)):
                w = conv.get(off, 0.0) + diff.get(off, 0.0) + (-0.5 * r if off == 0 else 0.0)
                rows2.append(rr); cols2.append(idx(i_int, j + off)); vals2.append(np.full(len(i_int), w))
        # ---- A0: mixed derivative, interior v nodes only (v=0: factor v; v_max: u_v=0)
        if 0 < j < nv1 - 1:
            c = rho * sigma * S[i_int] * vj
            for oi, wi in ((-1, sbm), (0, sb0), (1, sbp)):
                for oj, wj in ((-1, vbm[j]), (0, vb0[j]), (1, vbp[j])):
                    rows0.append(rr); cols0.append(idx(i_int + oi, j + oj))
                    vals0.append(c * wi[i_int] * wj)

    n = nS1 * nv1
    mk = lambda R, C, V: sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, n))
    return mk(rows0, cols0, vals0), mk(rows1, cols1, vals1), mk(rows2, cols2, vals2)


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------

def price_barrier_call(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                        nS: int = 200, nv: int = 100, n_steps: int = 400,
                        v_max: float | None = None, S_max_mult: float = 3.0,
                        n_damp: int = 2, spots=None, no_barrier: bool = False,
                        return_grid: bool = False) -> PricingResult:
    """Price an up-and-out call (European or American; continuous or
    discrete monitoring) with HV-ADI (+ Ikonen-Toivanen for American).

    ``no_barrier=True`` prices the vanilla call (European or American) on
    [0, S_max_mult*K*...] instead, for validation against closed forms.
    """
    if jp.active:
        raise NotImplementedError("adi.py is a diffusion-only (Heston) benchmark")
    t0 = _time.perf_counter()
    K, H, T, r, q = spec.K, spec.H, spec.T, spec.r, spec.q
    v_max = v_max if v_max is not None else max(3.0, 15.0 * hp.theta, 15.0 * hp.v0)
    continuous = spec.is_continuous() and not no_barrier

    if continuous:
        S_max = H
        S = _density_grid(0.0, S_max, nS, centers=(K, H), widths=(0.1 * K, 0.05 * K),
                          weights=(8.0, 6.0), must_include=(K,))
    else:
        S_max = S_max_mult * (H if not no_barrier else 2.0 * K)
        S = _density_grid(0.0, S_max, nS, centers=(K, H) if not no_barrier else (K,),
                          widths=(0.1 * K, 0.05 * K), weights=(8.0, 6.0),
                          must_include=(K, H) if not no_barrier else (K,))
    v = _v_grid(v_max, nv)
    nS1, nv1 = len(S), len(v)
    # Discontinuous data at H (terminal payoff and every discrete monitoring
    # date): the node at H carries the average of the step over its dual
    # cell, w_H * (left limit), w_H = h_-/(h_- + h_+). Sampling the jump
    # one-sidedly instead (value 0 at H) makes the scheme only first order
    # in S (Pooley, Vetzal & Forsyth 2003, J. Comput. Finance 6(4)).
    if not continuous and not no_barrier:
        iH = int(np.argmin(np.abs(S - H)))
        wH = (S[iH] - S[iH - 1]) / (S[iH + 1] - S[iH - 1])
    else:
        iH, wH = None, 0.0
    SS = np.repeat(S[:, None], nv1, axis=1)  # (nS1, nv1)

    A0, A1, A2 = _build_operators(S, v, r, q, hp, dirichlet_top=True)
    A = (A0 + A1 + A2).tocsr()
    I = sp.identity(nS1 * nv1, format="csc")

    dt = T / n_steps
    th_hv = 0.5 + np.sqrt(3.0) / 6.0
    lu = {}

    def solver(Aj, th, key):
        if key not in lu:
            lu[key] = splu((I - th * dt * Aj).tocsc())
        return lu[key]

    payoff = np.maximum(SS - K, 0.0)
    if continuous:
        top = (H - K) if spec.american else 0.0
    elif no_barrier:
        top = None  # time-dependent vanilla far field, set every step
    else:
        top = (S_max - K) if spec.american else 0.0

    # terminal condition (maturity is always a monitoring date)
    U = payoff.copy()
    if not no_barrier:
        U[S >= H, :] = 0.0
        if not continuous:
            U[iH, :] = wH * max(H - K, 0.0)  # cell average of the jump at H
        if continuous:
            U[-1, :] = top
    U = U.ravel(order="F")
    phi = payoff.ravel(order="F")
    lam = np.zeros_like(U)

    if continuous or no_barrier:
        monitor_steps = set()
    else:
        cal = np.asarray(sorted(T - np.asarray(spec.monitor_dates)))
        idx = np.round(cal / dt).astype(int)
        if not np.allclose(idx * dt, cal, atol=1e-9 * max(1.0, T)):
            raise ValueError("monitoring dates must be multiples of T/n_steps")
        # calendar index k <-> tau = T - k dt <-> backward step n_steps - k
        monitor_steps = {n_steps - k for k in idx if 0 < k < n_steps}

    above_H = np.repeat((S >= H)[:, None], nv1, axis=1).ravel(order="F")
    H_rows = np.arange(nv1) * nS1 + iH if iH is not None else None
    top_rows = np.arange(nv1) * nS1 + (nS1 - 1)

    damp_left = n_damp
    for step in range(1, n_steps + 1):
        tau = step * dt
        if no_barrier:
            if spec.american:
                U[top_rows] = S_max - K
            else:
                U[top_rows] = S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
        extra = lam if spec.american else 0.0
        if damp_left > 0:
            th = 1.0
            Y0 = U + dt * (A @ U + extra)
            Y1 = solver(A1, th, ("1", th)).solve(Y0 - th * dt * (A1 @ U))
            Y2 = solver(A2, th, ("2", th)).solve(Y1 - th * dt * (A2 @ U))
            Unew = Y2
            damp_left -= 1
        else:
            th = th_hv
            AU = A @ U
            Y0 = U + dt * (AU + extra)
            Y1 = solver(A1, th, ("1", th)).solve(Y0 - th * dt * (A1 @ U))
            Y2 = solver(A2, th, ("2", th)).solve(Y1 - th * dt * (A2 @ U))
            Yt0 = Y0 + 0.5 * dt * (A @ Y2 - AU)
            Yt1 = solver(A1, th, ("1", th)).solve(Yt0 - th * dt * (A1 @ Y2))
            Unew = solver(A2, th, ("2", th)).solve(Yt1 - th * dt * (A2 @ Y2))
        if spec.american:
            Ubar = np.maximum(Unew - dt * lam, phi)
            lam = np.maximum(0.0, lam + (Ubar - Unew) / dt)  # = lam + (phi - Unew)/dt where active
            lam = np.where(Ubar > phi, 0.0, lam)
            Unew = Ubar
        if step in monitor_steps:
            left_limit = Unew[H_rows].copy()
            Unew = np.where(above_H, 0.0, Unew)
            Unew[H_rows] = wH * left_limit  # cell average of the jump at H
            lam = np.where(above_H, 0.0, lam)
            if spec.american:
                # exercise is continuous, so just before the monitoring
                # instant the holder compares the knocked-out continuation
                # with the payoff: V(t_i-) = max(1{S<H} V(t_i+), payoff).
                # Applying the knock-out after the exercise step instead
                # leaves S >= H at zero for a whole step (an O(dt^(1/2))-type
                # lag that dominated the time error).
                Unew = np.maximum(Unew, phi)
            damp_left = n_damp
        U = Unew

    Ugrid = U.reshape((nS1, nv1), order="F")
    spot_arr = np.atleast_1d(np.asarray([spec.S0] + ([] if spots is None else list(spots)), float))
    spline = RectBivariateSpline(S, v, Ugrid, kx=3, ky=3)
    prices = np.array([spline(s, hp.v0)[0, 0] for s in spot_arr])
    deltas = np.array([spline(s, hp.v0, dx=1)[0, 0] for s in spot_arr])
    gammas = np.array([spline(s, hp.v0, dx=2)[0, 0] for s in spot_arr])
    if continuous:
        dead = spot_arr >= H
        prices[dead] = 0.0; deltas[dead] = 0.0; gammas[dead] = 0.0

    extra = {"spots": spot_arr, "prices": prices, "deltas": deltas, "gammas": gammas,
             "nS": nS, "nv": nv, "n_steps": n_steps, "v_max": v_max, "S_max": S_max}
    if return_grid:
        extra.update({"S": S, "v": v, "U": Ugrid,
                      "lam": lam.reshape((nS1, nv1), order="F") if spec.american else None})
    return PricingResult(price=float(prices[0]), delta=float(deltas[0]), gamma=float(gammas[0]),
                         runtime=_time.perf_counter() - t0, extra=extra)


def exercise_boundary(result: PricingResult, K: float, H: float, tol: float = 1e-8):
    """Early-exercise boundary S*(v) at t = 0 from an American run made with
    ``return_grid=True``: for each variance node, the lowest spot in (K, H)
    at which the Ikonen-Toivanen Lagrange multiplier ``lam`` (``extra["lam"]``)
    turns from zero (continuation) to positive (exercise active).

    ``U - intrinsic`` is not used here: under continuous monitoring
    ``U(H) = H-K`` exactly (Lemma~\\ref{lem:stopped}), so ``U - intrinsic``
    is forced to 0 near ``H`` by the barrier condition alone, whether or not
    early exercise is optimal there. ``lam`` is the LCP complementarity
    multiplier and is free of this confound: it is exactly zero wherever the
    continuation value is used (``Ubar > phi``) and positive only where the
    exercise constraint binds.
    """
    S, v, lam = result.extra["S"], result.extra["v"], result.extra["lam"]
    out = np.full(len(v), np.nan)
    live = np.where((S > K) & (S < H))[0]
    for j in range(len(v)):
        a = lam[live, j]
        active = np.where(a > tol)[0]
        if len(active) == 0 or active[0] == 0:
            continue  # no exercise region below H, or exercised at K already
        k = active[0] - 1
        s1, s2 = S[live[k]], S[live[k + 1]]
        a1, a2 = a[k], a[k + 1]
        out[j] = s1 + (tol - a1) * (s2 - s1) / (a2 - a1) if a2 != a1 else s2
    return v, out
