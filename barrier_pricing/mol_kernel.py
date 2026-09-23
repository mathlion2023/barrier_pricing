"""Numba-JIT core of the method-of-lines time step: true Gauss-Seidel
sweeping across variance lines, in increasing m order, exactly as
Fortran's reference implementation does it (``Ame_cont.f90`` etc., loop
``do j=2,m1``, using the just-updated line j-1 as a neighbour).

This module exists because that Gauss-Seidel ordering is not just a
style choice: the alternative tried first here -- update every line at
once from the previous sweep's neighbours ("Jacobi"), which vectorizes
trivially across lines with plain NumPy -- has a mode (in V, wherever
the Riccati solution R(S) is near zero, so the price C=R*V+W stops
depending on V there) whose growth rate is a genuine >1 eigenvalue of
that iteration for realistic coefficient magnitudes here. No fixed
damping factor is safe across the range of grids/regimes this package
needs to support: small enough damping to survive the stiffest step
(late, fine-dtau, BDF2 regime) converges too slowly to be practical, and
any fixed damping factor eventually diverges given enough sweeps.
Gauss-Seidel does not have this failure mode (verified against an
independent shooting solution for the underlying ODE and by running
many more sweeps than any step here needs without drift), but sweeping
lines one at a time in pure Python/NumPy is dominated by per-call
overhead on the tiny per-line arrays. Numba compiles the whole
sweep -- both the m-loop and the inner S-index recursions -- to native
code, removing that overhead, which is what makes row-by-row Gauss-
Seidel fast enough to use here.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _riccati_row(S, A_row, B_row, R_row):
    n = S.shape[0]
    R_row[0] = 0.0
    for k in range(1, n):
        dy = S[k] - S[k - 1]
        c1 = A_row[k]
        c2 = 2.0 / dy + B_row[k]
        c3 = (B_row[k - 1] - 2.0 / dy) * R_row[k - 1] - 2.0 + A_row[k - 1] * R_row[k - 1] ** 2
        R_row[k] = -2.0 * c3 / (c2 + np.sqrt(c2 * c2 - 4.0 * c1 * c3))


@njit(cache=True)
def _forward_row(S, A_row, B_row, R_row, P_row, W_row):
    n = S.shape[0]
    W_row[0] = 0.0
    for k in range(1, n):
        dy = S[k] - S[k - 1]
        c1 = 1.0 / dy - A_row[k - 1] * R_row[k - 1] / 2.0
        c2 = 1.0 / dy + A_row[k] * R_row[k] / 2.0
        c3 = -(R_row[k] * P_row[k] + R_row[k - 1] * P_row[k - 1]) / 2.0
        W_row[k] = (c1 * W_row[k - 1] + c3) / c2


@njit(cache=True)
def _backward_row(S, A_row, B_row, R_row, W_row, P_row, start_idx, v_boundary, V_row):
    V_row[start_idx] = v_boundary
    for k in range(start_idx - 1, -1, -1):
        dy = S[k] - S[k + 1]
        c1 = 2.0 / dy + B_row[k + 1] + A_row[k + 1] * R_row[k + 1]
        c2 = 2.0 / dy - B_row[k] - A_row[k] * R_row[k]
        c3 = A_row[k] * W_row[k] + A_row[k + 1] * W_row[k + 1] + P_row[k + 1] + P_row[k]
        V_row[k] = (c1 * V_row[k + 1] + c3) / c2


@njit(cache=True)
def _free_boundary_row(S, R_row, W_row, K):
    """Returns (start_idx, v_boundary, boundary_S, has_boundary) for the
    American early-exercise search on one line: first sign change of
    phi(S)=R(S)+W(S)-(S-K) scanning down from the barrier (paper, Sec.
    3, "phi(S) will change sign at most once on [S0,H]").

    The crossing is located by a cubic Newton refinement through the
    four points bracketing it (Fortran's ``freboun`` does the same,
    citing better accuracy than the linear estimate alone -- the free
    boundary otherwise carries the full grid spacing as its location
    error, which then leaks into the exercised region's price/delta).
    """
    n = S.shape[0]
    prev_phi = R_row[n - 1] + W_row[n - 1] - (S[n - 1] - K)
    for k in range(n - 2, -1, -1):
        phi = R_row[k] + W_row[k] - (S[k] - K)
        if phi * prev_phi <= 0.0 and prev_phi != phi:
            w = prev_phi / (prev_phi - phi)
            b = S[k + 1] + w * (S[k] - S[k + 1])

            if k >= 1 and k + 2 <= n - 1:
                x0, x1, x2, x3 = S[k - 1], S[k], S[k + 1], S[k + 2]
                p0 = R_row[k - 1] + W_row[k - 1] - (x0 - K)
                p1 = phi
                p2 = prev_phi
                p3 = R_row[k + 2] + W_row[k + 2] - (x3 - K)
                c0 = p0 / ((x0 - x1) * (x0 - x2) * (x0 - x3))
                c1 = p1 / ((x1 - x0) * (x1 - x2) * (x1 - x3))
                c2 = p2 / ((x2 - x0) * (x2 - x1) * (x2 - x3))
                c3 = p3 / ((x3 - x0) * (x3 - x1) * (x3 - x2))
                r = b
                for _it in range(10):
                    f = (c0 * (r - x1) * (r - x2) * (r - x3)
                         + c1 * (r - x0) * (r - x2) * (r - x3)
                         + c2 * (r - x0) * (r - x1) * (r - x3)
                         + c3 * (r - x0) * (r - x1) * (r - x2))
                    fp = (c0 * ((r - x2) * (r - x3) + (r - x1) * (r - x3) + (r - x1) * (r - x2))
                          + c1 * ((r - x2) * (r - x3) + (r - x0) * (r - x3) + (r - x0) * (r - x2))
                          + c2 * ((r - x1) * (r - x3) + (r - x0) * (r - x3) + (r - x0) * (r - x1))
                          + c3 * ((r - x1) * (r - x2) + (r - x0) * (r - x2) + (r - x0) * (r - x1)))
                    if fp == 0.0:
                        break
                    rn = r - f / fp
                    converged = abs(rn - r) < 1e-8
                    r = rn
                    if converged:
                        break
                if x1 <= r <= x2:  # stay inside the bracket; else keep the linear estimate
                    b = r

            start_idx = k if b < S[k + 1] else k + 1
            return start_idx, 1.0, b, True
        prev_phi = phi
    return n - 1, 1.0, np.nan, False


@njit(cache=True)
def _interp_clamped(S, C_row, x):
    """Linear interpolation of one row at an arbitrary (possibly
    off-grid) spot ``x``, clamped flat outside [S[0], S[-1]]. Used for
    the jump-integral term, which needs C at S*Y for jump multipliers Y
    that may land near/outside the grid edges; flat extrapolation is
    the same simplification Fortran's jump handling makes for the
    (negligible, for reasonable jump parameters) probability mass
    beyond the grid.
    """
    n = S.shape[0]
    if x <= S[0]:
        return C_row[0]
    if x >= S[n - 1]:
        return C_row[n - 1]
    k = np.searchsorted(S, x) - 1
    if k >= n - 1:
        k = n - 2
    w = (x - S[k]) / (S[k + 1] - S[k])
    return C_row[k] * (1.0 - w) + C_row[k + 1] * w


@njit(cache=True)
def gauss_seidel_step(S, v, dv, A_over_S2, B_over_S, coefP, coefM, cross_coef,
                       prev_terms, C, V, R, recompute_R, K, H, american,
                       sor_tol, sor_max_iter, apply_barrier,
                       lam, jump_factor, jump_weight):
    """One full method-of-lines time step: Gauss-Seidel sweep across
    variance lines m=1..M until the price+delta residual falls below
    ``sor_tol`` or ``sor_max_iter`` sweeps are used. Row 0 (v=0) is
    filled by quadratic extrapolation, not solved.

    ``lam``, ``jump_factor``, ``jump_weight``: Merton/Bates log-normal
    jump intensity and precomputed Gauss-Hermite quadrature arrays (see
    ``mol.py``'s ``_jump_quadrature``) for the compensated-jump source
    term lam*(E[C(S*Y)] - C(S)); pass lam=0.0 to skip jumps entirely
    (Fortran's ``if(alam.ne.0)`` gate, ported the same way here). The
    "-lam*C(S)" half of that term is folded into A_m(S) already (see
    ``mol._line_coefficients``, the diag_C "+jp.lam" contribution); this
    kernel only needs to add the "+lam*E[C(S*Y)]" half, evaluated by
    interpolating the *current* sweep's own row profile at S*Y for each
    quadrature node Y -- a non-local term in S that the Riccati/sweep
    machinery cannot solve directly, so (as in Fortran) it is instead
    treated as a source frozen at the latest available iterate and
    refined across the same Gauss-Seidel sweeps used for the variance
    coupling, rather than a separate outer convergence loop.

    ``apply_barrier``: whether the S>=H wall is active *this* step. For
    continuous monitoring it always is (the whole domain is (0,H)). For
    discrete monitoring it must be False on every step that solves on
    the wide (0,S_max) domain (paper Sec. 2.2: no barrier constraint
    away from a monitoring instant) -- forcing C=0 there regardless
    would silently reintroduce continuous monitoring.

    ``C``, ``V`` are the incoming guess (typically the previous time
    step's converged values) and are overwritten in place with the
    result; ``R`` is overwritten with the Riccati solution only if
    ``recompute_R`` (grid or regime/dtau changed since the cached R).
    Returns the number of sweeps used and the boundary spot per line
    (NaN where American found none, unused for European).
    """
    M1, n = C.shape
    W = np.zeros((M1, n))
    P = np.zeros((M1, n))
    boundary_S = np.full(M1, np.nan)
    v_safe = np.where(v == 0.0, 1.0, v)

    A = np.empty((M1, n))
    B = np.empty((M1, n))
    for m in range(1, M1):
        for k in range(n):
            A[m, k] = A_over_S2[m] / (v_safe[m] * S[k] ** 2)
            B[m, k] = B_over_S[m] / (v_safe[m] * S[k])

    if recompute_R:
        for m in range(1, M1):
            _riccati_row(S, A[m], B[m], R[m])

    # Best-snapshot tracking: Gauss-Seidel across variance lines has no
    # proven convergence rate for this coupling (paper, footnote 7), and
    # for stiff regimes (many lines close to v=0 combined with a small
    # dtau -- both push A_m(S) up sharply there) sweeps have been
    # observed to converge cleanly for a while and only drift after --
    # in a component of V that the price residual (diff on C alone)
    # cannot see, since C=R*V+W is blind to V wherever R(S) is near
    # zero. Recording the best (lowest joint C+V change) sweep and
    # returning that instead of just the last one costs one extra pair
    # of array copies per sweep and makes the result robust to that
    # drift without having to know in advance how many sweeps are safe
    # for a given grid/parameter combination.
    best_diff = np.inf
    best_C = C.copy()
    best_V = V.copy()
    sweeps_used = sor_max_iter
    for sweep in range(sor_max_iter):
        max_diff = 0.0
        for m in range(1, M1):
            mp = m + 1 if m + 1 < M1 else m - 1  # Neumann mirror at v_max
            mm = m - 1
            for k in range(n):
                known = (cross_coef[m] * S[k] * (V[mp, k] - V[mm, k])
                         + coefP[m] * C[mp, k] + coefM[m] * C[mm, k]
                         + prev_terms[m, k])
                if lam != 0.0:
                    jump_val = 0.0
                    for l in range(jump_factor.shape[0]):
                        jump_val += jump_weight[l] * _interp_clamped(
                            S, C[m], S[k] * jump_factor[l])
                    known += lam * jump_val
                P[m, k] = -known / (v_safe[m] * S[k] ** 2 / 2.0)

            _forward_row(S, A[m], B[m], R[m], P[m], W[m])

            if american:
                start_idx, vb, b, found = _free_boundary_row(S, R[m], W[m], K)
                boundary_S[m] = b if found else np.nan
                if not found:
                    # No interior exercise boundary: Dirichlet value at the
                    # outer edge, imposed through C = R*V + W. At the barrier
                    # (continuous monitoring, or a monitoring step) this is
                    # C(H) = H-K (paper, Sec. 2.1: exercised at the wall
                    # rather than knocked out). On the wide domain it is
                    # immediate exercise, C(S_max) = S_max - K. (An earlier
                    # version imposed V = 1 here, which is a slope condition,
                    # not the stated Dirichlet condition.)
                    vb = (S[n - 1] - K - W[m, n - 1]) / R[m, n - 1]
            else:
                start_idx = n - 1
                # Dirichlet C(S[-1]) = 0 via C = R*V + W. At the barrier
                # this is the knock-out condition (eq. 9/15/35). On the wide
                # domain between monitoring dates, S_max = 2.2H lies far
                # above the barrier, where a knock-out call is (to within
                # the probability of falling back below H by the next
                # monitoring date, maturity included) worthless. Fortran's
                # Eur_disc.f90 likewise imposes a Dirichlet value there.
                # An earlier version used the vanilla far-field V = 1
                # (delta -> 1) on the wide domain, which is wrong for an
                # up-and-out call and made discrete-monitoring prices drift
                # away from the published values under grid refinement.
                vb = -W[m, n - 1] / R[m, n - 1]

            v_old = V[m].copy()
            _backward_row(S, A[m], B[m], R[m], W[m], P[m], start_idx, vb, V[m])

            for k in range(n):
                c_new = R[m, k] * V[m, k] + W[m, k]
                if apply_barrier and S[k] >= H:
                    # European: knocked out, worthless (eq. 9/15). American
                    # without an interior exercise boundary: exercised right
                    # at the wall rather than let it be knocked out (paper,
                    # Sec. 2.1, "if we cannot find b(v,tau)<H ... C(H,.)=H-K").
                    c_new = (S[k] - K) if american else 0.0
                d_c = abs(c_new - C[m, k])
                d_v = abs(V[m, k] - v_old[k])
                if d_c > max_diff:
                    max_diff = d_c
                if d_v > max_diff:
                    max_diff = d_v
                C[m, k] = c_new

            if american and not np.isnan(boundary_S[m]):
                for k in range(n):
                    if S[k] >= boundary_S[m]:
                        C[m, k] = S[k] - K
                        V[m, k] = 1.0

        # v=0 line: quadratic extrapolation through m=1,2,3 (Sec. 2 of the paper)
        for k in range(n):
            C[0, k] = C[3, k] - 3.0 * (C[2, k] - C[1, k])
            V[0, k] = V[3, k] - 3.0 * (V[2, k] - V[1, k])

        all_finite = True
        for m in range(M1):
            for k in range(n):
                if not (np.isfinite(C[m, k]) and np.isfinite(V[m, k])):
                    all_finite = False
                    break
            if not all_finite:
                break
        if not all_finite:
            sweeps_used = sweep + 1
            break

        if max_diff < best_diff:
            best_diff = max_diff
            best_C[:, :] = C
            best_V[:, :] = V

        if max_diff < sor_tol:
            sweeps_used = sweep + 1
            break

    # Stabilize V wherever C=R*V+W cannot meaningfully constrain it: where
    # |R(S)| is tiny, C is (locally) blind to V, so V there is carried
    # only by the recursion's own conditioning -- fine within one sweep,
    # but with no restoring force from the (well-behaved) price residual,
    # small noise introduced there (e.g. by interpolation when this
    # state is remeshed onto a different spot grid at a discrete
    # monitoring date) has nothing to damp it and compounds geometrically
    # across time steps/monitoring cycles. Patch just those points with a
    # central difference of the converged (and stable) price profile --
    # a numerical derivative, but confined to the handful of points the
    # ODE genuinely cannot pin down, not a wholesale fallback.
    R_floor = 0.001
    for m in range(1, M1):
        for k in range(n):
            if abs(R[m, k]) < R_floor:
                if 0 < k < n - 1:
                    best_V[m, k] = (best_C[m, k + 1] - best_C[m, k - 1]) / (S[k + 1] - S[k - 1])
                elif k == 0:
                    best_V[m, k] = (best_C[m, 1] - best_C[m, 0]) / (S[1] - S[0])
                else:
                    best_V[m, k] = (best_C[m, n - 1] - best_C[m, n - 2]) / (S[n - 1] - S[n - 2])

    C[:, :] = best_C
    V[:, :] = best_V
    return sweeps_used, boundary_S
