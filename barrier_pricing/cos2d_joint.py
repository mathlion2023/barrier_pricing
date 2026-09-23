"""Joint (X,v) COS / finite-difference recursion for American and European
up-and-out barrier calls under Heston stochastic volatility, optionally with
Merton/Bates log-normal jumps in the log-price.

This is the package's principal COS-type engine. Unlike ``cos2d.py`` /
``cos2d_american.py`` (frozen-variance-branch surrogates that never mix
variance between exercise/monitoring dates), it discretizes the *joint*
Heston transition at every time step and converges to the Heston price as
N, M and the number of time steps grow.

Scheme
------
Write the value on the truncated log-price interval [a, b] (x = ln(S/K)) as

    C(x, v, t) = sum'_k Re[ W_k(v, t) exp(i u_k (x - a)) ],   u_k = k pi/(b-a).

Substituting into the Heston(+jump) PDE diagonalizes every x-derivative,
including the mixed derivative, so each frequency k solves its own
one-dimensional problem in v (tau = time to maturity):

    dW_k/dtau = (sigma^2 v/2) W_k'' + [kappa(theta-v) + i u_k rho sigma v] W_k'
                + R_k(v) W_k,
    R_k(v) = -v u_k^2/2 + i u_k (r - q - lam*kbar - v/2) - r + lam (phiJ(u_k) - 1),

with phiJ the jump-size characteristic function and kbar = E[e^J]-1
(JumpParams convention: J ~ N(gam, del^2), see params.py). Jumps only add a
diagonal term per mode.

Variance discretization (``v_grid="sinh"``: non-uniform grid clustered at
v = 0, or ``"uniform"``; today's v0 exactly on a node in both cases):

* interior rows: central second difference for the diffusion; central
  first difference for the complex cross-drift i u rho sigma v; the real
  drift kappa(theta-v) is central where the cell Peclet number
  |kappa(theta-v)| dv / (sigma^2 v) <= 1 and upwind otherwise, so the
  real part of the operator stays an M-matrix;
* row v = 0: the PDE's own degenerate limit, which needs no boundary
  condition because the drift kappa*theta > 0 points into the domain
  (Fichera). This holds whether or not the Feller condition is met:
  dW/dtau = kappa theta (W_1 - W_0)/dv + R_k(0) W_0;
* row v = v_max: ``vmax_bc="outflow"`` (default) keeps the first-order
  terms, one-sided and upwind (the drift at v_max is negative, i.e. into
  the domain), and drops the diffusion. ``vmax_bc="neumann"`` imposes
  dW/dv = 0 instead. ``stability_diagnostics`` and the convergence scripts
  measure how sensitive prices are to this choice and to v_max.

Time: ``time_integrator="rbe"`` (default) is Richardson-extrapolated
backward Euler, 2 BE(dt/2)^2 - BE(dt): second order, stability function
R(z) = 2/(1+z/2)^2 - 1/(1+z) with |R| <= 1 on Re z >= 0 and R(inf) = 0
(A- and L-stable), three batched tridiagonal solves per step. ``"be"`` is
plain backward Euler. It is first order, and with the kink re-created by
the exercise step at every date its error is large enough to contaminate
the observed order. ``"expm"`` propagates each mode exactly with
exp(dt L_k): exact in time for the v-semi-discrete system, but it costs one
dense matrix exponential per mode. It is used here to verify that "rbe" has
no visible time-integration error.

Stability. Freeze the coefficients and take the Fourier symbol in v
(theta_v = frequency in the v-direction, s = sin(theta_v)/dv,
t = 2 sin(theta_v/2)/dv, so |s| <= t). With central differencing of the
cross term, the real part of the interior symbol is

    -r - (v/2) [u^2 + 2 u rho sigma s + sigma^2 t^2] + Re lam(phiJ - 1) - (upwind part)
    <= -r - (v/2)(1 - rho^2) u^2 <= -r,

because u^2 + 2u rho sigma s + sigma^2 s^2 >= (1-rho^2) u^2 and t^2 >= s^2.
So every mode is von Neumann stable under backward Euler for any dt, with
amplification <= 1/(1 + r dt). This is a frozen-coefficient argument.
``stability_diagnostics`` computes the true matrix quantities over all k,
including the boundary rows. Measured (joint_convergence_study.py,
section A): the spectral abscissa of every L_k equals -r in all tested
regimes (base, Feller-violated, high vol-of-vol, rho = -0.9), consistent
with the symbol bound. However, the one-step 2-norms
||(I - dt L_k)^{-1}||_2 and ||exp(dt L_k)||_2 are slightly above 1, about
1 + c dt with c ~ 2-5. The operator is non-normal (variable coefficients,
non-uniform grid), so it is not a Euclidean contraction; the scheme is
Lax-stable with ||products of n steps|| <= exp(c T), uniformly in dt. The earlier version of this module upwinded the cross term
together with the physical drift. That scheme has the same real symbol (so
the same stability) but is only first-order consistent in the cross term.

Between time steps the value is reconstructed on an oversampled midpoint
grid x_i = a + (i + 1/2)(b-a)/n_eval, n_eval = ``oversample``*N, with a
zero-padded DCT-III/DST-III (FFT, O(n_eval log n_eval) per variance node).
The exercise and barrier conditions are applied pointwise there, and the
first N cosine coefficients are recovered with a DCT-II, i.e. the midpoint
rule for the L2 projection. With oversample = 1 this reduces to DCT
interpolation. Interpolation aliases the kink at the exercise boundary and
the jump at the barrier into low, weakly damped modes at every step, so the
error builds up like n_steps * N^-2. Oversampling removes most of that
(see ``_reproject``). The truncation interval is shifted so that the barrier
h = ln(H/K) lies exactly on a (fine and coarse) cell boundary.

Barrier treatment (``barrier_method``)
---------------------------------------
* ``"zero"``: value set to 0 for x >= h at every monitoring instant.
  Applied at every time step, this is discrete monitoring with period dt,
  and its error relative to continuous monitoring is O(sqrt(dt))
  (Broadie-Glasserman-Kou), not O(dt).
* ``"bgk"`` (European, continuous monitoring): the same, but with the
  continuity-corrected barrier h - beta sqrt(v dt) per variance node,
  beta = -zeta(1/2)/sqrt(2 pi) ~ 0.5826, which removes the O(sqrt(dt)) term.
* ``"cap"`` (American, continuous monitoring, no jumps): with continuous
  paths the knock-out time tau_H is predictable, so exercising just before
  tau_H earns (H-K)^+ in the limit. The American up-and-out call therefore
  equals the American call on S stopped at H, paying (H-K)^+ at tau_H. The
  value for x >= h is set to e^h - 1 (unit-K scale), which makes the value
  function continuous across the barrier: no Gibbs jump, and the barrier no
  longer contributes an O(sqrt(dt)) error. This is also why Chiarella,
  Kang & Meyer impose C(H) = H - K. With jumps the process can jump over H
  and be knocked out, so the identity fails and ``"zero"`` is used.
* discrete monitoring: ``"zero"`` at exactly the monitoring dates (which
  must lie on the time grid).

Error expansion and extrapolation (``extrapolation``)
-----------------------------------------------------
With "rbe" the time-integration error is O(dt^2), so the leading errors in
dt come from the two discrete-time events:

* continuous barrier monitoring (European, ``"zero"``/``"bgk"``) and the
  stopping at H in the capped American formulation (``"cap"``): the value
  is checked against H only at the time nodes. This is a discretely
  monitored barrier, and its error expands as c1 dt^(1/2) + c2 dt + ...
  (Broadie-Glasserman-Kou). The American case keeps the dt^(1/2) term
  because, for spots near H, the unconstrained exercise boundary lies above
  H, so the holder exercises only on reaching H: smooth fit fails at H, and
  the constant-volatility limit (sigma -> 0) shows the same dt^(1/2)
  behaviour, so it is a property of the contract, not of Heston;
* Bermudan exercise at every node: O(dt) (smooth fit at an interior
  exercise boundary).

``"sqrt"`` (default for continuous monitoring) uses levels n, 2n, 4n, 8n,
eliminates the dt^(1/2) and dt terms from the finest three, and reports as
error estimate the change against the same extrapolant from the coarsest
three. ``"adaptive"`` (default for discrete monitoring) uses n, 2n, 4n with
the observed order p = log2(|C_n - C_2n|/|C_2n - C_4n|), and falls back to
the finest level when the sequence is not monotone or p is outside
[0.3, 3]. ``"p1"`` (2 C_2n - C_n) is the legacy fixed-order extrapolant, and
``"none"`` returns a single level.

Resolution requirements (measured; see ``joint_convergence_study.py``)
----------------------------------------------------------------------
* Projection ratchet. Each exercise date evaluates the truncated series,
  takes max(., payoff) and reprojects. The max clips the Gibbs undershoot of
  the truncated series and keeps the overshoot, so every date adds a
  positive error of the size of the local truncation error. Over n dates
  this adds up to n * err_N, visible as prices that rise linearly in n at
  fixed N. The one-step transition damps it only where the one-step
  diffusion length sqrt(v dt) is large compared with the collocation
  spacing dx = (b-a)/N. In practice N must grow like sqrt(n): with the
  base parameters, N = 2048 removes it for up to ~1600 steps (N <= 1024
  does not).
* Variance grid. dv = 0.01 (M = 100 on [0, 1]) gives about 1e-4 relative
  accuracy for the base parameters. dv = 0.02 leaves a 0.1% error in the
  European continuous case.
"""
from __future__ import annotations

import time as _time

import numba
import numpy as np
from scipy import fft as _fft

from .charfunc import _svjd_cumulants
from .params import HestonParams, JumpParams, NO_JUMPS, OptionSpec, PricingResult

BGK_BETA = 0.5825971579390106  # -zeta(1/2)/sqrt(2*pi)


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

def _variance_grid(v0: float, v_max: float, M: int, kind: str = "sinh",
                   d_frac: float = 0.02) -> tuple[np.ndarray, float, int]:
    """Variance grid 0 = v_0 < ... < v_M = v_max with today's v0 exactly on
    node m0. Returns (grid, smallest spacing, m0).

    ``kind="uniform"``: dv = v0/m0 with m0 = round(v0*M/v_max) (>= 1), so
    the grid is uniform all the way down to v = 0.
    ``kind="sinh"`` (default): v_j = d sinh(eta_j) with d = d_frac*v_max
    (in 't Hout & Foulon 2010), which clusters nodes near v = 0, where the
    coefficients degenerate and, when the Feller condition fails, where
    much of the probability mass lies. The computational coordinate eta is
    remapped piecewise-linearly so that v0 falls exactly on a node without
    creating a spacing jump.
    """
    if kind == "uniform":
        m0 = max(1, int(round(v0 * M / v_max)))
        dv = v0 / m0
        n_int = max(m0 + 2, int(np.ceil(v_max / dv - 1e-12)))
        grid = dv * np.arange(n_int + 1)
        return grid, dv, m0
    if kind != "sinh":
        raise ValueError("v_grid must be 'uniform' or 'sinh'")
    d = d_frac * v_max
    eta_max = np.arcsinh(v_max / d)
    eta0 = np.arcsinh(v0 / d)
    m0 = min(max(1, int(round(eta0 / eta_max * M))), M - 2)
    eta = np.concatenate([np.linspace(0.0, eta0, m0 + 1)[:-1], np.linspace(eta0, eta_max, M - m0 + 1)])
    grid = d * np.sinh(eta)
    grid[0], grid[m0], grid[-1] = 0.0, v0, v_max
    return grid, float(np.min(np.diff(grid))), m0


def _x_domain(spec: OptionSpec, hp: HestonParams, jp: JumpParams, spots: np.ndarray,
              N: int, L: float, dt_max: float, v_top: float) -> tuple[float, float]:
    """Truncation interval [a, b], shifted so that h = ln(H/K) is a
    collocation-cell boundary.

    Lower end: L standard deviations of ln(S_T/S0) below the lowest spot.
    Upper end:
      * continuous monitoring: h plus a margin covering one time step of
        the fastest variance node (and the jump size), which is all the
        recursion needs because the value above h is known (0 or the
        capped value) at every step;
      * discrete monitoring or no effective barrier: L standard deviations
        above the highest spot, as for a vanilla, but at least that same
        one-step margin above h.
    """
    c1, c2, c4 = _svjd_cumulants(spec.T, spec.r, spec.q, hp, jp)
    sd = np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    x_lo = np.log(spots.min() / spec.K)
    x_hi = np.log(spots.max() / spec.K)
    a0 = x_lo + c1 - L * sd
    h = np.log(spec.H / spec.K)
    jump_margin = (abs(jp.gam) + L * jp.del_) if jp.active else 0.0
    step_margin = L * np.sqrt(v_top * dt_max) + jump_margin
    vanilla_top = x_hi + c1 + L * sd
    if spec.is_continuous():
        b0 = min(vanilla_top, h + step_margin) if h < vanilla_top else vanilla_top
        b0 = max(b0, x_hi + step_margin)
    else:
        b0 = max(vanilla_top, h + step_margin)
    dx = (b0 - a0) / N
    if h < b0:
        # shift a so that h = a + j*dx exactly (cell boundary)
        j = int(np.ceil((h - a0) / dx))
        a = h - j * dx
    else:
        a = a0
    return a, a + N * dx


# --------------------------------------------------------------------------
# Per-mode variance operator and its tridiagonal factorization
# --------------------------------------------------------------------------

def _mode_operator(u: np.ndarray, v: np.ndarray, dv, r: float, q: float,
                   hp: HestonParams, jp: JumpParams, vmax_bc: str):
    """Tridiagonal L_k for every mode on the (possibly non-uniform) grid v:
    returns (lower, diag, upper), each (N, M1) complex, such that
    (L_k W)_i = lower_i W_{i-1} + diag_i W_i + upper_i W_{i+1}.

    Interior rows use the standard three-point non-uniform stencils (second
    order): with hm = v_i - v_{i-1}, hp = v_{i+1} - v_i,
      W''  ~ 2 [W_{i-1}/(hm(hm+hp)) - W_i/(hm hp) + W_{i+1}/(hp(hm+hp))],
      W'   ~ -hp/(hm(hm+hp)) W_{i-1} + (hp-hm)/(hm hp) W_i + hm/(hp(hm+hp)) W_{i+1}.
    The complex cross drift always uses the central W' stencil. The real
    drift kappa(theta-v) uses it where the cell Peclet number
    |kappa(theta-v)| max(hm,hp) / (sigma^2 v) <= 1, and first-order upwind
    otherwise. ``dv`` is unused (kept for signature compatibility).
    """
    N, M1 = len(u), len(v)
    D = 0.5 * hp.sigma**2 * v                     # (M1,)
    mu = hp.kappa * (hp.theta - v)               # (M1,) real drift
    iu = 1j * u[:, None]                         # (N, 1)
    cross = iu * hp.rho * hp.sigma * v[None, :]  # (N, M1) complex drift
    kbar = jp.ak1 if jp.active else 0.0
    R = (-0.5 * v[None, :] * u[:, None] ** 2
         + iu * (r - q - jp.lam * kbar - 0.5 * v[None, :]) - r) * np.ones((1, M1))
    if jp.active:
        phiJ = np.exp(1j * u * jp.gam - 0.5 * jp.del_**2 * u**2)
        R = R + (jp.lam * (phiJ - 1.0))[:, None]

    lower = np.zeros((N, M1), dtype=complex)
    diag = np.zeros((N, M1), dtype=complex)
    upper = np.zeros((N, M1), dtype=complex)

    h = np.diff(v)
    hm, hpp = h[:-1], h[1:]                      # spacings around interior nodes
    I = slice(1, M1 - 1)
    Di, mui = D[I], mu[I]
    s_m, s_0, s_p = 2.0 / (hm * (hm + hpp)), -2.0 / (hm * hpp), 2.0 / (hpp * (hm + hpp))
    c_m, c_0, c_p = -hpp / (hm * (hm + hpp)), (hpp - hm) / (hm * hpp), hm / (hpp * (hm + hpp))
    # diffusion
    lower[:, I] += Di * s_m
    diag[:, I] += Di * s_0
    upper[:, I] += Di * s_p
    # real drift: central where monotone, else first-order upwind
    central_ok = np.abs(mui) * np.maximum(hm, hpp) <= 2.0 * Di
    mu_c = np.where(central_ok, mui, 0.0)
    mu_f = np.where(~central_ok & (mui >= 0), mui, 0.0)
    mu_b = np.where(~central_ok & (mui < 0), mui, 0.0)
    lower[:, I] += mu_c * c_m - mu_b / hm
    diag[:, I] += mu_c * c_0 - mu_f / hpp + mu_b / hm
    upper[:, I] += mu_c * c_p + mu_f / hpp
    # complex cross drift: central
    lower[:, I] += cross[:, I] * c_m
    diag[:, I] += cross[:, I] * c_0
    upper[:, I] += cross[:, I] * c_p
    diag[:, I] += R[:, I]

    # v = 0: degenerate PDE, inflow drift kappa*theta (no BC needed)
    mu0 = mu[0]
    if mu0 < 0:  # kappa*theta < 0 cannot happen for valid parameters
        raise ValueError("kappa*theta must be positive")
    upper[:, 0] += mu0 / h[0]
    diag[:, 0] += -mu0 / h[0] + R[:, 0]

    # v = v_max
    hM = h[-1]
    if vmax_bc == "outflow":
        # one-sided (backward) first-order terms, diffusion dropped; the
        # real drift kappa(theta - v_max) is negative for v_max > theta, i.e.
        # the characteristic enters the domain, so no data is needed at v_max
        drift_top = mu[-1] + cross[:, -1]
        lower[:, -1] += -drift_top / hM
        diag[:, -1] += drift_top / hM + R[:, -1]
    elif vmax_bc == "neumann":
        # dW/dv = 0 via a mirrored ghost node: diffusion 2D(W_{M-1}-W_M)/h^2,
        # first-order terms vanish
        lower[:, -1] += 2.0 * D[-1] / hM**2
        diag[:, -1] += -2.0 * D[-1] / hM**2 + R[:, -1]
    else:
        raise ValueError("vmax_bc must be 'outflow' or 'neumann'")
    return lower, diag, upper


def _implicit_system(lower, diag, upper, dt):
    """Coefficients of (I - dt L_k) as (a, b, c) sub/diag/super arrays."""
    return -dt * lower, 1.0 - dt * diag, -dt * upper


def _expm_propagators(lower, diag, upper, dt, chunk: int = 64) -> np.ndarray:
    """P_k = exp(dt L_k) for every mode, shape (N, M1, M1) complex: the exact
    solution operator of the v-semi-discrete per-mode system over one step."""
    from scipy.linalg import expm
    N, M1 = diag.shape
    P = np.empty((N, M1, M1), dtype=np.complex128)
    ii = np.arange(M1)
    for k0 in range(0, N, chunk):
        k1 = min(N, k0 + chunk)
        Lc = np.zeros((k1 - k0, M1, M1), dtype=np.complex128)
        Lc[:, ii, ii] = diag[k0:k1]
        Lc[:, ii[1:], ii[:-1]] = lower[k0:k1, 1:]
        Lc[:, ii[:-1], ii[1:]] = upper[k0:k1, :-1]
        P[k0:k1] = expm(dt * Lc)
    return P


@numba.njit(cache=True)
def _thomas_factor(A, B, C):
    N, M1 = B.shape
    cp = np.empty((N, M1), dtype=np.complex128)
    inv = np.empty((N, M1), dtype=np.complex128)
    min_pivot = np.inf
    for k in range(N):
        den = B[k, 0]
        inv[k, 0] = 1.0 / den
        cp[k, 0] = C[k, 0] * inv[k, 0]
        if abs(den) < min_pivot:
            min_pivot = abs(den)
        for i in range(1, M1):
            den = B[k, i] - A[k, i] * cp[k, i - 1]
            if abs(den) < min_pivot:
                min_pivot = abs(den)
            inv[k, i] = 1.0 / den
            cp[k, i] = C[k, i] * inv[k, i]
    return cp, inv, min_pivot


@numba.njit(cache=True)
def _thomas_solve(A, cp, inv, rhs):
    N, M1 = cp.shape
    W = np.empty((N, M1), dtype=np.complex128)
    d = np.empty(M1, dtype=np.complex128)
    for k in range(N):
        d[0] = rhs[k, 0] * inv[k, 0]
        for i in range(1, M1):
            d[i] = (rhs[k, i] - A[k, i] * d[i - 1]) * inv[k, i]
        W[k, M1 - 1] = d[M1 - 1]
        for i in range(M1 - 2, -1, -1):
            W[k, i] = d[i] - cp[k, i] * W[k, i + 1]
    return W


# --------------------------------------------------------------------------
# Fast cosine reconstruction / reprojection on the midpoint grid
# --------------------------------------------------------------------------

def _reconstruct(W: np.ndarray, n_eval: int) -> np.ndarray:
    """Values on the n_eval-point midpoint grid x_i = a + (i+1/2)(b-a)/n_eval:
    values[i, m] = sum'_k Re(W_km) cos(k pi (i+1/2)/n_eval) - Im(W_km) sin(...),
    via a zero-padded DCT-III / DST-III.
    """
    N, M1 = W.shape
    re = np.zeros((n_eval, M1))
    re[:N] = np.real(W)
    cos_part = 0.5 * _fft.dct(re, type=3, axis=0, workers=-1)
    s = np.zeros((n_eval, M1))
    s[: N - 1] = np.imag(W)[1:]
    sin_part = 0.5 * _fft.dst(s, type=3, axis=0, workers=-1)
    return cos_part - sin_part


def _reproject(values: np.ndarray, N: int) -> np.ndarray:
    """First N cosine coefficients F_k = (2/(b-a)) int_a^b f cos(u_k(x-a)) dx,
    by the midpoint rule on the n_eval-point grid (a DCT-II).

    With n_eval = N this is the classical DCT interpolation (exact inverse of
    ``_reconstruct`` at the nodes). With n_eval = os*N (os > 1) it is a
    quadrature approximation of the L2 projection onto the first N modes:
    the part of max(continuation, payoff) and of the barrier clipping that
    lies outside the retained modes is discarded, rather than aliased into
    low (weakly damped) modes where it would build up over the n_steps
    projections. Both the barrier h and the strike x = 0 region are
    sampled on the fine grid, and h lies on a fine-cell boundary.
    """
    n_eval = values.shape[0]
    return _fft.dct(values, type=2, axis=0, workers=-1)[:N] / n_eval


# --------------------------------------------------------------------------
# One pricing run at a fixed number of time steps
# --------------------------------------------------------------------------

def _resolve_barrier_method(spec: OptionSpec, jp: JumpParams, barrier_method: str) -> str:
    if not spec.is_continuous():
        return "zero"
    if barrier_method == "auto":
        if spec.american:
            return "zero" if jp.active else "cap"
        return "bgk"
    if barrier_method in ("cap", "cap_bgk") and (jp.active or not spec.american):
        raise ValueError("barrier_method='cap' is exact only for American exercise "
                          "with continuous paths (no jumps)")
    return barrier_method


def _single_run(spec, hp, jp, spots, n_steps, N, M, v_max, L, vmax_bc, exercise_every,
                barrier_method, dt_domain, want_boundary, oversample, time_integrator, v_grid):
    T, K = spec.T, spec.K
    dt = T / n_steps
    h = np.log(spec.H / K)

    v, dv, m0 = _variance_grid(hp.v0, v_max, M, v_grid)
    M1 = len(v)
    a, b = _x_domain(spec, hp, jp, spots, N, L, dt_domain, v[-1])
    dx = (b - a) / N
    u = np.arange(N) * np.pi / (b - a)
    n_eval = oversample * N
    dxf = (b - a) / n_eval
    xs = a + (np.arange(n_eval) + 0.5) * dxf  # fine midpoint grid; h is a cell edge
    g = np.maximum(np.exp(xs) - 1.0, 0.0)
    above = xs >= h
    cap_value = max(np.exp(h) - 1.0, 0.0)

    lower, diag, upper = _mode_operator(u, v, dv, spec.r, spec.q, hp, jp, vmax_bc)
    if time_integrator == "expm":
        Pk = _expm_propagators(lower, diag, upper, dt)
        min_pivot = np.nan

        def propagate(F):
            return np.matmul(Pk, F.astype(np.complex128)[:, :, None])[:, :, 0]
    elif time_integrator == "be":
        A, B, C = _implicit_system(lower, diag, upper, dt)
        cp, inv, min_pivot = _thomas_factor(A, B, C)

        def propagate(F):
            return _thomas_solve(A, cp, inv, F.astype(np.complex128))
    elif time_integrator == "rbe":
        # Richardson-extrapolated backward Euler: 2*BE(dt/2)^2 - BE(dt).
        # Second order; stability function R(z) = 2/(1+z/2)^2 - 1/(1+z)
        # satisfies |R| <= 1 on Re z >= 0 and R(inf) = 0 (A- and L-stable),
        # so stiff high-frequency modes are damped as by BE itself.
        A, B, C = _implicit_system(lower, diag, upper, dt)
        cp, inv, min_pivot = _thomas_factor(A, B, C)
        Ah, Bh, Ch = _implicit_system(lower, diag, upper, 0.5 * dt)
        cph, invh, min_pivot_h = _thomas_factor(Ah, Bh, Ch)
        min_pivot = min(min_pivot, min_pivot_h)

        def propagate(F):
            Fc = F.astype(np.complex128)
            half = _thomas_solve(Ah, cph, invh, _thomas_solve(Ah, cph, invh, Fc))
            return 2.0 * half - _thomas_solve(A, cp, inv, Fc)
    else:
        raise ValueError("time_integrator must be 'expm', 'rbe' or 'be'")

    # barrier schedule on the time grid: node j <-> calendar time j*dt
    if spec.is_continuous():
        monitor = np.ones(n_steps + 1, dtype=bool)
    else:
        cal = np.asarray(sorted(T - np.asarray(spec.monitor_dates)))
        idx = np.round(cal / dt).astype(int)
        if not np.allclose(idx * dt, cal, atol=1e-9 * max(1.0, T)):
            raise ValueError(f"monitoring dates {cal} are not multiples of dt=T/{n_steps}; "
                              "choose n_steps as a multiple of the schedule")
        monitor = np.zeros(n_steps + 1, dtype=bool)
        monitor[idx] = True
        monitor[n_steps] = True  # maturity is always a monitoring date

    if barrier_method in ("bgk", "cap_bgk"):
        # fraction of each fine cell lying below the per-node shifted
        # barrier h - beta*sqrt(v dt): exact midpoint-rule weight for the
        # indicator, so the effective barrier moves continuously with dt
        # rather than in whole-cell jumps
        h_eff = h - BGK_BETA * np.sqrt(v * dt)                      # (M1,)
        left = (xs - 0.5 * dxf)[:, None]
        alive_w = np.clip((h_eff[None, :] - left) / dxf, 0.0, 1.0)  # (n_eval, M1)
    else:
        alive_w = None

    def apply_barrier(values):
        if barrier_method == "cap":
            values[above, :] = cap_value
        elif barrier_method == "cap_bgk":
            # stopped value e^h - 1 on the continuity-shifted region
            # x >= h - beta*sqrt(v dt) (fractional weight in the edge cell)
            values[:] = alive_w * values + (1.0 - alive_w) * cap_value
        elif barrier_method == "bgk":
            values *= alive_w
        else:
            values[above, :] = 0.0
        return values

    # terminal condition
    values = np.tile(g[:, None], (1, M1))
    values = apply_barrier(values)
    F = _reproject(values, N)

    for j in range(n_steps - 1, 0, -1):
        W = propagate(F)
        values = _reconstruct(W, n_eval)
        exercise_now = spec.american and (j % exercise_every == 0)
        if spec.is_continuous():
            # continuous monitoring: every node x >= h is already stopped
            # (value e^h - 1, "cap") or knocked out (0), so exercise is applied
            # first and the barrier step then overwrites x >= h
            if exercise_now:
                np.maximum(values, g[:, None], out=values)
            values = apply_barrier(values)
        else:
            # discrete monitoring date: knock-out first, then exercise, since
            # with (near-)continuous exercise the holder decides just before
            # the monitoring instant, V(t_i-) = max(1{x<h} V(t_i+), payoff)
            if monitor[j]:
                values = apply_barrier(values)
            if exercise_now:
                np.maximum(values, g[:, None], out=values)
        F = _reproject(values, N)

    W0 = propagate(F)
    w0 = W0[:, m0]
    weight = np.ones(N)
    weight[0] = 0.5

    x0s = np.log(spots / K)
    phase = np.exp(1j * u[None, :] * (x0s[:, None] - a)) * weight[None, :]
    price = np.real(phase @ w0)
    dx_ = np.real(phase @ (1j * u * w0))
    dxx = np.real(phase @ (-(u**2) * w0))
    intrinsic = np.maximum(spots - K, 0.0) / K
    delta = dx_ / spots
    gamma = (dxx - dx_) / spots**2
    if spec.american:
        ex = intrinsic >= price
        price = np.where(ex, intrinsic, price)
        delta = np.where(ex, np.where(spots > K, 1.0 / K, 0.0) * 1.0, delta)
        gamma = np.where(ex, 0.0, gamma)
    dead = x0s >= h if (spec.is_continuous() or monitor[0]) else np.zeros_like(x0s, bool)
    price = np.where(dead, 0.0, price)
    delta = np.where(dead, 0.0, delta)
    gamma = np.where(dead, 0.0, gamma)

    info = {"a": a, "b": b, "dx": dx, "n_eval": n_eval, "dv": dv, "M1": M1, "v_top": v[-1],
            "min_pivot": float(min_pivot), "barrier_method": barrier_method}
    if want_boundary and spec.american:
        cont = _reconstruct(W0, n_eval)  # continuation value at t=0 on the (x, v) grid
        info["boundary"] = _exercise_boundary(xs, cont, g, h, v, K)
    return K * price, K * delta, K * gamma, info


def _exercise_boundary(xs, cont, g, h, v, K):
    """Early-exercise boundary S*(v) at t=0: for each variance node, the
    lowest spot in (K, H) above which continuation <= intrinsic, located by
    linear interpolation of cont - intrinsic between collocation nodes.
    """
    out = np.full(len(v), np.nan)
    live = (xs > 0) & (xs < h)
    idx = np.where(live)[0]
    for m in range(len(v)):
        phi = cont[idx, m] - g[idx]
        pos = np.where(phi > 1e-10)[0]
        if len(pos) == 0:
            out[m] = K * np.exp(xs[idx[0]])
            continue
        last = pos[-1]
        if last == len(idx) - 1:
            continue  # no exercise region below the barrier on this line
        x1, x2 = xs[idx[last]], xs[idx[last + 1]]
        p1, p2 = phi[last], phi[last + 1]
        xb = x1 + p1 * (x2 - x1) / (p1 - p2)
        out[m] = K * np.exp(xb)
    return {"v": v, "S_star": out}


# --------------------------------------------------------------------------
# Extrapolation across time-step levels
# --------------------------------------------------------------------------

def _extrapolate(levels: list[np.ndarray], mode: str):
    """Return (value, err_estimate, p_hat) arrays from per-level arrays."""
    if mode == "none" or len(levels) == 1:
        z = np.full_like(levels[-1], np.nan)
        return levels[-1], z, z
    if mode == "sqrt":
        # C(dt) = C* + c1 dt^(1/2) + c2 dt + ...: eliminate the dt^(1/2) term
        # (level ratio sqrt(2)), then the dt term (ratio 2). With four
        # levels the value uses the finest three and the error estimate is
        # the change against the value from the coarsest three.
        def two_term(c1, c2, c3):
            r2 = np.sqrt(2.0)
            d1 = (r2 * c2 - c1) / (r2 - 1.0)
            d2 = (r2 * c3 - c2) / (r2 - 1.0)
            return 2.0 * d2 - d1
        val = two_term(*levels[-3:])
        if len(levels) >= 4:
            err = np.abs(val - two_term(*levels[-4:-1]))
        else:
            err = np.abs(levels[-1] - levels[-2])
        return val, err, np.full_like(val, 0.5)
    if mode == "p1":
        c1, c2 = levels[-2], levels[-1]
        return 2 * c2 - c1, np.abs(c2 - c1), np.ones_like(c2)
    c1, c2, c3 = levels
    d1, d2 = c1 - c2, c2 - c3
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.log2(np.abs(d1) / np.abs(d2))
    ok = (np.sign(d1) == np.sign(d2)) & np.isfinite(p) & (p >= 0.3) & (p <= 3.0) & (d2 != 0)
    val = np.where(ok, c3 + (c3 - c2) / (2.0 ** np.where(ok, p, 1.0) - 1.0), c3)
    err = np.where(ok, np.abs(val - c3), np.maximum(np.abs(d1), np.abs(d2)))
    tiny = (np.abs(d1) < 1e-12) & (np.abs(d2) < 1e-12)
    val = np.where(tiny, c3, val)
    err = np.where(tiny, 0.0, err)
    return val, err, np.where(ok, p, np.nan)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _exercise_stride(n_steps: int, n_exercise_dates: int | None) -> int:
    if n_exercise_dates is None:
        return 1
    if n_steps % n_exercise_dates:
        raise ValueError(f"n_steps={n_steps} is not a multiple of n_exercise_dates={n_exercise_dates}")
    return n_steps // n_exercise_dates


def price_barrier_call(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                        n_steps: int | None = None, N: int = 2048, M: int = 100,
                        v_max: float | None = None, L: float = 10.0,
                        vmax_bc: str = "outflow", n_exercise_dates: int | None = None,
                        barrier_method: str = "auto", extrapolation: str = "auto",
                        spots=None, return_boundary: bool = False,
                        oversample: int = 4, time_integrator: str = "rbe",
                        v_grid: str = "sinh",
                        n_exercise: int | None = None, richardson: bool | None = None,
                        n_eval=None) -> PricingResult:
    """Price an up-and-out (or, via ``spec.knock_in``, up-and-in) call.

    ``n_steps``: base number of backward-Euler time steps (legacy alias
    ``n_exercise``). With ``n_exercise_dates=None`` (default) American
    exercise is checked at every time step on every level, i.e. the
    continuous-exercise limit is approached together with the time step.
    An integer ``n_exercise_dates`` instead fixes a Bermudan contract with
    that many equally spaced exercise dates (n_steps must be a multiple of
    it), identical on every level, which separates the Bermudan-exercise
    error from the time-integration and barrier-monitoring errors.

    ``extrapolation``: ``"auto"`` (``"sqrt"`` for continuous monitoring,
    ``"adaptive"`` otherwise), ``"sqrt"`` (levels n..8n, dt^(1/2) and dt
    terms eliminated), ``"adaptive"`` (levels n, 2n, 4n, observed order),
    ``"p1"`` (levels n, 2n, fixed first order) or ``"none"``. The legacy
    ``richardson=True/False`` maps to ``"p1"``/``"none"``.

    ``oversample``: collocation oversampling factor for the projection
    step (n_eval = oversample*N); 1 = plain DCT interpolation.

    ``spots``: optional array of additional spot prices, all priced from
    the same run (``extra["spots"]``, ``extra["prices"]``, ...).

    ``extra`` also contains the per-level prices, the observed order
    ``p_hat``, an error estimate ``err_est`` and grid diagnostics.
    """
    if n_steps is None:
        n_steps = n_exercise if n_exercise is not None else 50
    if richardson is not None:
        extrapolation = "p1" if richardson else "none"
    v_max = v_max if v_max is not None else max(1.0, 6.0 * hp.theta, 6.0 * hp.v0)
    bm = _resolve_barrier_method(spec, jp, barrier_method)

    t_start = _time.perf_counter()
    spot_arr = np.atleast_1d(np.asarray([spec.S0] + ([] if spots is None else list(spots)),
                                        dtype=float))
    if extrapolation == "auto":
        # continuous monitoring: barrier/stopping discretization error
        # expands in dt^(1/2), dt (see module docstring); otherwise the
        # observed-order estimator
        extrapolation = "sqrt" if spec.is_continuous() else "adaptive"
    n_lvl = {"none": 1, "p1": 2, "adaptive": 3, "sqrt": 4}[extrapolation]
    levels = [n_steps * 2**i for i in range(n_lvl)]
    dt_domain = spec.T / n_steps  # domain fixed across levels (coarsest dt)

    P, D, G, infos = [], [], [], []
    for i, n in enumerate(levels):
        p, d, g_, info = _single_run(spec, hp, jp, spot_arr, n, N, M, v_max, L, vmax_bc,
                                     _exercise_stride(n, n_exercise_dates), bm, dt_domain,
                                     return_boundary and i == len(levels) - 1, oversample,
                                     time_integrator, v_grid)
        P.append(p); D.append(d); G.append(g_); infos.append(info)

    price, err, p_hat = _extrapolate(P, extrapolation)
    delta, _, _ = _extrapolate(D, extrapolation)
    gamma, _, _ = _extrapolate(G, extrapolation)

    if spec.knock_in:
        from .charfunc import cos_vanilla_call
        if spec.american:
            raise NotImplementedError("American knock-in is not a parity instrument")
        hb = 1e-2 * spot_arr
        van = np.array([cos_vanilla_call(s, spec.K, spec.r, spec.q, spec.T, hp, jp) for s in spot_arr])
        up = np.array([cos_vanilla_call(s + e, spec.K, spec.r, spec.q, spec.T, hp, jp) for s, e in zip(spot_arr, hb)])
        dn = np.array([cos_vanilla_call(s - e, spec.K, spec.r, spec.q, spec.T, hp, jp) for s, e in zip(spot_arr, hb)])
        price = van - price
        delta = (up - dn) / (2 * hb) - delta
        gamma = (up - 2 * van + dn) / hb**2 - gamma

    runtime = _time.perf_counter() - t_start
    extra = {"levels": levels, "level_prices": [float(x[0]) for x in P],
             "p_hat": float(p_hat[0]), "err_est": float(err[0]),
             "N": N, "M": M, "v_grid": v_grid, "oversample": oversample, "time_integrator": time_integrator, "v_max": v_max, "vmax_bc": vmax_bc, "L": L,
             "barrier_method": bm, "extrapolation": extrapolation,
             "n_exercise_dates": n_exercise_dates, "grid": infos[-1],
             "spots": spot_arr, "prices": price, "deltas": delta, "gammas": gamma,
             "err_ests": err, "p_hats": p_hat,
             # legacy keys
             "n_exercise": n_steps, "richardson": extrapolation != "none"}
    if len(P) >= 2:
        extra["bermudan_n"] = float(P[0][0])
        extra["bermudan_2n"] = float(P[1][0])
    if return_boundary and "boundary" in infos[-1]:
        extra["boundary"] = infos[-1]["boundary"]
    return PricingResult(price=float(price[0]), delta=float(delta[0]), gamma=float(gamma[0]),
                         runtime=runtime, extra=extra)


def stability_diagnostics(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                          n_steps: int = 200, N: int = 1024, M: int = 100,
                          v_max: float | None = None, L: float = 10.0,
                          vmax_bc: str = "outflow", n_modes_sample: int = 64,
                          v_grid: str = "sinh") -> dict:
    """Matrix-level stability check of the per-mode backward-Euler operator,
    including the boundary rows (complements the frozen-coefficient
    von Neumann argument in the module docstring).

    For a sample of modes k (always including k=0 and k=N-1), returns
    the spectral abscissa max Re eig(L_k), the 2-norm of the one-step solution
    operator ||(I - dt L_k)^{-1}||_2 (<= 1 means contractive), and the
    smallest pivot of the unpivoted Thomas factorization over all modes.
    """
    v_max = v_max if v_max is not None else max(1.0, 6.0 * hp.theta, 6.0 * hp.v0)
    dt = spec.T / n_steps
    v, dv, m0 = _variance_grid(hp.v0, v_max, M, v_grid)
    a, b = _x_domain(spec, hp, jp, np.array([spec.S0]), N, L, dt, v[-1])
    u = np.arange(N) * np.pi / (b - a)
    lower, diag, upper = _mode_operator(u, v, dv, spec.r, spec.q, hp, jp, vmax_bc)
    A, B, C = _implicit_system(lower, diag, upper, dt)
    _, _, min_pivot = _thomas_factor(A, B, C)
    ks = np.unique(np.concatenate([[0, N - 1], np.linspace(0, N - 1, n_modes_sample).astype(int)]))
    max_abscissa, max_norm, max_expm_norm = -np.inf, 0.0, 0.0
    M1 = len(v)
    for k in ks:
        Lk = np.diag(diag[k]) + np.diag(lower[k, 1:], -1) + np.diag(upper[k, :-1], 1)
        Ak = np.eye(M1) - dt * Lk
        max_abscissa = max(max_abscissa, float(np.max(np.linalg.eigvals(Lk).real)))
        max_norm = max(max_norm, float(np.linalg.norm(np.linalg.inv(Ak), 2)))
        from scipy.linalg import expm
        max_expm_norm = max(max_expm_norm, float(np.linalg.norm(expm(dt * Lk), 2)))
    return {"max_spectral_abscissa": max_abscissa, "max_step_norm_2": max_norm,
            "max_expm_step_norm_2": max_expm_norm,
            "bound_1_over_1_plus_r_dt": 1.0 / (1.0 + spec.r * dt),
            "min_thomas_pivot": float(min_pivot), "M1": M1, "dv": dv, "dt": dt,
            "u_max": float(u[-1])}
