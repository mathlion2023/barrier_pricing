"""Method-of-lines (MOL) pricer for up-and-out barrier options under Heston
stochastic volatility (+ optional Merton log-normal jumps), following

    Chiarella, Kang & Meyer (2012), "The evaluation of barrier option
    prices under stochastic volatility", Comput. Math. Appl. 64, 2034-2048.

Equation numbers referenced in comments below (eq. 21-35 etc.) are from
that paper. The Riccati-transform recursions themselves (eq. 31-34) are
carried over unchanged from ``Fortran/Ame_cont.f90`` (Chiarella/Kang's own
reference implementation, subroutines ``riccati``/``fwswp``/``bwswp``),
including its Gauss-Seidel sweep order across variance lines (increasing
m, each line using the just-updated m-1 neighbour) -- see ``mol_kernel.py``
for why that ordering is load-bearing here, not just a style choice
carried over out of caution. The per-line recursions over the spot grid
are themselves compiled with Numba so that sweeping ~100-300 lines does
not mean paying Python-loop overhead per line; this module builds the
grids, assembles the per-regime coefficients, drives the time stepping
(including the discrete-monitoring domain switch and the American free
boundary read-out) and hands each time step's linear-algebra to that
compiled kernel.

Delta (V) is a state of the ODE system, not a finite difference of the
price -- so it converges at the same rate as the price itself (paper,
Section 4). Gamma is likewise read off algebraically from the right-hand
side of the V-ODE (eq. 30/34): gamma = A(S)*C + B(S)*V + P(S), again no
numerical differentiation.
"""
from __future__ import annotations

import time as _time

import numpy as np

from .grid import make_dual_S_grid, make_v_grid, remesh_2d
from .mol_kernel import gauss_seidel_step
from .params import HestonParams, JumpParams, NO_JUMPS, OptionSpec, PricingResult


class MOLGrids:
    """Bundles the two spot grids a discretely-monitored option needs.

    ``narrow`` covers (0, H) -- used only exactly at a monitoring instant,
    where the Dirichlet knock-out condition C(H,.)=0 (eq. 16) is the
    domain's own outer boundary. ``wide`` covers (0, S_max) and is used at
    every other time step (eq. 17: no barrier constraint away from
    monitoring dates). For continuously-monitored options only ``narrow``
    is ever used (the whole domain is (0,H), eq. 8).
    """

    def __init__(self, S_floor: float, H: float, S_max: float, n: int,
                 concentrate_at: float):
        stretch_n = 0.10 * (H - S_floor)
        stretch_w = 0.10 * (S_max - S_floor)
        # Concentrate near the strike *and* near the barrier (0.98*H, just
        # inside the wall) -- the barrier develops a sharp boundary layer
        # that a strike-only grid badly under-resolves.
        self.narrow = make_dual_S_grid(S_floor, H, n, concentrate_at, 0.98 * H, stretch_n)
        # on the wide grid the knock-out projection puts a jump exactly at H,
        # so the local refinement is centred on H itself
        wide = make_dual_S_grid(S_floor, S_max, n, concentrate_at, H, stretch_w)
        # the barrier must be a node of the wide grid, so the knock-out
        # projection at a monitoring instant places the jump exactly at H
        j = int(np.argmin(np.abs(wide - H)))
        wide[j] = H
        self.wide = np.unique(wide)


def _line_coefficients(v: np.ndarray, dv: float, r: float, q: float,
                        hp: HestonParams, jp: JumpParams, time_diag: float):
    """A_m(S)-independent-of-S pieces, B_m(S)-independent-of-S pieces, and
    the upwind/diffusion split coefficients for the source term -- all of
    these depend only on the variance line ``m`` (via v[m]) and on the
    current time-differencing regime, so they are computed once per
    regime and reused across every outer (line-Jacobi) sweep and every
    time step until the regime changes (eq. 27 vs 28, or a grid switch).

    Returns per-line arrays (length M+1; row 0 unused/ignored):
      A_over_S2, B_over_S  such that  A_m(S)=A_over_S2[m]/S**2,
                                       B_m(S)=B_over_S[m]/S
      coefP, coefM : multiply C_{m+1}, C_{m-1} in the source term
      cross_coef   : multiplies S*(V_{m+1}-V_{m-1}) in the source term
    """
    M1 = len(v)
    alpha = hp.kappa * hp.theta
    beta = hp.kappa  # market price of volatility risk (lambda_v) not modelled: beta = kappa
    d = alpha - beta * v  # drift of the variance process, eq. (6)/(24)

    diffusion = hp.sigma**2 * v / (2.0 * dv**2)
    abs_d_over_dv = np.abs(d) / dv

    # diag_C (eq. 27/28 rearranged): coefficient of the unknown C_m in the
    # ODE, before dividing through by (v_m S^2/2). The extra jump-intensity
    # term (-lam) is Fortran's ``delta()`` correction, ported unchanged.
    diag_C = -(r + jp.lam + 2.0 * diffusion + abs_d_over_dv + time_diag)

    # A_m(S) = -2*diag_C/(v_m S^2), B_m(S) = -2(r-q-lam*ak1)/(v_m S) (paper's
    # worked example under eq. 30, jump-compensated per Fortran's
    # ``beta=(rate-q-alam*ak1)*y``, Ame_cont.f90 line ~526 -- the Python
    # port previously dropped the ``-alam*ak1`` compensator here, which
    # left the discounted spot price without its risk-neutral drift
    # correction whenever jumps were active; jp.ak1=0 when jp is NO_JUMPS
    # so this is a no-op in the no-jump case). Both still need dividing by
    # v_m*S^k in `_apply_A_B`; row 0 (v_m=0) is guarded there and never
    # actually used (row 0 is filled by extrapolation, see
    # `_quadratic_extrapolate_row0`).
    A_over_S2 = -2.0 * diag_C
    B_over_S = np.full(M1, -2.0 * (r - q - jp.lam * jp.ak1))

    # Upwind split of the (alpha - beta*v)*dC/dv term (eq. 24), written as
    # coefficients on C_{m+1} and C_{m-1} so it can be added directly into
    # the source term alongside the central second-difference in v.
    coefP = np.where(d >= 0.0, diffusion + d / dv, diffusion)
    coefM = np.where(d >= 0.0, diffusion, diffusion - d / dv)

    cross_coef = hp.rho * hp.sigma * v / (2.0 * dv)

    return A_over_S2, B_over_S, coefP, coefM, cross_coef, v


def _time_diag_and_prev(regime: str, dtau: float, C_prev, C_prevprev):
    """Backward-difference-in-tau contribution, split into the part that
    multiplies the unknown C_m^n (folded into diag_C via ``time_diag``,
    see ``_line_coefficients``) and the known part built from previous
    time levels (eq. 25 for the first few steps, eq. 26 -- BDF2 --
    afterwards).
    """
    if regime == "euler":
        return 1.0 / dtau, C_prev / dtau
    # BDF2 (eq. 26): (3/2 C^n - 2 C^{n-1} + 1/2 C^{n-2})/dtau
    return 1.5 / dtau, (2.0 * C_prev - 0.5 * C_prevprev) / dtau


def _final_gamma(S, v, dv, A_over_S2, B_over_S, coefP, coefM, cross_coef,
                  prev_terms, C, V, jp: JumpParams = NO_JUMPS,
                  jump_factor=None, jump_weight=None):
    """Gamma at the converged final time level: RHS of eq. 30/34,
    A(S)*C+B(S)*V+P(S). Not part of the per-sweep kernel (that only
    needs C, V) -- computed once, in plain NumPy, since it runs a single
    time after the kernel's Gauss-Seidel loop has already converged.
    """
    v_safe = np.where(v == 0.0, 1.0, v)[:, None]
    S_row = S[None, :]
    A = A_over_S2[:, None] / (v_safe * S_row**2)
    B = B_over_S[:, None] / (v_safe * S_row)

    C_plus = np.roll(C, -1, axis=0)
    C_plus[-1] = C[-2]
    C_minus = np.roll(C, 1, axis=0)
    V_plus = np.roll(V, -1, axis=0)
    V_plus[-1] = V[-2]
    V_minus = np.roll(V, 1, axis=0)

    known = (cross_coef[:, None] * S_row * (V_plus - V_minus)
             + coefP[:, None] * C_plus + coefM[:, None] * C_minus + prev_terms)

    if jp.active:
        jump_sum = np.zeros_like(C)
        for factor, weight in zip(jump_factor, jump_weight):
            for m_row in range(C.shape[0]):
                jump_sum[m_row] += weight * np.interp(
                    S * factor, S, C[m_row], left=C[m_row, 0], right=C[m_row, -1])
        known = known + jp.lam * jump_sum

    P = -known / (v_safe * S_row**2 / 2.0)
    return A * C + B * V + P


def _jump_quadrature(jp: JumpParams, HM: int = 12):
    """Gauss-Hermite nodes/weights for the compensated jump-integral
    source term lam*(E[C(S*Y)]-C(S)), Y=e^J, J~N(gam,del^2) (Merton/
    Bates log-normal jumps). Standard substitution J=gam+del*sqrt(2)*x
    turns E[C(S*e^J)] into (1/sqrt(pi)) * sum_l w_l * C(S*jump_factor[l])
    with jump_factor[l]=exp(gam+del*sqrt(2)*x_l) -- the same substitution
    Fortran's ``hermite`` subroutine makes, but using NumPy's built-in
    Hermite quadrature (``numpy.polynomial.hermite.hermgauss``) instead
    of re-deriving Fortran's hand-rolled Newton root finder.
    """
    x, w = np.polynomial.hermite.hermgauss(HM)
    jump_factor = np.exp(jp.gam + jp.del_ * np.sqrt(2.0) * x)
    jump_weight = w / np.sqrt(np.pi)
    return jump_factor, jump_weight


def _build_time_grid(T: float, n_steps: int, monitor_taus: list[float] | None):
    """Uniform time grid over [0,T] refined so that every monitoring
    tau lands exactly on a grid node (needed so the domain switch in
    ``price_barrier_call`` happens at an exact time step, not between
    two of them).
    """
    if not monitor_taus:
        return np.linspace(0.0, T, n_steps + 1), np.zeros(0, dtype=bool)

    nodes = sorted(set([0.0, *monitor_taus, T]))
    grid = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        seg_steps = max(1, round(n_steps * (b - a) / T))
        seg = np.linspace(a, b, seg_steps + 1)
        grid.extend(seg[:-1].tolist())
    grid.append(T)
    tau = np.array(grid)
    is_monitor = np.isin(np.round(tau, 10), np.round(monitor_taus, 10))
    return tau, is_monitor


def price_barrier_call(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                        n_S: int = 200, M: int = 200, v_max: float | None = None,
                        n_time_steps: int = 200, S_max_mult: float = 2.2,
                        sor_tol: float = 1e-8, sor_max_iter: int = 60,
                        monitor_mode: str = "instant") -> PricingResult:
    """Price an up-and-out (or, via ``spec.knock_in``, up-and-in) call.

    Grid resolution knobs mirror the paper's (N, M, S_pts) notation in
    Tables 2-5: ``n_time_steps``=N, ``M``=variance lines, ``n_S``=S_pts.
    The per-time-step linear algebra (Gauss-Seidel sweep across variance
    lines, each line's Riccati/forward/backward recursion) is delegated
    to the Numba-compiled kernel in ``mol_kernel.py``.

    ``monitor_mode`` (discrete monitoring only):
      * ``"instant"`` (default): the step ending at a monitoring date is
        solved on the wide domain like any other, and the knock-out is then
        applied as an instantaneous projection C = V = 0 for S >= H. That is
        the contract: the barrier is observed at that instant only.
      * ``"ckm"``: the legacy treatment (Fortran ``Eur_disc.f90`` /
        ``Ame_disc.f90``, paper Sec. 2.2): the step ending at the monitoring
        date is solved on the narrow domain (0, H) with the barrier condition
        at H. That makes the barrier binding continuously over one whole
        time step, which is an O(sqrt(dt)) approximation of instantaneous
        monitoring.
    """
    if monitor_mode not in ("instant", "ckm"):
        raise ValueError("monitor_mode must be 'instant' or 'ckm'")
    t_start = _time.perf_counter()
    v_max = v_max if v_max is not None else max(1.0, 8.0 * hp.theta)
    v, dv = make_v_grid(v_max, M)

    # S_floor must not sit too close to 0: A_m(S) ~ 1/(v_m*S^2), and for
    # small v_m lines this is already large (dominated by the 1/dtau time
    # term); combined with a tiny S it pushes the backward-sweep
    # recursion's denominator into catastrophic cancellation (verified:
    # S_floor=0.01*K reliably goes non-finite a few steps in for
    # realistic grids/dtau, S_floor>=0.1*K does not). A deep-OTM call is
    # worth ~0 there regardless, so this costs nothing accuracy-wise for
    # any spot the option is actually priced at.
    S_floor = 0.2 * spec.K
    S_max = S_max_mult * spec.H
    grids = MOLGrids(S_floor, spec.H, S_max, n_S, concentrate_at=spec.K)

    jump_factor, jump_weight = _jump_quadrature(jp) if jp.active else (np.zeros(1), np.zeros(1))

    continuous = spec.is_continuous()
    monitor_taus = None if continuous else sorted(spec.T - np.asarray(spec.monitor_dates))
    tau_grid, is_monitor = _build_time_grid(spec.T, n_time_steps, monitor_taus)

    S = grids.narrow if continuous else grids.wide.copy()
    n = S.shape[0]
    M1 = M + 1

    # Terminal condition (eq. 7/14): payoff (S-K)^+, clipped to the active
    # domain (knock-out call is worthless at/above the barrier already).
    C = np.maximum(S[None, :] - spec.K, 0.0) * np.ones((M1, 1))
    C[:, S >= spec.H] = 0.0
    if not continuous:
        iH = int(np.argmin(np.abs(S - spec.H)))
        if abs(S[iH] - spec.H) < 1e-12 and 0 < iH < len(S) - 1:
            C[:, iH] = (S[iH] - S[iH - 1]) / (S[iH + 1] - S[iH - 1]) * max(spec.H - spec.K, 0.0)
    V = np.where(S[None, :] > spec.K, 1.0, 0.0) * np.ones((M1, 1))
    C_prev = C.copy()
    C_prevprev = C.copy()

    R = None  # Riccati solution cache, shape (M1, n_of_current_grid); recomputed as needed
    regime = "euler"
    euler_steps_left = 2  # first two steps use eq. 25 (implicit Euler), then switch to eq. 26
    prev_regime = None
    prev_dtau = None
    time_diag = prev_terms = A_over_S2 = B_over_S = coefP = coefM = cross_coef = None

    for step in range(1, len(tau_grid)):
        dtau = tau_grid[step] - tau_grid[step - 1]
        regime = "euler" if euler_steps_left > 0 else "bdf2"

        at_monitor = bool(is_monitor[step]) if not continuous else False
        project_after = at_monitor and monitor_mode == "instant"
        if project_after:
            at_monitor = False  # solve this step on the wide grid, project afterwards
        if at_monitor:
            # Solve *this* step on the narrow (0,H) domain: remesh the
            # incoming wide-domain state onto it first (Sec. 2.2).
            S_step = grids.narrow
            C_in = remesh_2d(S, S_step, C)
            Cp_in = remesh_2d(S, S_step, C_prev)
            Cpp_in = remesh_2d(S, S_step, C_prevprev)
            V_in = remesh_2d(S, S_step, V)
            recompute_R = True  # grid changed: Riccati R must be recomputed
        else:
            S_step = S
            C_in, Cp_in, Cpp_in = C, C_prev, C_prevprev
            V_in = V
            # A_m(S) (and hence R_m(S)) depends on the regime AND on dtau
            # itself (through time_diag=1/dtau or 1.5/dtau) -- both must be
            # unchanged from the previous step for the cached R to still
            # be valid, not just an unchanged spot grid.
            recompute_R = (R is None) or regime != prev_regime or dtau != prev_dtau
        prev_regime, prev_dtau = regime, dtau

        time_diag, prev_terms = _time_diag_and_prev(regime, dtau, Cp_in, Cpp_in)
        A_over_S2, B_over_S, coefP, coefM, cross_coef, _ = _line_coefficients(
            v, dv, spec.r, spec.q, hp, jp, time_diag)

        if recompute_R:
            R = np.zeros((M1, len(S_step)))
        C_iter = C_in.copy()
        V_iter = V_in.copy()
        apply_barrier = continuous or at_monitor
        gauss_seidel_step(S_step, v, dv, A_over_S2, B_over_S, coefP, coefM,
                           cross_coef, prev_terms, C_iter, V_iter, R,
                           recompute_R, spec.K, spec.H, spec.american,
                           sor_tol, sor_max_iter, apply_barrier,
                           jp.lam, jump_factor, jump_weight)
        # C_iter, V_iter and (if recompute_R) R were updated in place.

        if at_monitor:
            # Knock-out projection (eq. 16 made explicit off S=H too: any
            # path already at/above H at a monitoring date is dead) and
            # carry the surviving region back onto the wide grid.
            C_next = np.zeros((M1, n))
            V_next = np.zeros((M1, n))
            below = S < spec.H
            C_next[:, below] = remesh_2d(S_step, S[below], C_iter)
            V_next[:, below] = remesh_2d(S_step, S[below], V_iter)
            C_prevprev, C_prev = C_prev, C_next
            C, V = C_next, V_next
            R = None  # about to switch back to the wide grid next step
        else:
            if project_after:
                # instantaneous knock-out at the monitoring date: C = V = 0
                # on S >= H (H is a node of the wide grid)
                dead = S >= spec.H
                iH = int(np.argmin(np.abs(S - spec.H)))
                wH = (S[iH] - S[iH - 1]) / (S[iH + 1] - S[iH - 1])
                left = C_iter[:, iH].copy()
                C_iter[:, dead] = 0.0
                V_iter[:, dead] = 0.0
                # cell average of the jump at the node H (one-sided sampling
                # would make the projection first order and grid-dependent)
                C_iter[:, iH] = wH * left
            C_prevprev, C_prev = C_prev, C_iter
            C, V = C_iter, V_iter

        if euler_steps_left > 0:
            euler_steps_left -= 1
        if at_monitor or project_after:
            euler_steps_left = 2  # restart Euler for 3 steps after a monitoring date (paper, fn. 8)

    # Gamma at the final (tau=T / t=0) level: RHS of eq. 34, no extra solve
    # needed, just one more (non-iterative, plain NumPy) evaluation of the
    # same coefficients/source term the kernel used on its last sweep.
    gamma = _final_gamma(S, v, dv, A_over_S2, B_over_S, coefP, coefM,
                          cross_coef, prev_terms, C, V, jp, jump_factor, jump_weight)

    price, delta, g = _bilinear_readout(S, v, C, V, gamma, spec.S0, hp.v0)

    if spec.knock_in:
        from .charfunc import cos_vanilla_call
        vanilla = cos_vanilla_call(spec.S0, spec.K, spec.r, spec.q, spec.T, hp, jp)
        # Vanilla delta/gamma via a tiny central bump of the same COS
        # pricer (cheap relative to the PDE solve; kept independent of
        # the MOL machinery to avoid coupling the parity leg to it).
        h = 1e-2 * spec.S0
        up = cos_vanilla_call(spec.S0 + h, spec.K, spec.r, spec.q, spec.T, hp, jp)
        dn = cos_vanilla_call(spec.S0 - h, spec.K, spec.r, spec.q, spec.T, hp, jp)
        v_delta = (up - dn) / (2 * h)
        v_gamma = (up - 2 * vanilla + dn) / h**2
        price, delta, g = vanilla - price, v_delta - delta, v_gamma - g

    runtime = _time.perf_counter() - t_start
    return PricingResult(price=price, delta=delta, gamma=g, runtime=runtime,
                          extra={"regime": regime})


def _bilinear_readout(S, v, C, V, gamma, S0, v0):
    """Bilinear interpolation of price/delta/gamma at (S0, v0) from the
    computed grid, matching the read-out Fortran does for its reported
    spot/vol slice.
    """
    ik = np.searchsorted(S, S0) - 1
    ik = np.clip(ik, 0, len(S) - 2)
    p1 = (S0 - S[ik]) / (S[ik + 1] - S[ik])
    p0 = 1.0 - p1

    jm = np.searchsorted(v, v0) - 1
    jm = np.clip(jm, 0, len(v) - 2)
    q1 = (v0 - v[jm]) / (v[jm + 1] - v[jm])
    q0 = 1.0 - q1

    def bilerp(Z):
        return (q0 * (p0 * Z[jm, ik] + p1 * Z[jm, ik + 1])
                + q1 * (p0 * Z[jm + 1, ik] + p1 * Z[jm + 1, ik + 1]))

    return float(bilerp(C)), float(bilerp(V)), float(bilerp(gamma))
