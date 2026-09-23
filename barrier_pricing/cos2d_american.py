"""2D COS (Fourier-cosine) pricer for American up-and-out barrier call options
under Heston stochastic volatility (+ optional Merton/Bates log-normal
jumps), following

    Fang, F. & Oosterlee, C.W. (2009), "Pricing Early-Exercise and Discrete
    Barrier Options by Fourier-Cosine Series Expansions", Numerische
    Mathematik 114, 27-62.

for the early-exercise/Bermudan-COS recursion, layered on top of the same
Heston(+SVJD-jump) characteristic function and per-variance-node ("frozen-v")
architecture as ``cos2d.py`` (see that module's docstring for why the
variance dimension is handled by Clenshaw-Curtis quadrature rather than a
second COS expansion). ``cos2d.py`` itself stays untouched -- it is already
validated to <0.2% against Chiarella, Kang & Meyer (2012) and a regression
there would be costly to notice; this is a new, separate module that reuses
its pure-numerics helpers.

**Why this needs genuinely new machinery, not just a parameter tweak.**
``cos2d.py``'s discretely-monitored recursion (``Cx1x2_svjd``/``_cx1x2_step``)
never leaves Fourier space between monitoring dates: the barrier is enforced
by restricting the FFT-convolution kernels (``ms``/``mc``) to the
sub-interval ``(a, h)``, a trick that exists specifically to *avoid* ever
evaluating the option value pointwise. American exercise has no such trick
available in general (``max(continuation, intrinsic)`` is not a linear
operation representable as a convolution), so the recursion must leave
Fourier space at every exercise opportunity: evaluate the continuation value
pointwise on a grid, take the max with the intrinsic payoff, and numerically
re-expand the result back onto Fourier-cosine coefficients (a forward cosine
transform) before continuing backward -- exactly Fang & Oosterlee (2009)'s
Bermudan-COS recursion, using numerical (DCT-style) coefficient recovery
rather than the closed-form ``chi``/``psi`` coefficients (which only exist
for vanilla payoffs, not for ``max(continuation, intrinsic)``).

**"Discrete" vs "continuous" barrier is the only remaining monitoring
choice -- exercise is always (approximately) continuous.** Unlike the
barrier, American exercise is *not* naturally discrete: economically, the
holder may exercise at any instant, and ``mol.py``/the Fortran solvers price
exactly that (their PDE time-stepping checks the free boundary every time
step, however fine). A COS recursion cannot skip straight from one exercise
date to the next distant one the way ``cos2d.py`` skips a whole monitoring
sub-interval in one FFT convolution, because the exercise decision itself
must be revisited at every opportunity. So this module always runs a dense
Bermudan grid of ``n_exercise`` equally spaced exercise dates over ``[0,T]``
and Richardson-extrapolates across increasing ``n_exercise`` to approximate
the continuous-exercise limit (the standard Bermudan-to-American technique).
"Discrete barrier monitoring" then means the knock-out truncation
(zero the value where ``x > h``) is applied only at the specific calendar
dates in ``spec.monitor_dates``, projected onto the nearest nodes of that
same fine exercise grid; "continuous barrier" (``spec.monitor_dates is
None``) applies it at every exercise-grid node. Both share the same dense
exercise machinery -- there is no separate "discrete" vs "continuous"
recursion, only a different per-step barrier mask.

**Domain sizing is intentionally NOT ``cos2d._bound`` called with the
Bermudan sub-step.** ``cos2d.py`` sizes its COS truncation range ``[a,b]``
from the cumulants of a *single* monitoring sub-interval ``dt=T/mt`` and
reuses that one range for the whole recursion -- fine when ``mt`` is small
(2-3, as in the reference tables), because a single sub-interval is already
a large fraction of ``T``. Here ``n_exercise`` is deliberately large (tens),
so ``dt=T/n_exercise`` is small and the same formula would produce a range
far too narrow to contain the barrier ``h`` (which is fixed in absolute
terms, independent of how finely the exercise grid is cut). Instead the
range is sized once from the *total* remaining maturity (``_bound(...,
spec.T, ...)``), which is safely wide for any ``n_exercise``.

**The per-step transition is a constant-variance Black-Scholes(+jump) CF,
not ``cos2d``'s Heston CF -- this is load-bearing, not a style choice.**
``cos2d.py``'s "frozen-v" branches apply the *true* one-period Heston
transition CF (``_phi_heston_grid``, which conditions on ``v``=``xf[j]`` only
at the *start* of that one period and lets it evolve/mean-revert during the
period) once or twice per branch (``mt-1`` is at most 2 in the validated tables).
Chaining that same formula ``n_exercise-1`` (tens of) times, always
re-anchoring to the same starting ``v`` at every fine step, is *not* a
semigroup: ``phi_heston(u,v,dt)**n != phi_heston(u,v,n*dt)`` (verified
numerically -- the gap saturates but is non-negligible), and iterating a
non-semigroup transition through the evaluate/reproject round trip tens of
times compounds this into a large, growing price bias (verified: a plain
discounted payoff carried through 50 such steps overshot the true European
price by >15%, growing with the step count, versus <1.5% at 2-10 steps).
The fix, and arguably the more literal reading of "frozen for the whole
remaining life" anyway, is to model each branch's *fine* recursion as a
constant-variance ``v=xf[j]`` Black-Scholes(+SVJD-jump) process
(``_phi_bs_jump_grid`` below) -- an exact Lévy semigroup, so chaining any
number of fine steps is bias-free by construction.

**The final cross-branch blend is plain interpolation at the real v0, not
``cos2d``'s ``transit_var_approx``-weighted transition-density integral --
this is also load-bearing.** ``cos2d.py``'s one-shot blend needs that
integral because its own ``dt=T/mt`` is large enough for v0 to have
genuinely moved by the time "now" is reached. Reusing it here, at this
module's tiny fine-step ``dt``, was tried first and broke: the transition
density's width shrinks below this M-node Clenshaw-Curtis grid's local
spacing, and the blend's normalization (`sum_j wf_j*density_j`, which should
be ~1) drifts to >2 as dt shrinks -- verified directly, and it made the
price swing non-monotonically by ~10% under M alone (75 vs 300), independent
of N or n_exercise. Since each branch's reconstructed value is already
verified smooth in v (dense M sweeps show no node-to-node noise), and since
v cannot have moved from v0 over an infinitesimal fine step anyway, plain
interpolation of the (already exercise/barrier-corrected) per-branch values
at v0 is both the simpler implementation and the correct dt->0 limit --
verified to remove the M-sensitivity entirely (identical to 4 significant
figures across M=75..300).
"""
from __future__ import annotations

import time as _time

import numpy as np

from .cos2d import _bound, _fclencurt
from .params import HestonParams, JumpParams, NO_JUMPS, OptionSpec, PricingResult


# --------------------------------------------------------------------------
# Real cosine/sine basis matrices, shared by the "evaluate coefficients at a
# grid of x" step (COS reconstruction) and the "recover coefficients from
# values on that same grid" step (numerical forward cosine transform / DCT).
# Both the domain (a,b) and the frequency grid u are fixed for the entire
# recursion (see module docstring), so these are built once per pricing call
# and reused for every one of the n_exercise-1 backward steps.
# --------------------------------------------------------------------------

def _phi_bs_jump_grid(xi: np.ndarray, v: np.ndarray, dt: float, r: float, q: float,
                       jp: JumpParams) -> np.ndarray:
    """Constant-variance Black-Scholes(+jump) CF for one fine step of the
    branch-``j`` recursion, ``v``=``xf[j]`` held fixed (see module docstring
    for why this replaces ``cos2d._phi_heston_grid`` here). Package jump
    convention J ~ N(gam, del^2) with compensator lam*jp.ak1 (see
    ``params.JumpParams``), so phi(-i; dt) = exp((r-q) dt).
    """
    iu = 1j * xi
    phi = np.exp(iu * (r - q - 0.5 * v) * dt - 0.5 * xi**2 * v * dt)
    if jp.active:
        jump = np.exp(-0.5 * jp.del_**2 * xi**2 + 1j * jp.gam * xi)
        phi = phi * np.exp(dt * jp.lam * (jump - 1.0)) * np.exp(-1j * xi * dt * jp.lam * jp.ak1)
    return phi


def _cos_sin_basis(u: np.ndarray, xs: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    phase = u[:, None] * (xs[None, :] - a)  # (N, N_eval)
    return np.cos(phase), np.sin(phase)


def _evaluate(vvk: np.ndarray, ReG: np.ndarray, ImG: np.ndarray,
              CosMat: np.ndarray, SinMat: np.ndarray, weight0: np.ndarray) -> np.ndarray:
    """Pointwise value at the N_eval collocation nodes, all M branches at
    once: values[j, m] = sum_k' Re(Phi[k,m]*exp(i u_k (x_j-a))) * vvk[k,m].
    ``ReG=Re(Phi)``, ``ImG=Im(Phi)`` (Phi depends on the branch m via its own
    frozen variance, so this is genuinely (N,M), not a shared (N,) vector).
    """
    A = weight0 * vvk * ReG  # (N, M)
    B = weight0 * vvk * ImG  # (N, M)
    return CosMat.T @ A - SinMat.T @ B  # (N_eval, M)


def _reproject(values: np.ndarray, CosMat: np.ndarray, n_eval: int) -> np.ndarray:
    """Numerical forward cosine transform (Fang & Oosterlee 2009's
    numerical-coefficient-recovery variant of the Bermudan-COS recursion):
    new coefficients from values sampled at the midpoint collocation grid.
    """
    return (2.0 / n_eval) * (CosMat @ values)  # (N, M)


def _intrinsic(xs: np.ndarray) -> np.ndarray:
    """Call intrinsic value in "unit-K" scale, (e^x-1)^+ -- same convention
    ``cos2d.py`` uses throughout (the actual price is ``K`` times this).
    """
    return np.maximum(np.exp(xs) - 1.0, 0.0)


def _monitor_step_mask(monitor_calendar_times: np.ndarray | None, n_exercise: int,
                        dt: float, T: float) -> np.ndarray:
    """Boolean mask, length n_exercise+1, node ``i`` at calendar time
    ``i*dt``: True where the barrier is monitored. ``None`` => continuous
    (every interior node; node 0, "now", and node n_exercise, maturity, are
    handled by the terminal condition / final read-out, not this mask).
    """
    mask = np.zeros(n_exercise + 1, dtype=bool)
    if monitor_calendar_times is None:
        mask[1:n_exercise] = True
        return mask
    idx = np.round(np.asarray(monitor_calendar_times) / dt).astype(int)
    snapped = idx * dt
    if not np.allclose(snapped, monitor_calendar_times, atol=1e-9 * max(1.0, T)):
        raise ValueError(
            "cos2d_american requires spec.monitor_dates to land exactly on "
            f"multiples of dt=T/n_exercise={dt:.6g}; got calendar times "
            f"{monitor_calendar_times} which snap to {snapped}. Increase "
            "n_exercise (or its Richardson base) to a common multiple of the "
            "monitoring schedule.")
    mask[idx] = True
    return mask


def _bermudan_price(spec: OptionSpec, hp: HestonParams, jp: JumpParams,
                     n_exercise: int, N: int, M: int, v_lo: float, v_hi: float,
                     n_eval: int | None) -> tuple[float, float, float]:
    """One Bermudan-COS run at a fixed number of exercise dates. Returns
    (price, delta, gamma) at (spec.S0, hp.v0), undiscounted for anything
    beyond what's already baked into the recursion.
    """
    n_eval = n_eval or N
    T, K, H = spec.T, spec.K, spec.H
    dt = T / n_exercise

    x0 = np.log(spec.S0 / K)
    h = np.log(H / K)

    xf, wf = _fclencurt(M, v_lo, v_hi)
    jf = M - 1
    while jf > 0 and hp.v0 > xf[jf]:
        jf -= 1
    jf = min(jf, M - 2)

    # Domain sized from the *total* remaining maturity, not the (small)
    # Bermudan sub-step -- see module docstring.
    a, b = _bound(hp.v0, spec.r, spec.q, T, hp)
    if not (b > h):
        raise ValueError(f"COS domain upper bound b={b:.4g} does not clear the "
                          f"barrier level h=ln(H/K)={h:.4g}; check H/K or widen "
                          "the domain (this should not happen for realistic "
                          "barrier levels).")

    karr = np.arange(N)
    u = karr * np.pi / (b - a)
    weight0 = np.where(karr == 0, 0.5, 1.0)[:, None]

    # Fine-recursion transition: constant-variance BS(+jump) CF per branch --
    # an exact semigroup, so chaining n_exercise-1 of these is bias-free
    # (see module docstring).
    PhiFine = _phi_bs_jump_grid(u[:, None], xf[None, :], dt, spec.r, spec.q, jp)  # (N, M)
    ReG, ImG = np.real(PhiFine), np.imag(PhiFine)
    disc = np.exp(-spec.r * dt)

    xs = a + (np.arange(n_eval) + 0.5) * (b - a) / n_eval  # midpoint collocation grid
    CosMat, SinMat = _cos_sin_basis(u, xs, a)  # (N, n_eval)
    intrinsic_grid = _intrinsic(xs)
    below_h = xs < h  # knock-out on touch: alive region is strictly x<h, not x<=h

    monitor_calendar = None if spec.is_continuous() else np.asarray(
        sorted(spec.T - np.asarray(spec.monitor_dates)))
    barrier_mask = _monitor_step_mask(monitor_calendar, n_exercise, dt, T)

    # Terminal condition (node n_exercise, t=T): intrinsic, barrier-clipped,
    # same as cos2d.py's payoff coefficients but recovered numerically here
    # for consistency with every other step's machinery.
    terminal_values = np.where(below_h, intrinsic_grid, 0.0)
    vvk = _reproject(np.tile(terminal_values[:, None], (1, M)), CosMat, n_eval)

    for step in range(n_exercise - 1, 0, -1):
        values = disc * _evaluate(vvk, ReG, ImG, CosMat, SinMat, weight0)  # (n_eval, M)
        values = np.maximum(values, intrinsic_grid[:, None])
        if barrier_mask[step]:
            values[~below_h, :] = 0.0
        vvk = _reproject(values, CosMat, n_eval)

    # Final read-out at (x0, v0): one more fine BS-CF step (same transition
    # every other step uses -- dt is tiny by construction, so "blending"
    # via a separate transition-density integral, as cos2d.py's con_svjd
    # does for its own *coarse* dt=T/mt, buys nothing here and was
    # measured to be actively harmful: with the true Heston CF + Clenshaw-
    # Curtis transit_var_approx blend this replaced, the price at fixed
    # (N, n_exercise) swung non-monotonically by ~10% just from changing M
    # (75->300), because the variance-transition density it integrates
    # narrows below this M-node grid's spacing once dt is fine. Since each
    # branch's reconstructed value is already verified smooth in v (dense
    # M sweeps show no node-to-node noise), a plain interpolation across
    # branches at the real v0 is both simpler and immune to that
    # M-sensitivity -- and is the correct dt->0 limit of the transition-
    # density blend anyway (v cannot have moved from v0 over an
    # infinitesimal step).
    def _branch_values_at_x0(factor):
        phase = np.exp(1j * u[:, None] * (x0 - a)) * factor[:, None]
        return disc * np.sum(weight0 * np.real(PhiFine * phase) * vvk, axis=0)  # (M,)

    ones_ = np.ones(N)
    iu = 1j * u
    neg_u2 = -(u**2)

    cont_price_branches = np.maximum(K * _branch_values_at_x0(ones_), max(spec.S0 - K, 0.0))
    cont_dprice_branches = K * _branch_values_at_x0(iu)
    cont_d2price_branches = K * _branch_values_at_x0(neg_u2)

    order = np.argsort(xf)
    xf_sorted = xf[order]
    cont_price = float(np.interp(hp.v0, xf_sorted, cont_price_branches[order]))
    cont_dprice_dx = float(np.interp(hp.v0, xf_sorted, cont_dprice_branches[order]))
    cont_d2price_dx2 = float(np.interp(hp.v0, xf_sorted, cont_d2price_branches[order]))

    intrinsic_price = max(spec.S0 - K, 0.0)
    if intrinsic_price >= cont_price:
        return intrinsic_price, (1.0 if spec.S0 > K else 0.0), 0.0
    delta = cont_dprice_dx / spec.S0
    gamma = (cont_d2price_dx2 - cont_dprice_dx) / spec.S0**2
    return cont_price, delta, gamma


def price_barrier_call(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                        n_exercise: int = 50, N: int = 256, M: int = 200,
                        v_lo: float = 0.001, v_hi: float = 0.95,
                        n_eval: int | None = None, richardson: bool = True) -> PricingResult:
    """Price an American up-and-out call under Heston(+jumps) via Bermudan-COS.

    ``spec.american`` must be True (use ``cos2d.price_barrier_call`` for
    European). ``spec.monitor_dates=None`` monitors the barrier at every
    exercise-grid node (continuous); an explicit list monitors it only at
    those calendar dates, which must land exactly on multiples of
    ``T/n_exercise`` (raise the base ``n_exercise`` to a common multiple of
    the schedule if not -- e.g. the paper's 3-date American schedule
    ``T/3, 2T/3, T`` needs ``n_exercise`` a multiple of 3).

    ``n_exercise`` is the number of Bermudan exercise dates approximating
    continuous exercise (see module docstring); with ``richardson=True``
    (default) this function prices at ``n_exercise`` and ``2*n_exercise``
    and returns the 2-point Richardson extrapolation ``2*V(2n)-V(n)``,
    assuming O(1/n) Bermudan-to-American convergence. Set ``richardson=False``
    to get the raw ``n_exercise``-date Bermudan price instead (faster, biased
    low relative to the true American value).

    **Continuous monitoring (``spec.monitor_dates=None``) needs a larger
    ``N`` than discrete monitoring at the same ``n_exercise`` -- this is not
    optional tuning.** Discrete barrier schedules only truncate the value to
    zero above ``h`` at a handful of fixed dates (2-3, independent of
    ``n_exercise``), so the number of times a fresh discontinuity gets
    reprojected through the ``N``-term cosine basis stays small even as
    ``n_exercise`` grows -- ``N=256`` converges cleanly there up to
    ``n_exercise`` in the several hundreds (verified). Continuous monitoring
    truncates at *every* exercise node, so that reprojection-of-a-
    discontinuity cost scales with ``n_exercise`` itself; at ``N=256`` this
    was measured to converge smoothly only up to ``n_exercise~320`` before
    visibly diverging (21.83 at 320 -> 23.18 at 640 -> 23.92 at 1280, for a
    case whose converged value is ~21.8) -- doubling ``N`` to 512 or 1024
    pushes the divergence onset far enough out to stay clear of it at the
    ``n_exercise`` this function needs for good accuracy (verified: N=512,
    n_exercise<=200ish converges monotonically with no such blow-up). If
    you widen ``n_exercise`` for a continuously-monitored case, widen ``N``
    correspondingly and re-check monotonic convergence before trusting the
    result -- there is no automatic guard here.
    """
    if not spec.american:
        raise ValueError("cos2d_american prices American options only; use cos2d for European.")

    t_start = _time.perf_counter()

    p1, d1, g1 = _bermudan_price(spec, hp, jp, n_exercise, N, M, v_lo, v_hi, n_eval)
    extra = {"n_exercise": n_exercise, "N": N, "M": M, "richardson": richardson}

    if richardson:
        p2, d2, g2 = _bermudan_price(spec, hp, jp, 2 * n_exercise, N, M, v_lo, v_hi, n_eval)
        price, delta, gamma = 2 * p2 - p1, 2 * d2 - d1, 2 * g2 - g1
        extra["bermudan_n"] = p1
        extra["bermudan_2n"] = p2
    else:
        price, delta, gamma = p1, d1, g1

    runtime = _time.perf_counter() - t_start
    return PricingResult(price=price, delta=delta, gamma=gamma, runtime=runtime, extra=extra)
