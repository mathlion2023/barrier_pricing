"""2D COS (Fourier-cosine) pricer for discretely-monitored up-and-out barrier
options under Heston stochastic volatility (+ optional Merton/Bates
log-normal jumps), ported from ``Matlab Codes/Heston/SVJBarrier.m`` /
``SVJBarrier_T.m`` (function-for-function: ``fclencurt``, ``bound``, ``Gk``,
``phi_heston``/``phi_svjd``, ``mj``/``ms``/``mc``, ``ub_svjd``/``us_svjd``/
``uc_svjd``, ``Cx1x2_svjd``, ``con_svjd``/``con_svjd_delta``,
``transit_var_approx``).

**The scheme, in one paragraph.** Heston is Markov in (x, v) with x =
ln(S/K), so the discretely-monitored barrier problem could in principle be
solved by a 1D COS recursion in x alone if v were known at every monitoring
date -- but v itself is a random path. This method sidesteps that by
discretizing v on M Clenshaw-Curtis quadrature nodes ``xf`` and, for *each*
node independently, running the standard Fang-Oosterlee discretely-monitored
COS recursion in x *as if* v stayed frozen at that node for the entire
remaining life of the option (``Cx1x2_svjd``, called mt-1 times, one per
monitoring sub-interval). Only at the very last sub-interval (the one ending
"now", handled by ``con_svjd``/``con_svjd_delta`` rather than one more
``Cx1x2_svjd`` step) is the *true* transition density of v from today's v0 to
each quadrature node vf[j] used to average the mt-1 "frozen-v" COS
reconstructions into a single price -- via a small-dt Bessel/CIR density
approximation, ``transit_var_approx``. This is a faithful, if unusual,
resolution of the v-dimension; it is not re-derived or improved on here, only
ported (see CLAUDE.md: fidelity to the reference implementation is the
priority, not the numerical-analysis choices made 15 years ago by its
authors). Cross-check against ``mol``/``mc`` (structurally unrelated methods)
is what actually validates it, not this docstring.

**European, discretely-monitored, knock-out only.** No early exercise
(``con_svjd``/``Cx1x2_svjd`` have no free-boundary logic at all) and no
continuous-monitoring mode (the domain restriction to (0,H) only happens at
the mt-1 ``Cx1x2_svjd`` steps, i.e. only ever at discrete instants) --
``spec.american`` and ``spec.is_continuous()`` are both rejected outright.
Use ``mol`` for those cases; use ``spec.knock_in`` (via the same vanilla
COS-parity leg ``mol``/``mc`` already use, ``charfunc.cos_vanilla_call``) for
up-and-in.

**Jump convention.** The same as every other engine in this package
(``params.JumpParams``): J ~ N(gam, del^2), compensator
lam*(exp(gam + del^2/2) - 1). The MATLAB original (``phi_svjd.m``) uses the
mean-shifted location gam_legacy = gam + del^2/2 and no drift compensator.
The compensator is required for phi(-i;T) = exp((r-q)T), i.e. for the
discounted spot price to be a martingale. Legacy (e.g. MATLAB-calibrated)
parameters must be converted with ``JumpParams.from_legacy_matlab``.

**Optimizations over the MATLAB original** (behavior-preserving, not
approximations): (1) ``phi_svjd`` is evaluated on the full (N, M) frequency x
variance-node grid *once* per pricing call via NumPy broadcasting, then
reused for every one of the mt-1 backward steps *and* the final
reconstruction -- the original recomputes it from scratch inside every
single call to ``Cx1x2_svjd``/``con_svjd``/``con_svjd_delta`` (worth little
at mt=2, as hardcoded in ``SVJBarrier.m``'s own demo, but this port supports
arbitrary mt, e.g. the paper's weekly-monitoring examples, mt~26, where the
saving is an ~26x reduction in the dominant cost). (2) the ``ms``/``mc``
FFT-convolution kernels in ``Cx1x2_svjd`` depend only on the (fixed) barrier
level and domain, never on the current step's coefficients -- they and their
FFTs are hoisted out of the mt-1 loop and computed once, instead of being
rebuilt (and re-transformed) on every step as the MATLAB code does. (3) The
``mj``-derived kernels are computed as their true shape, a length-2N vector,
rather than as an (2N, M) matrix of M identical columns as ``ms.m``/``mc.m``
literally construct it (harmless in MATLAB's own vectorized ``fft``, but a
straight port to NumPy would otherwise waste an M-fold-redundant FFT).
"""
from __future__ import annotations

import time as _time

import numpy as np

from .charfunc import _chi_psi_call, cos_vanilla_call
from .params import HestonParams, JumpParams, NO_JUMPS, OptionSpec, PricingResult


# --------------------------------------------------------------------------
# fclencurt.m -- fast Clenshaw-Curtis quadrature nodes/weights on [a,b] via a
# single FFT (von Winckel, 2005). n1 points.
# --------------------------------------------------------------------------

def _fclencurt(n1: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    n = n1 - 1
    bma = b - a
    c = np.zeros((n1, 2))
    idx_odd = np.arange(0, n1, 2)
    k = np.arange(2, n + 1, 2, dtype=float)
    c[idx_odd, 0] = 2.0 / np.concatenate([[1.0], 1.0 - k**2])
    c[1, 1] = 1.0

    top = c[:n1, :]
    bottom = c[n - 1:0:-1, :]
    f = np.real(np.fft.ifft(np.vstack([top, bottom]), axis=0))

    w = bma * np.concatenate([[f[0, 0]], 2.0 * f[1:n, 0], [f[n1 - 1, 0]]]) / 2.0
    x = 0.5 * ((b + a) + n * bma * f[:n1, 1])
    return x, w


# --------------------------------------------------------------------------
# bound.m -- COS truncation range [a,b] for the log-price domain, from the
# Heston-only (no-jump) cumulants of one monitoring sub-interval dt.
# --------------------------------------------------------------------------

def _bound(v: float, r: float, q: float, dt: float, hp: HestonParams) -> tuple[float, float]:
    kappa, theta, sigma, rho = hp.kappa, hp.theta, hp.sigma, hp.rho
    ekt = np.exp(-kappa * dt)
    c1 = (r - q) * dt + (1 - ekt) * (theta - v) / (2 * kappa) - 0.5 * theta * dt
    c2 = (sigma * dt * kappa * ekt * (v - theta) * (8 * kappa * rho - 4 * sigma)
          + kappa * rho * sigma * (1 - ekt) * (16 * theta - 8 * v)
          + 2 * theta * kappa * dt * (-4 * kappa * rho * sigma + sigma**2 + 4 * kappa**2)
          + sigma**2 * ((theta - 2 * v) * np.exp(-2 * kappa * dt) + theta * (6 * ekt - 7) + 2 * v)
          + 8 * kappa**2 * (v - theta) * (1 - ekt)) / (8 * kappa**3)
    width = 12.0 * np.sqrt(abs(c2))
    return c1 - width, c1 + width


# --------------------------------------------------------------------------
# Gk.m (alp=1, K=1 branch -- the only one SVJBarrier.m ever uses): COS
# coefficients of a call payoff (e^y-1)^+ restricted to [x1,x2]. Reuses
# charfunc._chi_psi_call, which implements the identical kai()-psi_cos()
# formula (see module docstring of charfunc.py).
# --------------------------------------------------------------------------

def _cos_payoff_coeffs(a: float, b: float, N: int, x1: float, x2: float) -> np.ndarray:
    k = np.arange(N)
    return 2.0 / (b - a) * _chi_psi_call(a, b, k, x1, x2)


# --------------------------------------------------------------------------
# phi_heston.m / phi_svjd.m, vectorized over an (N, M) frequency x
# variance-node grid. ``v`` here is a free parameter (the frozen/conditioning
# variance for this dt-step, see module docstring) -- NOT hp.v0.
# --------------------------------------------------------------------------

def _phi_heston_grid(xi: np.ndarray, v: np.ndarray, dt: float, r: float, q: float,
                      hp: HestonParams) -> np.ndarray:
    kappa, theta, sigma, rho = hp.kappa, hp.theta, hp.sigma, hp.rho
    iu = 1j * xi
    D = np.sqrt((kappa - rho * sigma * iu) ** 2 + (xi**2 + iu) * sigma * sigma)
    G = (kappa - rho * sigma * iu - D) / (kappa - rho * sigma * iu + D)
    edt = np.exp(-D * dt)
    return (np.exp(iu * (r - q) * dt
                    + (v / sigma**2) * ((1.0 - edt) / (1.0 - G * edt)) * (kappa - rho * sigma * iu - D))
            * np.exp((kappa * theta / sigma**2)
                     * (dt * (kappa - rho * sigma * iu - D) - 2.0 * np.log((1.0 - G * edt) / (1.0 - G)))))


def _phi_svjd_grid(xi: np.ndarray, v: np.ndarray, dt: float, r: float, q: float,
                    hp: HestonParams, jp: JumpParams) -> np.ndarray:
    """Heston(+jump) CF over dt from variance v, package jump convention
    J ~ N(gam, del^2) with compensator lam*(E[e^J]-1) = lam*jp.ak1, so that
    phi(-i; dt) = exp((r-q) dt). Identical, term for term, to
    ``charfunc.heston_jump_cf``'s jump factor.
    """
    phi = _phi_heston_grid(xi, v, dt, r, q, hp)
    if jp.active:
        jump = np.exp(-0.5 * jp.del_**2 * xi**2 + 1j * jp.gam * xi)
        phi = phi * np.exp(dt * jp.lam * (jump - 1.0)) * np.exp(-1j * xi * dt * jp.lam * jp.ak1)
    return phi


# --------------------------------------------------------------------------
# mj.m / ms.m / mc.m -- FFT-convolution kernels for the discretely-monitored
# COS recursion (Fang & Oosterlee 2009). x1, x2 are scalars here (the
# barrier-restricted sub-domain (a, h) never depends on the variance node --
# see module docstring, optimization (3)).
# --------------------------------------------------------------------------

def _mj(j: np.ndarray, a: float, b: float, x1: float, x2: float) -> np.ndarray:
    j = np.asarray(j, dtype=float)
    out = np.empty(j.shape, dtype=complex)
    zero = j == 0
    out[zero] = (x2 - x1) * np.pi * 1j / (b - a)
    nz = ~zero
    jz = j[nz]
    out[nz] = (np.exp(1j * jz * (x2 - a) * np.pi / (b - a))
               - np.exp(1j * jz * (x1 - a) * np.pi / (b - a))) / jz
    return out


def _ms_kernel(a: float, b: float, x1: float, x2: float, N: int) -> np.ndarray:
    freq = np.concatenate([-np.arange(N), [0.0], 2 * N - np.arange(N + 1, 2 * N)])
    m = _mj(freq, a, b, x1, x2)
    m[N] = 0.0  # ms.m explicitly zeroes this row regardless of mj(...,0)
    return m


def _mc_kernel(a: float, b: float, x1: float, x2: float, N: int) -> np.ndarray:
    freq = 2 * N - np.arange(2 * N) - 1
    return _mj(freq, a, b, x1, x2)


# --------------------------------------------------------------------------
# ub_svjd.m / us_svjd.m / uc_svjd.m / Cx1x2_svjd.m -- one backward step of
# the discretely-monitored recursion, propagating COS coefficients vvk
# (shape (N, M), one column per variance quadrature node) back across a
# single monitoring interval dt.
# --------------------------------------------------------------------------

def _cx1x2_step(Phi: np.ndarray, vvk: np.ndarray, N: int, dt: float, r: float,
                 ms_fft: np.ndarray, mc_fft: np.ndarray) -> np.ndarray:
    M = vvk.shape[1]
    ub = Phi * vvk
    ub[0, :] *= 0.5

    us = np.zeros((2 * N, M), dtype=complex)
    us[:N, :] = ub
    uc = np.zeros((2 * N, M), dtype=complex)
    uc[N:, :] = ub

    conv_s = np.fft.ifft(np.fft.fft(us, axis=0) * ms_fft[:, None], axis=0)
    conv_c = np.fft.ifft(np.fft.fft(uc, axis=0) * mc_fft[:, None], axis=0)

    return np.exp(-r * dt) * np.imag(np.flipud(conv_c[:N, :]) + conv_s[:N, :]) / np.pi


# --------------------------------------------------------------------------
# transit_var_approx.m -- small-dt Bessel/CIR transition-density
# approximation for the variance process, orders 0-3 (con_svjd uses order 1,
# con_svjd_delta uses order 3 -- both ported unchanged, an asymmetry inherited
# from the reference implementation rather than introduced here).
# --------------------------------------------------------------------------

def _transit_var_approx(v0: float, v1: np.ndarray, sigma: float, kappa: float,
                         theta: float, dt: float, order: int) -> np.ndarray:
    y0 = 2.0 * np.sqrt(v0) / sigma
    y1 = 2.0 * np.sqrt(v1) / sigma
    q = 2.0 * kappa * theta / sigma**2

    p0 = (np.exp(-(y1 - y0) ** 2 / (2.0 * dt) - y1**2 * kappa / 4.0 + y0**2 * kappa / 4.0)
          * y1 ** (-0.5 + q) * y0 ** (0.5 - q) / np.sqrt(dt * 2.0 * np.pi))
    p0 = p0 / (sigma * np.sqrt(v0))

    if order == 0:
        return p0

    c1 = -(48 * theta**2 * kappa**2 - 48 * theta * kappa * sigma**2 + 9.0 * sigma**4
           + y1 * kappa**2 * sigma**2 * (-24 * theta + y1**2 * sigma**2) * y0
           + y1**2 * kappa**2 * sigma**4 * y0**2
           + y1 * kappa**2 * sigma**4 * y0**3) / (24.0 * y1 * y0 * sigma**4)

    if order == 1:
        return p0 * (1 + c1 * dt)

    if order == 2:
        alpha, y = theta, y1
        c2 = (9 * (256 * alpha**4 * kappa**4 - 512 * alpha**3 * kappa**3 * sigma**2
                   + 224 * alpha**2 * kappa**2 * sigma**4 + 32 * alpha * kappa * sigma**6 - 15 * sigma**8)
              + 6 * y * kappa**2 * sigma**2 * (-24 * alpha + y**2 * sigma**2)
                  * (16 * alpha**2 * kappa**2 - 16 * alpha * kappa * sigma**2 + 3 * sigma**4) * y0
              + y**2 * kappa**2 * sigma**4 * (672 * alpha**2 * kappa**2
                  - 48 * alpha * kappa * (2 + y**2 * kappa) * sigma**2 + (-6 + y**4 * kappa**2) * sigma**4) * y0**2
              + 2 * y * kappa**2 * sigma**4 * (48 * alpha**2 * kappa**2
                  - 24 * alpha * kappa * (2 + y**2 * kappa) * sigma**2 + (9 + y**4 * kappa**2) * sigma**4) * y0**3
              + 3 * y**2 * kappa**4 * sigma**6 * (-16 * alpha + y**2 * sigma**2) * y0**4
              + 2 * y**3 * kappa**4 * sigma**8 * y0**5
              + y**2 * kappa**4 * sigma**8 * y0**6) / (576.0 * y**2 * y0**2 * sigma**8)
        return p0 * (1 + c1 * dt + c2 * dt * dt * 0.5)

    # order == 3: plain Gaussian (Euler) approximation of the CIR transition
    return (np.exp(-(v1 - (v0 + kappa * (theta - v0) * dt)) ** 2 / (2.0 * sigma**2 * dt * v0))
            / (np.sqrt(2.0 * np.pi * dt * v0) * sigma))


# --------------------------------------------------------------------------
# con_svjd.m / con_svjd_delta.m -- final step: reconstruct the COS series at
# the actual (x, vf[j]) for every quadrature node j (``factor``=1 for price,
# ``1j*u`` for delta -- see ``price_barrier_call`` for the analytic gamma
# extension, ``factor=-u**2``, which has no MATLAB counterpart), then
# integrate over vf[j] against the v0->vf[j] transition density and the
# Clenshaw-Curtis weights.
# --------------------------------------------------------------------------

def _reconstruct_pht(x: float, a: float, N: int, u: np.ndarray, Phi_full: np.ndarray,
                      vvk: np.ndarray, factor: np.ndarray) -> np.ndarray:
    weight0 = np.where(np.arange(N) == 0, 0.5, 1.0)[:, None]
    phase = np.exp(1j * u[:, None] * (x - a)) * factor[:, None]
    return np.sum(weight0 * np.real(Phi_full * phase) * vvk, axis=0)


def _con_svjd(x: float, v: float, a: float, N: int, xf: np.ndarray, wf: np.ndarray,
              u: np.ndarray, Phi_full: np.ndarray, vvk: np.ndarray, dt: float, r: float,
              hp: HestonParams, factor: np.ndarray, transit_order: int) -> float:
    pht = _reconstruct_pht(x, a, N, u, Phi_full, vvk, factor)
    tv = _transit_var_approx(v, xf, hp.sigma, hp.kappa, hp.theta, dt, transit_order)
    return float(np.exp(-r * dt) * np.sum(wf * pht * tv))


# --------------------------------------------------------------------------
# Driver, mirroring SVJBarrier_T.m / mol.price_barrier_call's calling style.
# --------------------------------------------------------------------------

def price_barrier_call(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                        N: int = 2**10, M: int = 200, v_lo: float = 0.001,
                        v_hi: float = 0.95) -> PricingResult:
    """Price a discretely-monitored up-and-out (or, via ``spec.knock_in``,
    up-and-in) call under Heston(+jumps) via the 2D COS method.

    ``N``, ``M`` mirror SVJBarrier.m's hardcoded ``N=2^10, M=200``.
    ``spec.monitor_dates`` must give ``mt`` *uniformly spaced* monitoring
    dates covering (0, T] in steps of ``dt=T/mt`` (matches SVJBarrier_T.m's
    own ``dt=T/mt`` assumption, used for every backward step) -- irregular
    schedules aren't supported by this port (or the reference).
    """
    if spec.american:
        raise NotImplementedError("cos2d prices European barrier options only; use mol for American.")
    if spec.is_continuous():
        raise NotImplementedError("cos2d prices discretely-monitored options only; use mol for continuous monitoring.")

    t_start = _time.perf_counter()

    calendar_times = np.array(sorted(spec.T - np.asarray(spec.monitor_dates)))
    mt = len(calendar_times)
    dt = spec.T / mt
    if not np.allclose(calendar_times, dt * np.arange(1, mt + 1), atol=1e-9 * max(1.0, spec.T)):
        raise ValueError("cos2d requires monitor_dates uniformly spaced so calendar "
                          "monitoring times are {dt, 2*dt, ..., T} (SVJBarrier_T.m's dt=T/mt)")

    x = np.log(spec.S0 / spec.K)
    h = np.log(spec.H / spec.K)

    xf, wf = _fclencurt(M, v_lo, v_hi)

    jf = M - 1
    while jf > 0 and hp.v0 > xf[jf]:
        jf -= 1
    jf = min(jf, M - 2)

    a, b = _bound(hp.v0, spec.r, spec.q, dt, hp)

    karr = np.arange(N)
    u = karr * np.pi / (b - a)
    Phi = _phi_svjd_grid(u[:, None], xf[None, :], dt, spec.r, spec.q, hp, jp)  # (N, M)

    ms_fft = np.fft.fft(_ms_kernel(a, b, a, h, N))
    mc_fft = np.fft.fft(_mc_kernel(a, b, a, h, N))

    g = _cos_payoff_coeffs(a, b, N, 0.0, h)
    vvk = np.tile(g[:, None], (1, M))

    for _ in range(mt - 1):
        vvk = _cx1x2_step(Phi, vvk, N, dt, spec.r, ms_fft, mc_fft)

    ones_ = np.ones(N)
    iu = 1j * u
    neg_u2 = -(u**2)

    def _pair(factor, transit_order):
        hi = _con_svjd(x, xf[jf], a, N, xf, wf, u, Phi, vvk, dt, spec.r, hp, factor, transit_order)
        lo = _con_svjd(x, xf[jf + 1], a, N, xf, wf, u, Phi, vvk, dt, spec.r, hp, factor, transit_order)
        w1 = (hp.v0 - xf[jf]) / (xf[jf + 1] - xf[jf])
        return (hi - lo) * w1 + lo

    price = spec.K * _pair(ones_, 1)
    dprice_dx = spec.K * _pair(iu, 3)          # con_svjd_delta.m's exact formula
    d2price_dx2 = spec.K * _pair(neg_u2, 3)    # analytic extension: no MATLAB counterpart

    delta = dprice_dx / spec.S0
    gamma = (d2price_dx2 - dprice_dx) / spec.S0**2

    if spec.knock_in:
        vanilla = cos_vanilla_call(spec.S0, spec.K, spec.r, spec.q, spec.T, hp, jp)
        hbump = 1e-2 * spec.S0
        up = cos_vanilla_call(spec.S0 + hbump, spec.K, spec.r, spec.q, spec.T, hp, jp)
        dn = cos_vanilla_call(spec.S0 - hbump, spec.K, spec.r, spec.q, spec.T, hp, jp)
        v_delta = (up - dn) / (2 * hbump)
        v_gamma = (up - 2 * vanilla + dn) / hbump**2
        price, delta, gamma = vanilla - price, v_delta - delta, v_gamma - gamma

    runtime = _time.perf_counter() - t_start
    return PricingResult(price=price, delta=delta, gamma=gamma, runtime=runtime,
                          extra={"mt": mt, "N": N, "M": M})
