"""Diagnostic script for Section 5.2 of JAM_Paper.tex: quantifies the cost of
collapsing the Bermudan-COS engine's final cross-branch blend to a point
evaluation at v0 (see cos2d_american.py's module docstring and _bermudan_price),
and checks whether either of two natural remedies -- widening the blend window
to the full horizon, or widening the COS truncation domain to cover the
highest-variance branch -- recovers any accuracy.

Isolation case: S0=80, K=100, q=0, H removed (effectively infinite) so the
American and European prices are identical (a call with no dividends is never
optimally exercised early) -- this isolates the branch/blend architecture from
both the barrier-truncation machinery and the early-exercise correction. The
reference is the exact closed-form Heston European price via Gil-Pelaez
inversion (charfunc.heston1993_price), not another COS run.

Run from this directory: python blend_domain_diagnostics.py
"""
import numpy as np

from barrier_pricing.params import HestonParams, NO_JUMPS, OptionSpec
from barrier_pricing.charfunc import heston1993_price
from barrier_pricing.cos2d import _bound, _fclencurt, _transit_var_approx
from barrier_pricing.cos2d_american import (
    _phi_bs_jump_grid, _cos_sin_basis, _evaluate, _reproject, _intrinsic,
    _monitor_step_mask,
)

hp = HestonParams(kappa=2.00, theta=0.10, sigma=0.10, rho=-0.50, v0=0.10)
S0, K, H, r, q, T = 80.0, 100.0, 1.0e6, 0.03, 0.0, 0.5
N, M, n_exercise = 512, 200, 200
v_lo, v_hi = 0.001, 0.95


def _price_variant(blend: str, domain: str) -> float:
    """One Bermudan-COS run (no Richardson extrapolation -- both variants are
    compared at the same fixed n_exercise, so any 1/n bias cancels in the
    comparison) with the final blend and/or truncation domain swapped out.
    """
    spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=T, american=True, monitor_dates=None)
    dt = T / n_exercise
    x0 = np.log(S0 / K)
    h = np.log(H / K)

    xf, wf = _fclencurt(M, v_lo, v_hi)
    jf = M - 1
    while jf > 0 and hp.v0 > xf[jf]:
        jf -= 1
    jf = min(jf, M - 2)

    a, b = _bound(hp.v0, r, q, T, hp)
    if domain == "wide":
        a_hi, b_hi = _bound(v_hi, r, q, T, hp)
        a, b = min(a, a_hi), max(b, b_hi)

    karr = np.arange(N)
    u = karr * np.pi / (b - a)
    weight0 = np.where(karr == 0, 0.5, 1.0)[:, None]

    PhiFine = _phi_bs_jump_grid(u[:, None], xf[None, :], dt, r, q, NO_JUMPS)
    ReG, ImG = np.real(PhiFine), np.imag(PhiFine)
    disc = np.exp(-r * dt)

    n_eval = N
    xs = a + (np.arange(n_eval) + 0.5) * (b - a) / n_eval
    CosMat, SinMat = _cos_sin_basis(u, xs, a)
    intrinsic_grid = _intrinsic(xs)
    below_h = xs <= h

    barrier_mask = _monitor_step_mask(None, n_exercise, dt, T)

    terminal_values = np.where(below_h, intrinsic_grid, 0.0)
    vvk = _reproject(np.tile(terminal_values[:, None], (1, M)), CosMat, n_eval)

    for step in range(n_exercise - 1, 0, -1):
        values = disc * _evaluate(vvk, ReG, ImG, CosMat, SinMat, weight0)
        values = np.maximum(values, intrinsic_grid[:, None])
        if barrier_mask[step]:
            values[~below_h, :] = 0.0
        vvk = _reproject(values, CosMat, n_eval)

    phase = np.exp(1j * u[:, None] * (x0 - a))
    cont_price_branches = np.maximum(
        K * disc * np.sum(weight0 * np.real(PhiFine * phase) * vvk, axis=0),
        max(S0 - K, 0.0),
    )

    if blend == "point":
        order = np.argsort(xf)
        return float(np.interp(hp.v0, xf[order], cont_price_branches[order]))
    else:  # "wide": full-horizon transit density over the same M branch nodes
        tv = _transit_var_approx(hp.v0, xf, hp.sigma, hp.kappa, hp.theta, T, order=1)
        return float(np.sum(wf * cont_price_branches * tv))


if __name__ == "__main__":
    exact = heston1993_price(S0, K, r, q, T, hp, NO_JUMPS)
    baseline = _price_variant("point", "default")
    wide_blend = _price_variant("wide", "default")
    wide_domain = _price_variant("point", "wide")

    for name, price in (("exact (Gil-Pelaez)", exact),
                         ("baseline (point blend, default domain)", baseline),
                         ("wide blend (full-horizon transit density)", wide_blend),
                         ("wide domain (covers v_hi)", wide_domain)):
        diff = 100 * (price - exact) / exact
        print(f"{name:45s} price={price:.4f}  diff={diff:+.2f}%")
