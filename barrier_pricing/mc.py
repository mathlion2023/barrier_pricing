"""Monte Carlo cross-check for the MOL / COS barrier pricers.

Simulates the same Heston(+Merton-jump) dynamics used elsewhere in this
package and prices the up-and-out (or, via parity, up-and-in) call by
averaging discounted payoffs over simulated paths. This is deliberately a
*different* numerical method (no PDE, no Fourier series) so that
agreement with ``mol``/``cos2d`` is real evidence of correctness rather
than three implementations of the same mistake.

Variance is simulated with the "full truncation" Euler scheme (Lord,
Koekkoek & van Dijk, 2010): simple, unconditionally usable for any Feller
ratio, and accurate enough at the step counts used here once combined
with the Brownian-bridge continuity correction below for the barrier.

Continuous monitoring from a *discretely-stepped* simulation is a classic
source of bias (a path can poke through the barrier between two grid
times without either endpoint recording it): the standard fix (Broadie,
Glasserman & Kou 1997) is applied here -- given a step's two endpoints
both below H, the probability that a Brownian bridge between them
breached H in between has a closed form, and each path is knocked out
with that probability rather than only when a grid point lands on/above
H. This removes most of the discretization bias without needing very
small time steps. The correction is applied only under continuous
monitoring. Under discrete monitoring the contract observes S only at the
monitoring dates, so a path is knocked out if and only if S >= H at one of
those dates.
"""
from __future__ import annotations

import time as _time

import numpy as np

from .params import HestonParams, JumpParams, NO_JUMPS, OptionSpec, PricingResult


def _simulate_paths(spec: OptionSpec, hp: HestonParams, jp: JumpParams,
                     n_paths: int, n_steps: int, rng: np.random.Generator,
                     antithetic: bool):
    """Simulate log-spot and variance paths, and the per-step
    Brownian-bridge continuity-corrected knock-out indicator.

    Returns (logS, v, alive) where ``alive`` is a boolean array of shape
    (n_paths,) that is False for any path that touched/crossed H at any
    monitoring instant (continuous: every step; discrete: only at
    ``spec.monitor_dates``).
    """
    half = n_paths // 2 if antithetic else n_paths
    total = 2 * half if antithetic else n_paths

    dt = spec.T / n_steps
    sqdt = np.sqrt(dt)
    lnH = np.log(spec.H)

    logS = np.full(total, np.log(spec.S0))
    v = np.full(total, hp.v0)
    alive = np.ones(total, dtype=bool)

    if spec.is_continuous():
        monitor_steps = None  # every step
    else:
        # Round each monitoring tau to the nearest simulation step so the
        # continuity correction (below) is applied on exactly those steps.
        monitor_taus = spec.T - np.asarray(spec.monitor_dates)
        monitor_steps = set(np.round(monitor_taus / dt).astype(int).tolist())
        monitor_steps.add(n_steps)  # maturity is always a monitoring date (as in mol/cos engines)
        if not np.allclose(np.round(monitor_taus / dt) * dt, monitor_taus, atol=1e-9 * max(1.0, spec.T)):
            raise ValueError("discrete monitoring dates must be multiples of T/n_steps")

    drift_comp = jp.lam * jp.ak1

    for step in range(1, n_steps + 1):
        if antithetic:
            zv = rng.standard_normal(half)
            zv = np.concatenate([zv, -zv])
            zi = rng.standard_normal(half)
            zi = np.concatenate([zi, -zi])
        else:
            zv = rng.standard_normal(total)
            zi = rng.standard_normal(total)
        z1 = hp.rho * zv + np.sqrt(1.0 - hp.rho**2) * zi

        v_pos = np.maximum(v, 0.0)
        v_new = v + hp.kappa * (hp.theta - v_pos) * dt + hp.sigma * np.sqrt(v_pos) * sqdt * zv

        logS_new = (logS + (spec.r - spec.q - 0.5 * v_pos - drift_comp) * dt
                    + np.sqrt(v_pos * dt) * z1)

        if jp.active:
            n_jumps = rng.poisson(jp.lam * dt, size=total)
            has_jump = n_jumps > 0
            if np.any(has_jump):
                jump_size = rng.normal(jp.gam, jp.del_, size=total) * n_jumps
                logS_new = logS_new + np.where(has_jump, jump_size, 0.0)

        if monitor_steps is not None:
            # Discrete monitoring: the barrier is observed only at the
            # monitoring instant itself. No bridge correction here: a path
            # that crossed H between two simulation steps and came back
            # below it by the monitoring date is alive under the contract.
            # (Applying the continuous-monitoring bridge probability on
            # these steps, as an earlier version did, knocked such paths
            # out and biased discrete-barrier prices low.)
            if step in monitor_steps:
                alive &= logS_new < lnH
        else:
            both_below = (logS < lnH) & (logS_new < lnH)
            # Broadie-Glasserman-Kou continuity correction: bridge
            # touch probability for a (locally) Brownian path between
            # two sub-barrier endpoints, using this step's local variance.
            p_touch = np.zeros(total)
            var_step = v_pos * dt
            safe = both_below & (var_step > 0)
            p_touch[safe] = np.exp(
                -2.0 * (lnH - logS[safe]) * (lnH - logS_new[safe]) / var_step[safe])
            touched_grid = logS_new >= lnH
            touched_bridge = safe & (rng.random(total) < p_touch)
            alive &= ~(touched_grid | touched_bridge)

        logS, v = logS_new, v_new

    return logS, v, alive


def price_barrier_call_mc(spec: OptionSpec, hp: HestonParams, jp: JumpParams = NO_JUMPS,
                           n_paths: int = 200_000, n_steps: int = 100,
                           seed: int | None = None, antithetic: bool = True,
                           bump_frac: float = 0.01, greeks: bool = True) -> PricingResult:
    """Price by simulation; delta/gamma via central bumps of S0 replayed
    with the *same* random draws (common random numbers), which cancels
    most Monte Carlo noise between the bumped and base valuations --
    much tighter than independent-sample finite differences, though
    still noisier than the greeks MOL/COS give directly from the PDE/
    Fourier solution (barrier payoffs are discontinuous, so pathwise/
    likelihood-ratio estimators would be needed to match that -- out of
    scope here; this package's MOL delta/gamma are the accurate ones,
    this module exists to cross-check the *price*).

    Note: American (early-exercise) barrier options are not priced by
    this simulator -- that needs a regression-based (Longstaff-Schwartz)
    scheme, a materially different estimator not implemented here.
    """
    if spec.american:
        raise NotImplementedError(
            "Monte Carlo pricing here covers European knock-out/knock-in only; "
            "American barrier options need a Longstaff-Schwartz-type estimator.")

    t_start = _time.perf_counter()
    seed_seq = np.random.SeedSequence(seed)
    common_state = np.random.default_rng(seed_seq).bit_generator.state
    h = bump_frac * spec.S0

    def value_at(S0_):
        s = OptionSpec(S0=S0_, K=spec.K, H=spec.H, r=spec.r, q=spec.q, T=spec.T,
                        american=False, knock_in=False, monitor_dates=spec.monitor_dates)
        rng_local = np.random.default_rng()
        rng_local.bit_generator.state = common_state  # same draws for every bump
        logS, _, alive = _simulate_paths(s, hp, jp, n_paths, n_steps, rng_local, antithetic)
        payoff = np.where(alive, np.maximum(np.exp(logS) - s.K, 0.0), 0.0)
        disc = np.exp(-spec.r * spec.T) * payoff
        return disc

    disc0 = value_at(spec.S0)
    price = float(disc0.mean())
    stderr = float(disc0.std(ddof=1) / np.sqrt(len(disc0)))
    if greeks:
        disc_up = value_at(spec.S0 + h)
        disc_dn = value_at(spec.S0 - h)
        delta = float((disc_up.mean() - disc_dn.mean()) / (2 * h))
        gamma = float((disc_up.mean() - 2 * disc0.mean() + disc_dn.mean()) / h**2)
    else:  # price only (e.g. benchmark tables): skips the two bumped revaluations
        delta = gamma = float("nan")

    if spec.knock_in:
        from .charfunc import cos_vanilla_call
        vanilla = cos_vanilla_call(spec.S0, spec.K, spec.r, spec.q, spec.T, hp, jp)
        vanilla_up = cos_vanilla_call(spec.S0 + h, spec.K, spec.r, spec.q, spec.T, hp, jp)
        vanilla_dn = cos_vanilla_call(spec.S0 - h, spec.K, spec.r, spec.q, spec.T, hp, jp)
        price = vanilla - price
        delta = (vanilla_up - vanilla_dn) / (2 * h) - delta
        gamma = (vanilla_up - 2 * vanilla + vanilla_dn) / h**2 - gamma

    runtime = _time.perf_counter() - t_start
    return PricingResult(price=price, delta=delta, gamma=gamma, runtime=runtime,
                          extra={"stderr": stderr, "ci95": 1.96 * stderr})
