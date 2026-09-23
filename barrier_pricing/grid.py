"""Spatial grids for the method-of-lines solver.

The spot grid is concentrated near the strike using a sinh (Tavella-Randall
style) stretching so that the region where the payoff kinks gets the most
resolution without needing the three-piece linear mesh the original Fortran
code used. The variance grid is uniform, as in the paper (Section 3).
"""
from __future__ import annotations

import numpy as np


def make_S_grid(S_floor: float, S_upper: float, n: int, concentrate_at: float,
                 stretch: float) -> np.ndarray:
    """Monotone sinh-stretched grid on [S_floor, S_upper], n points.

    ``stretch`` controls how tightly points cluster around
    ``concentrate_at``: smaller -> tighter clustering. A good default is
    ~10-25% of the domain width.
    """
    xi = np.linspace(0.0, 1.0, n)
    c_lo = np.arcsinh((S_floor - concentrate_at) / stretch)
    c_hi = np.arcsinh((S_upper - concentrate_at) / stretch)
    S = concentrate_at + stretch * np.sinh(c_hi * xi + c_lo * (1.0 - xi))
    # Clean up any fp round-off so the endpoints are exact.
    S[0] = S_floor
    S[-1] = S_upper
    return S


def make_dual_S_grid(S_floor: float, S_upper: float, n: int, k1: float, k2: float,
                      stretch: float) -> np.ndarray:
    """Sinh-stretched grid concentrated near the strike (``k1``), *with a
    local refinement band added near the barrier* (``k2``).

    A single concentration point badly under-resolves whichever of the
    payoff kink (at the strike) or the knock-out wall (at the barrier)
    it isn't centred on -- the option value develops a sharp boundary
    layer approaching the barrier that a strike-centred grid alone
    cannot see. The tempting fix -- build two independent full-domain
    sinh grids (one per centre) and merge them -- backfires: each grid
    is sparse away from its own centre, so the merge alternates dense
    and sparse points and produces wildly uneven spacing (neighbouring
    gaps varying by two orders of magnitude), which is its own source
    of numerical trouble in the finite-difference sweeps. Instead, only
    ``k2``'s neighbourhood gets extra, purely *local* points -- inserted
    into gaps that already exist in the strike-centred grid, not laid
    over the whole domain -- so spacing stays smooth everywhere.
    """
    n_local = max(6, n // 4)
    base = make_S_grid(S_floor, S_upper, n - n_local, k1, stretch)

    # Local refinement: a small sinh-grid of its own, concentrated at k2,
    # confined to the single [lo, hi] cell of `base` that already
    # contains k2. Because it only fills in a gap that was already
    # there, it cannot introduce a spacing jump larger than the base
    # grid's own local resolution.
    idx = np.clip(np.searchsorted(base, k2), 1, len(base) - 1)
    lo, hi = base[idx - 1], base[idx]
    local_stretch = 0.1 * (hi - lo)
    extra = make_S_grid(lo, hi, n_local + 2, k2, local_stretch)[1:-1]

    merged = np.unique(np.concatenate([base, extra]))
    return merged


def make_v_grid(v_max: float, M: int) -> tuple[np.ndarray, float]:
    """Uniform variance grid v_0=0, ..., v_M=v_max (M+1 points)."""
    v = np.linspace(0.0, v_max, M + 1)
    dv = v_max / M
    return v, dv


def remesh_2d(S_old: np.ndarray, S_new: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Linearly interpolate each row of ``C`` (shape (M+1, len(S_old))) from
    ``S_old`` onto ``S_new``, vectorized across all variance lines at once.

    Needed when a discretely-monitored option's domain switches between
    (0, H) at a monitoring instant and (0, S_max) elsewhere (paper Sec.
    2.2): the whole (v, S) price/delta surface must be carried across onto
    the new spot grid before the next time step.
    """
    idx = np.searchsorted(S_old, S_new, side="right") - 1
    idx = np.clip(idx, 0, len(S_old) - 2)
    s0, s1 = S_old[idx], S_old[idx + 1]
    w = (S_new - s0) / (s1 - s0)
    return C[:, idx] * (1.0 - w) + C[:, idx + 1] * w
