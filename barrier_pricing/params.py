"""Shared parameter containers for the Heston(+Merton-jump) barrier pricers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class HestonParams:
    """Heston (1993) stochastic-volatility parameters, risk-neutral measure.

    dS = (r-q) S dt + sqrt(v) S dZ1
    dv = kappa*(theta-v) dt + sigma*sqrt(v) dZ2,  corr(dZ1,dZ2) = rho
    """

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def feller_ratio(self) -> float:
        """2*kappa*theta / sigma**2 ; >= 1 satisfies the Feller condition."""
        return 2.0 * self.kappa * self.theta / self.sigma**2


@dataclass(frozen=True)
class JumpParams:
    """Merton/Bates log-normal jumps in the log-price, intensity ``lam``.

    The single convention used by every engine in this package: jump size
    J ~ N(gam, del**2) in log-price, characteristic exponent
    lam*(exp(i u gam - del**2 u**2/2) - 1), and ak1 = E[e^J] - 1 =
    exp(gam + del**2/2) - 1 is the risk-neutral drift compensator.

    The legacy MATLAB code (``phi_svjd.m``, and hence the calibrations in
    ``CalibParams_SVJD.mat``) parameterizes the same family by the
    *mean-shifted* location gam_legacy = gam + del**2/2 (so that
    E[e^J] = exp(gam_legacy)). Use ``JumpParams.from_legacy_matlab`` to map
    such parameters; do not pass them directly.
    """

    lam: float = 0.0
    gam: float = 0.0
    del_: float = 1e-8  # avoid div-by-zero when lam == 0; has no effect then

    @classmethod
    def from_legacy_matlab(cls, lam: float, gam_legacy: float, del_: float) -> "JumpParams":
        """Map the MATLAB ``phi_svjd.m`` parameterization
        J ~ N(gam_legacy - del^2/2, del^2) onto this package's convention."""
        return cls(lam=lam, gam=gam_legacy - 0.5 * del_**2, del_=del_)

    @property
    def legacy_gam(self) -> float:
        """gam in the MATLAB mean-shifted parameterization (inverse map)."""
        return self.gam + 0.5 * self.del_**2

    @property
    def ak1(self) -> float:
        import numpy as np

        return float(np.exp(self.gam + 0.5 * self.del_**2) - 1.0)

    @property
    def active(self) -> bool:
        return self.lam != 0.0


NO_JUMPS = JumpParams(lam=0.0, gam=0.0, del_=1e-8)


@dataclass(frozen=True)
class OptionSpec:
    """Contract terms for an up-and-out (or up-and-in, via parity) barrier call.

    monitor_dates: sorted times-to-maturity (tau = T - t, tau=0 at expiry,
    tau=T "now") at which the barrier is observed. ``None`` means
    continuous monitoring over the whole life. An empty/explicit list
    gives discrete monitoring at exactly those tau values.
    """

    S0: float
    K: float
    H: float
    r: float
    q: float
    T: float
    american: bool = False
    knock_in: bool = False
    monitor_dates: Sequence[float] | None = None  # None => continuous

    def is_continuous(self) -> bool:
        return self.monitor_dates is None


@dataclass(frozen=True)
class PricingResult:
    price: float
    delta: float
    gamma: float
    runtime: float
    extra: dict = field(default_factory=dict)
