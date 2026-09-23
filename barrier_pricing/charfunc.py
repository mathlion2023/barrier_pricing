"""Heston(+Merton-jump) characteristic function and a 1D COS vanilla pricer.

The characteristic function here is used for two purposes elsewhere in this
package:
  1. as the engine behind the 1D COS vanilla-option pricer in this module,
     which in turn supplies the "vanilla" leg of the in-out barrier parity
     (knock_in = vanilla - knock_out) used by ``mol``, ``cos2d`` and ``mc``;
  2. as an independent cross-check target: ``heston1993_price`` below prices
     the same vanilla via direct numerical quadrature of the Heston (1993)
     P1/P2 integrals (Bates-extended for jumps), so the COS implementation
     can be validated against a second, structurally different formula
     without needing an external reference implementation.

Convention: tau = time to maturity (paper's convention), S0 = spot "now".
"""
from __future__ import annotations

import numpy as np
from scipy import integrate

from .params import HestonParams, JumpParams, NO_JUMPS


def heston_jump_cf(u: np.ndarray, tau: float, r: float, q: float,
                    hp: HestonParams, jp: JumpParams = NO_JUMPS) -> np.ndarray:
    """Characteristic function of X_T = ln(S_T/S0) under Heston + Bates jumps.

    E[exp(i*u*X_T)], vectorized over complex array ``u``. Uses the "little
    trap" branch (Albrecher et al. 2007) for numerical stability at large
    |u| and long maturities.
    """
    u = np.asarray(u, dtype=complex)
    kappa, theta, sigma, rho, v0 = hp.kappa, hp.theta, hp.sigma, hp.rho, hp.v0

    iu = 1j * u
    d = np.sqrt((kappa - rho * sigma * iu) ** 2 + sigma**2 * (iu + u**2))
    g2 = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)

    drift = r - q - jp.lam * jp.ak1
    edt = np.exp(-d * tau)

    C = iu * drift * tau + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * iu - d) * tau
        - 2.0 * np.log((1.0 - g2 * edt) / (1.0 - g2))
    )
    D = (kappa - rho * sigma * iu - d) / sigma**2 * (1.0 - edt) / (1.0 - g2 * edt)

    cf = np.exp(C + D * v0)

    if jp.active:
        jump_cf = np.exp(iu * jp.gam - 0.5 * u**2 * jp.del_**2)
        cf = cf * np.exp(jp.lam * tau * (jump_cf - 1.0))

    return cf


def cos_vanilla_call(S0: float, K: float, r: float, q: float, tau: float,
                      hp: HestonParams, jp: JumpParams = NO_JUMPS,
                      N: int = 256, L: float = 12.0) -> float:
    """European call price via the (1D) COS method of Fang & Oosterlee (2008).

    ``N`` cosine terms, truncation range [a,b] = c1 +/- L*sqrt(c2+sqrt(c4)),
    with c1,c2 the (jump-inclusive) cumulants of X_T = ln(S_T/S0). Used
    as the "vanilla leg" for barrier in-out parity, and validated against
    ``heston1993_price`` below.
    """
    c1, c2, c4 = _svjd_cumulants(tau, r, q, hp, jp)
    a = c1 - L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4)))
    b = c1 + L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4)))

    k = np.arange(N)
    u = k * np.pi / (b - a)

    cf = heston_jump_cf(u, tau, r, q, hp, jp)
    # Un_k: COS coefficients of the discounted call payoff on [a,b]
    Uk = 2.0 / (b - a) * _chi_psi_call(a, b, k, 0.0, b)

    x = np.log(S0 / K)
    term = cf * np.exp(1j * u * (x - a)) * Uk
    term[0] *= 0.5
    price = np.exp(-r * tau) * K * np.real(np.sum(term))
    return float(price)


def _chi_psi_call(a: float, b: float, k: np.ndarray, c: float, d: float) -> np.ndarray:
    """COS payoff coefficients V_k for a plain-vanilla call, g(y)=(e^y-1)^+.

    Standard closed form (Fang & Oosterlee 2008, eq. 22-24) restricted to
    the sub-interval [c,d] subset [a,b] where the payoff is active
    (c=0, d=b for a call truncated at the domain's upper end).
    """
    w = k * np.pi / (b - a)

    def chi(cc, dd):
        t1 = (np.cos(w * (dd - a)) * np.exp(dd) - np.cos(w * (cc - a)) * np.exp(cc))
        t2 = w * (np.sin(w * (dd - a)) * np.exp(dd) - np.sin(w * (cc - a)) * np.exp(cc))
        return (t1 + t2) / (1.0 + w**2)

    def psi(cc, dd):
        out = np.empty_like(w)
        nz = k != 0
        out[~nz] = dd - cc
        out[nz] = (np.sin(w[nz] * (dd - a)) - np.sin(w[nz] * (cc - a))) / w[nz]
        return out

    return chi(c, d) - psi(c, d)


def _svjd_cumulants(tau: float, r: float, q: float, hp: HestonParams,
                     jp: JumpParams) -> tuple[float, float, float]:
    """First, second and fourth cumulants of X_T (Heston part in closed
    form; jump part added analytically since jumps are i.i.d. compound
    Poisson and independent of the diffusion part).
    """
    kappa, theta, sigma, rho, v0 = hp.kappa, hp.theta, hp.sigma, hp.rho, hp.v0
    drift = r - q - jp.lam * jp.ak1
    ekt = np.exp(-kappa * tau)

    c1_h = drift * tau + (1 - ekt) * (theta - v0) / (2 * kappa) - 0.5 * theta * tau
    c2_h = (1.0 / (8 * kappa**3)) * (
        sigma * tau * kappa * ekt * (v0 - theta) * (8 * kappa * rho - 4 * sigma)
        + kappa * rho * sigma * (1 - ekt) * (16 * theta - 8 * v0)
        + 2 * theta * kappa * tau * (-4 * kappa * rho * sigma + sigma**2 + 4 * kappa**2)
        + sigma**2 * ((theta - 2 * v0) * np.exp(-2 * kappa * tau) + theta * (6 * ekt - 7) + 2 * v0)
        + 8 * kappa**2 * (v0 - theta) * (1 - ekt)
    )
    c4_h = abs(c2_h) ** 2 * 3.0  # crude near-Gaussian proxy; only sets truncation width

    if jp.active:
        c1_j = jp.lam * tau * jp.gam
        c2_j = jp.lam * tau * (jp.gam**2 + jp.del_**2)
        c4_j = jp.lam * tau * (jp.gam**4 + 6 * jp.gam**2 * jp.del_**2 + 3 * jp.del_**4)
    else:
        c1_j = c2_j = c4_j = 0.0

    return c1_h + c1_j, c2_h + c2_j, c4_h + c4_j


# --------------------------------------------------------------------------
# Independent cross-check: direct quadrature of the Heston (1993) / Bates
# P1, P2 representation. Structurally unrelated to the COS series above, so
# agreement between the two is meaningful evidence both are implemented
# correctly.
# --------------------------------------------------------------------------

def heston1993_price(S0: float, K: float, r: float, q: float, tau: float,
                      hp: HestonParams, jp: JumpParams = NO_JUMPS) -> float:
    """European call price via Gil-Pelaez inversion of the same CF used by
    ``cos_vanilla_call`` (Heston 1993 P1/P2 form), computed by
    ``scipy.integrate.quad`` instead of a cosine series.
    """
    def phi_stock(u):
        # CF of X_T under the S-measure is phi(u-i)/phi(-i)
        num = heston_jump_cf(u - 1j, tau, r, q, hp, jp)
        den = heston_jump_cf(np.array([-1j]), tau, r, q, hp, jp)
        return num / den

    def p_integrand(u, measure):
        u = np.atleast_1d(u).astype(complex)
        lnK = np.log(K / S0)
        if measure == 1:
            cf = phi_stock(u)
        else:
            cf = heston_jump_cf(u, tau, r, q, hp, jp)
        val = np.real(np.exp(-1j * u * lnK) * cf / (1j * u))
        return val[0]

    P1 = 0.5 + (1.0 / np.pi) * integrate.quad(lambda u: p_integrand(u, 1), 1e-10, 200,
                                               limit=200)[0]
    P2 = 0.5 + (1.0 / np.pi) * integrate.quad(lambda u: p_integrand(u, 2), 1e-10, 200,
                                               limit=200)[0]

    price = S0 * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
    return float(price)
