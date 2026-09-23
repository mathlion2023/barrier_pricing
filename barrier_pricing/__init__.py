"""Barrier option pricing under Heston stochastic volatility (+ optional Merton jumps).

Three independent pricing engines, cross-validated against each other and
against Chiarella, Kang & Meyer (2012), Comput. Math. Appl. 64, 2034-2048:

- ``mol``    Method-of-lines PDE solver (Riccati transform). Continuous or
             discrete monitoring, European or American, knock-out or
             knock-in (via in-out parity), price+delta+gamma in one pass.
- ``cos2d``  2D Fourier-cosine (COS) expansion pricer, ported from
             ``Matlab Codes/Heston``. Knock-out barrier, European only.
- ``mc``     Monte Carlo simulator (QE scheme for the variance process),
             discrete or continuity-corrected continuous monitoring,
             used as an independent cross-check of the above.
"""
