# barrier_pricing

Python implementation of independent pricing engines for up-and-out barrier
call options under Heston stochastic volatility, optionally with
Merton/Bates log-normal jumps. Companion code to:

> M. Alfeus, *A Fourier--Finite-Difference Method for American Barrier
> Options under Stochastic Volatility* (2026).

Five independent engines for the same pricing problem, cross-validated
against each other, against Monte Carlo, and against the published
numerical examples of

> Chiarella, Kang & Meyer (2012), "The evaluation of barrier option prices
> under stochastic volatility", *Comput. Math. Appl.* 64, 2034-2048.

See `Replication_Report.pdf` for the full replication write-up (accuracy
tables, a documented grid-sensitivity investigation, the COS-American
engine's design and two numerical instabilities found and fixed while
building it, and a runtime/recommendation comparison across the engines).

## Engines

| Module | Method | Monitoring | Exercise | Notes |
|---|---|---|---|---|
| `mol.py` | Method of lines (Riccati transform, Numba-JIT per-line kernel) | continuous or discrete | European or American | price + delta + gamma all directly from the ODE system, no finite differencing |
| `cos2d.py` | 2D COS (Fourier-cosine, Clenshaw-Curtis over variance) | discrete only | European only | fastest by ~2 orders of magnitude where it applies |
| `cos2d_american.py` | Bermudan-COS (dense exercise grid + Richardson extrapolation, on top of `cos2d`'s frozen-variance-branch architecture) | continuous or discrete | American only | see `Replication_Report.pdf` Section 4 for accuracy (excellent ATM/ITM, a modest deep-OTM bias) and for why continuous monitoring needs a larger `N` than discrete at the same `n_exercise` |
| `cos2d_joint.py` | Fourier--finite-difference: a genuine joint $(X,v)$ recursion (Fourier-diagonalized cross term, implicit backward-Euler finite differences in variance, no frozen branches) | continuous or discrete | American only | the paper's main contribution -- removes `cos2d_american.py`'s deep-OTM bias almost completely, includes jumps, and comes with a proved frozen-coefficient stability bound and an error expansion that justifies Richardson extrapolation |
| `adi.py` | Alternating-direction-implicit (Hundsdorfer--Verwer, Ikonen--Toivanen splitting for American exercise) finite-difference solver on the full 2D $(S,v)$ grid | continuous or discrete | European or American | independent benchmark for `cos2d_joint.py`; shares no code with any other engine |
| `mc.py` | Monte Carlo (full-truncation Euler, Broadie-Glasserman-Kou continuity correction) | continuous or discrete | European only | independent cross-check; no shared code with the other engines |

`mol.py` is a from-scratch Python port of the accompanying Fortran
method-of-lines solver (Riccati transform / line-SOR) used in the paper.
`cos2d.py` is a from-scratch Python port of the accompanying MATLAB 2D-COS
pricer. Neither re-derives or "improves" the underlying numerical method,
only the implementation. `cos2d_american.py`, `cos2d_joint.py` and `adi.py`
are new numerical-methods work built for the paper.

## Installation

```bash
pip install -r requirements.txt
```

No `pyproject.toml`/`setup.py` yet -- run scripts from this directory (or
add it to `PYTHONPATH`) so `import barrier_pricing` resolves.

## Quick start

```python
from barrier_pricing.params import HestonParams, JumpParams, NO_JUMPS, OptionSpec

hp = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)

# Continuously-monitored European up-and-out call
from barrier_pricing import mol
spec = OptionSpec(S0=100.0, K=100.0, H=130.0, r=0.03, q=0.05, T=0.5,
                   american=False, knock_in=False, monitor_dates=None)
result = mol.price_barrier_call(spec, hp, NO_JUMPS, n_S=150, M=150, n_time_steps=300)
print(result.price, result.delta, result.gamma, result.runtime)

# American, continuously monitored, via the paper's joint (X,v) scheme
from barrier_pricing import cos2d_joint
spec_am = OptionSpec(S0=100.0, K=100.0, H=130.0, r=0.03, q=0.05, T=0.5,
                      american=True, knock_in=False, monitor_dates=None)
cos2d_joint.price_barrier_call(spec_am, hp, NO_JUMPS, n_steps=50, N=2048, M=100)

# Independent ADI cross-check of the same contract
from barrier_pricing import adi
adi.price_barrier_call(spec_am, hp, nS=400, nv=200, n_steps=800)

# With Merton/Bates log-normal jumps
jp = JumpParams(lam=0.5, gam=-0.05, del_=0.1)
cos2d_joint.price_barrier_call(spec_am, hp, jp, n_steps=50, N=2048, M=100)
```

All `price_*` functions return a `PricingResult` (`barrier_pricing/params.py`):
`price`, `delta`, `gamma`, `runtime`, and an engine-specific `extra` dict
(e.g. MC's `stderr`/`ci95`, MOL's SOR `regime`, `cos2d_joint`'s error
estimate and extrapolation levels).

### The `monitor_dates` convention (read this before using discrete monitoring)

`OptionSpec.monitor_dates` is `None` for continuous monitoring, or a list of
**time-to-maturity (tau = T - t)** values for discrete monitoring, one per
monitoring date. To monitor at calendar times `t_1 < ... < t_n` (measured
from today), pass `monitor_dates = [T - t_1, ..., T - t_n]`. For example,
"check at the midpoint and at maturity" for `T=0.5` is
`monitor_dates=[0.25, 0.0]`, **not** `[0.25, 0.5]`.

## Reproducing the paper's tables

```bash
python joint_convergence_study.py   # runs every benchmark/convergence/stability study (slow: ~1 hour)
python make_paper_tables.py         # turns joint_convergence_study_results.json into tables/*.tex
```

`joint_convergence_study.py` accepts a subset of section names as arguments
(see `SECTIONS` at the bottom of the file) to rerun only part of the study.
It writes incrementally to `joint_convergence_study_results.json`, so an
interrupted run can be resumed by re-invoking with the same or a smaller set
of sections.

The four standalone scripts below are permanent, runnable records of where
specific published numbers come from (project convention: no hand-transcribed
figures):

- `validate_cos_american.py` -- reproduces `Replication_Report.pdf`'s COS-American tables.
- `referee_response_validation.py` -- limiting cases, a jump-martingale check, and a stress sweep.
- `joint_recursion_validation.py` -- staged validation of `cos2d_joint.py` against closed-form Heston, MOL, and the frozen-branch COS engine.
- `blend_domain_diagnostics.py` -- isolates why the frozen-variance-branch architecture (`cos2d_american.py`) carries a deep-OTM bias that `cos2d_joint.py` removes by construction.

## Known limitations

- `cos2d.py` is European, discrete-monitoring, knock-out (or knock-in via
  parity) only -- raises `NotImplementedError` otherwise.
- `cos2d_american.py` matches `mol.py` to <1% at- and in-the-money
  ($S_0 \ge K$) but carries a real, roughly constant-*absolute* (~$0.11)
  bias deep out-of-the-money (~8% relative at $S_0=80$): collapsing the
  final cross-branch blend to a point evaluation at `v0` discards the
  convexity-driven variance-mixing a true frozen-branch blend would
  contribute (see `blend_domain_diagnostics.py` for a reproducible
  isolation test). `cos2d_joint.py` resolves this by construction (genuine
  joint $(X,v)$ mixing at every exercise date, not one final blend).
- `cos2d_joint.py`'s own exercise-boundary diagnostic
  (`price_barrier_call(..., return_boundary=True)`) is unreliable within a
  few percent of a continuously monitored barrier: the raw
  continuation-vs-intrinsic crossing drifts toward the barrier under time-step
  refinement instead of converging, rather than locating a genuine free
  boundary. `adi.py`'s `exercise_boundary()` (based on the Ikonen-Toivanen
  Lagrange multiplier, the correct LCP complementarity signal) does not have
  this problem and should be preferred for that diagnostic.
- `cos2d_american.py`'s continuous-monitoring mode needs `N` increased
  alongside `n_exercise` (unlike discrete monitoring, where `N=256` is fine
  into the hundreds of exercise dates) -- reprojecting the barrier's
  discontinuity at *every* exercise node makes the Gibbs-type reprojection
  error scale with `n_exercise` itself. No automatic guard -- see the module
  docstring.
- `mc.py` does not price American exercise (would need a
  Longstaff-Schwartz-type estimator) -- raises `NotImplementedError`.
- `mc.py`'s continuity correction is only meaningful for continuous
  monitoring; for discrete monitoring it degenerates to a negligible
  fine-grid correction, not a source of bias.
- No `tests/` yet (`barrier_pricing/tests/__init__.py` is a placeholder);
  validation so far is via the scripts listed above, reproducing published
  benchmark tables, not a checked-in regression suite.
- No `pyproject.toml`/`setup.py`.

## License

MIT -- see [`LICENSE`](LICENSE).
