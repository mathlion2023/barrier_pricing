"""Reproduces every COS-American number in Replication_Report.pdf's Tables
3, 5, and the jump table (Section 5). Run from this directory:

    python validate_cos_american.py

Kept as a permanent, runnable record of where the report's numbers come
from (matching the project convention: no hand-transcribed figures).
"""
import numpy as np

from barrier_pricing.params import HestonParams, JumpParams, NO_JUMPS, OptionSpec
from barrier_pricing import cos2d_american as ca
from barrier_pricing import mol

hp = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)
T = 0.5
K, H, r, q = 100.0, 130.0, 0.03, 0.05
SPOTS = (80, 90, 100, 110, 120)

MOL_PAPER_CONT = {80: 1.4012, 90: 3.9364, 100: 8.3003, 110: 14.4033, 120: 21.8219}
MOL_PY_CONT = {80: 1.4029, 90: 3.9383, 100: 8.3051, 110: 14.4206, 120: 21.8732}
MOL_PAPER_DISC = {80: 1.4008, 90: 3.9339, 100: 8.3010, 110: 14.4446, 120: 22.0389}
MOL_PY_DISC = {80: 1.4120, 90: 3.9512, 100: 8.3256, 110: 14.4826, 120: 22.0926}


def table3_continuous():
    print("=== Table 3: continuous monitoring, American ===")
    for S0 in SPOTS:
        spec = OptionSpec(S0=float(S0), K=K, H=H, r=r, q=q, T=T,
                           american=True, knock_in=False, monitor_dates=None)
        res = ca.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=200, N=512, M=200, richardson=True)
        diff = 100 * (res.price - MOL_PAPER_CONT[S0]) / MOL_PAPER_CONT[S0]
        print(f"S0={S0}: COS(py)={res.price:.4f}  MOL(py)={MOL_PY_CONT[S0]:.4f}  "
              f"MOL(paper)={MOL_PAPER_CONT[S0]:.4f}  COS vs paper={diff:+.2f}%  runtime={res.runtime:.1f}s")


def table5_discrete():
    print("=== Table 5: discrete monitoring (3 dates T/3,2T/3,T), American ===")
    for S0 in SPOTS:
        spec = OptionSpec(S0=float(S0), K=K, H=H, r=r, q=q, T=T,
                           american=True, knock_in=False, monitor_dates=[2 * T / 3, T / 3])
        res = ca.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=48, N=256, M=200, richardson=True)
        diff = 100 * (res.price - MOL_PAPER_DISC[S0]) / MOL_PAPER_DISC[S0]
        print(f"S0={S0}: COS(py)={res.price:.4f}  MOL(py)={MOL_PY_DISC[S0]:.4f}  "
              f"MOL(paper)={MOL_PAPER_DISC[S0]:.4f}  COS vs paper={diff:+.2f}%  runtime={res.runtime:.1f}s")


def jump_table():
    print("=== Jump table: American barrier with jumps (lam=0.5, gam=-0.05, del=0.1) ===")
    print("(single package-wide jump convention J ~ N(gam, del^2), compensated;")
    print(" both engines receive the same JumpParams)")
    jp = JumpParams(lam=0.5, gam=-0.05, del_=0.1)
    jp_mol = jp
    print("-- Discrete monitoring (3 dates) --")
    for S0 in (80, 100, 120):
        spec = OptionSpec(S0=float(S0), K=K, H=H, r=r, q=q, T=T,
                           american=True, knock_in=False, monitor_dates=[2 * T / 3, T / 3])
        res_j = ca.price_barrier_call(spec, hp, jp, n_exercise=48, N=256, M=200, richardson=True)
        mol_j = mol.price_barrier_call(spec, hp, jp_mol, n_S=150, M=150, n_time_steps=250)
        diff = 100 * (res_j.price - mol_j.price) / mol_j.price
        print(f"S0={S0}: COS+jump={res_j.price:.4f}  MOL+jump={mol_j.price:.4f}  diff={diff:+.2f}%")
    print("-- Continuous monitoring --")
    for S0 in (80, 100, 120):
        spec = OptionSpec(S0=float(S0), K=K, H=H, r=r, q=q, T=T,
                           american=True, knock_in=False, monitor_dates=None)
        res_j = ca.price_barrier_call(spec, hp, jp, n_exercise=150, N=512, M=200, richardson=True)
        mol_j = mol.price_barrier_call(spec, hp, jp_mol, n_S=150, M=150, n_time_steps=250)
        diff = 100 * (res_j.price - mol_j.price) / mol_j.price
        print(f"S0={S0}: COS+jump={res_j.price:.4f}  MOL+jump={mol_j.price:.4f}  diff={diff:+.2f}%")


def no_barrier_sanity():
    print("=== No-barrier American vanilla sanity (H far away) ===")
    for q_ in (0.0, 0.05):
        spec = OptionSpec(S0=100.0, K=100.0, H=1000.0, r=0.03, q=q_, T=0.5,
                           american=True, knock_in=False, monitor_dates=None)
        spec_eur = OptionSpec(S0=100.0, K=100.0, H=1000.0, r=0.03, q=q_, T=0.5,
                               american=False, knock_in=False, monitor_dates=None)
        res = ca.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=25, N=128, M=100, richardson=True)
        mol_am = mol.price_barrier_call(spec, hp, NO_JUMPS, n_S=120, M=120, n_time_steps=200)
        mol_eu = mol.price_barrier_call(spec_eur, hp, NO_JUMPS, n_S=120, M=120, n_time_steps=200)
        print(f"q={q_}: COS-American={res.price:.4f}  MOL-American={mol_am.price:.4f}  "
              f"MOL-European={mol_eu.price:.4f}")


if __name__ == "__main__":
    no_barrier_sanity()
    print()
    table3_continuous()
    print()
    table5_discrete()
    print()
    jump_table()
