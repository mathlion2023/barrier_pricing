"""Staged validation of barrier_pricing/cos2d_joint.py (the new joint (X,v)
Bermudan-COS recursion) before trusting it enough to touch JAM_Paper.tex.

Stage 1: no barrier (H far away), European, vs the closed-form Heston price
         (charfunc.cos_vanilla_call) -- isolates whether the tree+COS joint
         recursion reproduces plain Heston pricing at all.
Stage 2: with barrier, European, vs MOL-European and the old frozen-branch
         cos2d.py engine.
Stage 3: with barrier, American, vs MOL-American and the old frozen-branch
         cos2d_american.py engine -- the actual test of whether the deep-OTM
         bias shrinks.
"""
import numpy as np

from barrier_pricing.params import HestonParams, JumpParams, NO_JUMPS, OptionSpec
from barrier_pricing import cos2d_joint as cj
from barrier_pricing import cos2d_american as ca
from barrier_pricing import cos2d
from barrier_pricing import mol
from barrier_pricing.charfunc import cos_vanilla_call

hp = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)
T, K, H, r, q = 0.5, 100.0, 130.0, 0.03, 0.05
SPOTS = (80.0, 90.0, 100.0, 110.0, 120.0)


def rule(title):
    print(f"\n=== {title} ===")


def stage1_no_barrier_european():
    rule("Stage 1: no barrier, European, joint recursion vs closed-form Heston")
    for S0 in (80.0, 100.0, 120.0):
        spec = OptionSpec(S0=S0, K=K, H=1000.0, r=r, q=q, T=T, american=False,
                           knock_in=False, monitor_dates=None)
        res = cj.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=200, N=512,
                                     M=300, richardson=True)
        closed = cos_vanilla_call(S0, K, r, q, T, hp, NO_JUMPS)
        diff = 100 * (res.price - closed) / closed
        print(f"S0={S0}: joint={res.price:.4f}  closed-form={closed:.4f}  "
              f"diff={diff:+.3f}%  runtime={res.runtime:.2f}s")


def stage2_barrier_european():
    rule("Stage 2: WITH barrier, European (2-date discrete: T/2, T -- CKM's "
         "own European convention), joint vs MOL vs old frozen-branch "
         "cos2d.py vs published CKM values")
    # cos2d.py's own convention (unlike cos2d_american.py's) requires EVERY
    # monitoring date explicit in the list, maturity included (tau=0).
    monitor_dates_full = [T / 2, 0.0]
    for S0 in SPOTS:
        spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=T, american=False,
                           knock_in=False, monitor_dates=monitor_dates_full)
        res_joint = cj.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=48,
                                           N=256, M=200, richardson=True)
        res_old = cos2d.price_barrier_call(spec, hp, NO_JUMPS, N=1024, M=200)
        res_mol = mol.price_barrier_call(spec, hp, NO_JUMPS, n_S=150, M=150,
                                          n_time_steps=200)
        print(f"S0={S0}: joint={res_joint.price:.4f}  old-frozen-branch={res_old.price:.4f}  "
              f"MOL={res_mol.price:.4f}  "
              f"joint-vs-MOL={100*(res_joint.price-res_mol.price)/res_mol.price:+.3f}%  "
              f"old-vs-MOL={100*(res_old.price-res_mol.price)/res_mol.price:+.3f}%  "
              f"joint runtime={res_joint.runtime:.2f}s")


def stage3_barrier_american():
    rule("Stage 3 (the real test): WITH barrier, American, continuous "
         "monitoring, joint vs old frozen-branch cos2d_american.py vs MOL vs "
         "published CKM values -- does the deep-OTM bias shrink?")
    MOL_PAPER_CONT = {80: 1.4012, 90: 3.9364, 100: 8.3003, 110: 14.4033, 120: 21.8219}
    for S0 in SPOTS:
        spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=T, american=True,
                           knock_in=False, monitor_dates=None)
        res_joint = cj.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=100,
                                           N=512, M=300, richardson=True)
        res_old = ca.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=200,
                                         N=512, M=200, richardson=True)
        ckm = MOL_PAPER_CONT[S0]
        print(f"S0={S0}: joint={res_joint.price:.4f}  old-frozen-branch={res_old.price:.4f}  "
              f"CKM(published)={ckm:.4f}  "
              f"joint-vs-CKM={100*(res_joint.price-ckm)/ckm:+.3f}%  "
              f"old-vs-CKM={100*(res_old.price-ckm)/ckm:+.3f}%  "
              f"joint runtime={res_joint.runtime:.2f}s")


if __name__ == "__main__":
    stage1_no_barrier_european()
    stage2_barrier_european()
    stage3_barrier_american()
