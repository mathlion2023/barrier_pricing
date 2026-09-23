"""Additional validation requested by the critical mathematical referee report
(JAM_Paper_Critical_Mathematical_Review.pdf, Section 6): limiting cases, a
jump martingale check, and a multi-parameter-set stress sweep (COS vs MOL,
plus Monte Carlo for the European/no-barrier cases where MC applies).

Kept as a permanent, runnable record of every new number quoted in the
paper's "Additional validation" subsection -- run from this directory:

    python referee_response_validation.py
"""
import numpy as np

from barrier_pricing.params import HestonParams, JumpParams, NO_JUMPS, OptionSpec
from barrier_pricing import cos2d_american as ca
from barrier_pricing import cos2d
from barrier_pricing.cos2d import _phi_svjd_grid
from barrier_pricing import mol
from barrier_pricing import mc
from barrier_pricing.charfunc import heston_jump_cf, cos_vanilla_call

BASE_HP = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)
T, K, H, r, q = 0.5, 100.0, 130.0, 0.03, 0.05


def rule(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------
# 1. Jump martingale check: phi(-i) = exp((r-q)*T), under BOTH jump
#    conventions in the codebase (charfunc.py's literal-mean J~N(gam,del^2)
#    and cos2d.py's mean-shifted J~N(gam-0.5*del^2,del^2)) -- both must
#    independently satisfy the martingale restriction after the referee-
#    response fix (previously cos2d's convention failed this, see
#    JAM_Paper.tex Section 3 and the module docstrings in cos2d.py /
#    cos2d_american.py for what was found and fixed).
# ---------------------------------------------------------------------
def martingale_check():
    rule("Martingale check: E[S_T]/S0 = exp((r-q)T), phi(-i;T)=exp((r-q)T)")
    v0 = np.array([BASE_HP.v0])
    for lam, gam, delv in [(0.0, 0.0, 1e-8), (0.5, -0.05, 0.10),
                            (2.0, 0.10, 0.30), (5.0, -0.20, 0.05)]:
        jp = JumpParams(lam=lam, gam=gam, del_=delv)
        target = np.exp((r - q) * T)
        got_charfunc = heston_jump_cf(np.array([-1j]), T, r, q, BASE_HP, jp)[0].real
        got_cos2d = _phi_svjd_grid(np.array([-1j]), v0, T, r, q, BASE_HP, jp)[0].real
        print(f"lam={lam:.2f} gam={gam:+.2f} del={delv:.2f}: "
              f"charfunc phi(-i)={got_charfunc:.10f}  cos2d phi(-i)={got_cos2d:.10f}  "
              f"exp((r-q)T)={target:.10f}  "
              f"abs err (charfunc)={abs(got_charfunc - target):.2e}  "
              f"abs err (cos2d)={abs(got_cos2d - target):.2e}")


# ---------------------------------------------------------------------
# 2. Limiting cases.
# ---------------------------------------------------------------------
def limiting_cases():
    rule("Limiting case: sigma (vol-of-vol) -> 0, v0=theta (degenerate to "
         "constant-variance BS), European, vs 1D COS closed-form Heston "
         "(itself -> BS as sigma->0) and MC")
    hp_lowvov = HestonParams(kappa=2.0, theta=0.1, sigma=1e-4, rho=-0.5, v0=0.1)
    for S0 in (80.0, 100.0, 120.0):
        spec = OptionSpec(S0=S0, K=K, H=1.0e6, r=r, q=q, T=T,
                           american=False, knock_in=False, monitor_dates=None)
        cos1d = cos_vanilla_call(S0, K, r, q, T, hp_lowvov, NO_JUMPS)
        mc_res = mc.price_barrier_call_mc(spec, hp_lowvov, NO_JUMPS,
                                           n_paths=400_000, n_steps=50, seed=1)
        print(f"S0={S0}: COS-1D(closed form)={cos1d:.4f}  "
              f"MC={mc_res.price:.4f} +/- {mc_res.extra['ci95']:.4f} (95% CI)")

    rule("Limiting case: lambda -> 0 (jump engine reduces to pure Heston)")
    jp0 = JumpParams(lam=1e-10, gam=-0.05, del_=0.10)
    for S0 in (80.0, 100.0, 120.0):
        spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=T, american=True,
                           knock_in=False, monitor_dates=None)
        p_nojump = ca.price_barrier_call(spec, BASE_HP, NO_JUMPS, n_exercise=50,
                                          N=256, M=150, richardson=True).price
        p_lam0 = ca.price_barrier_call(spec, BASE_HP, jp0, n_exercise=50,
                                        N=256, M=150, richardson=True).price
        print(f"S0={S0}: no-jump engine={p_nojump:.4f}  "
              f"jump engine, lambda=1e-10={p_lam0:.4f}  "
              f"diff={100*(p_lam0-p_nojump)/p_nojump:+.4f}%")

    rule("Limiting case: H far away AND q=0 (up-and-out American reduces "
         "to the closed-form vanilla European price: no dividend removes "
         "the exercise incentive, and a barrier at H=1000=10K is essentially "
         "never touched over T=0.5 at these vol levels)")
    for S0 in (80.0, 100.0, 120.0):
        spec_am = OptionSpec(S0=S0, K=K, H=1000.0, r=r, q=0.0, T=T, american=True,
                              knock_in=False, monitor_dates=None)
        cos_am_nobarrier = ca.price_barrier_call(spec_am, BASE_HP, NO_JUMPS,
                                                  n_exercise=100, N=512, M=200,
                                                  richardson=True).price
        cos_vanilla = cos_vanilla_call(S0, K, r, 0.0, T, BASE_HP, NO_JUMPS)
        print(f"S0={S0}: Bermudan-COS American engine (H=1e6, q=0)="
              f"{cos_am_nobarrier:.4f}  1D COS vanilla (closed form)="
              f"{cos_vanilla:.4f}  "
              f"diff={100*(cos_am_nobarrier-cos_vanilla)/cos_vanilla:+.4f}%")

    rule("Limiting case, corrected: q=0 does NOT make an up-and-out American "
         "call equal to its European counterpart with a live barrier -- the "
         "classical no-dividend theorem is a statement about VANILLA calls "
         "(it uses E[S_T]=S0*exp((r-q)T) under the terminal measure, which "
         "the barrier's knock-out invalidates: waiting also risks losing the "
         "option to the barrier, so early exercise can have positive value "
         "even at q=0). Numbers below (H=130, continuous monitoring) confirm "
         "this rather than assume it: MOL-American vs MOL-European diverge "
         "sharply, showing a large early-exercise premium survives at q=0 "
         "once a barrier is active. The valid form of this limiting case -- "
         "American=European once the barrier is also removed -- is the "
         "separate H-far-away-AND-q=0 check above, which DOES hold (up to "
         "the same deep-OTM bias already documented).")
    for S0 in (80.0, 100.0, 120.0):
        spec_am = OptionSpec(S0=S0, K=K, H=H, r=r, q=0.0, T=T, american=True,
                              knock_in=False, monitor_dates=None)
        spec_eu = OptionSpec(S0=S0, K=K, H=H, r=r, q=0.0, T=T, american=False,
                              knock_in=False, monitor_dates=None)
        p_am_cos = ca.price_barrier_call(spec_am, BASE_HP, NO_JUMPS, n_exercise=100,
                                          N=512, M=200, richardson=True).price
        p_am_mol = mol.price_barrier_call(spec_am, BASE_HP, NO_JUMPS, n_S=150,
                                           M=150, n_time_steps=200).price
        p_eu_mol = mol.price_barrier_call(spec_eu, BASE_HP, NO_JUMPS, n_S=150,
                                           M=150, n_time_steps=200).price
        print(f"S0={S0}: COS-American={p_am_cos:.4f}  MOL-American={p_am_mol:.4f}  "
              f"MOL-European={p_eu_mol:.4f}  "
              f"early-exercise premium (MOL American-European)/European="
              f"{100*(p_am_mol-p_eu_mol)/p_eu_mol:+.1f}%")

    rule("Limiting case: rho=0")
    hp_rho0 = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=0.0, v0=0.1)
    for S0 in (80.0, 100.0, 120.0):
        spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=T, american=True,
                           knock_in=False, monitor_dates=None)
        p_cos = ca.price_barrier_call(spec, hp_rho0, NO_JUMPS, n_exercise=100,
                                       N=512, M=200, richardson=True).price
        p_mol = mol.price_barrier_call(spec, hp_rho0, NO_JUMPS, n_S=150, M=150,
                                        n_time_steps=200).price
        print(f"S0={S0}: COS={p_cos:.4f}  MOL={p_mol:.4f}  "
              f"diff={100*(p_cos-p_mol)/p_mol:+.4f}%")

    rule("Limiting case: v0=theta (start at long-run mean)")
    hp_v0theta = HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.5, v0=0.1)
    # (identical to BASE_HP here since theta=v0=0.1 already; kept explicit for
    # traceability against the referee's checklist item.)
    for S0 in (80.0, 100.0, 120.0):
        spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=T, american=True,
                           knock_in=False, monitor_dates=None)
        p_cos = ca.price_barrier_call(spec, hp_v0theta, NO_JUMPS, n_exercise=100,
                                       N=512, M=200, richardson=True).price
        p_mol = mol.price_barrier_call(spec, hp_v0theta, NO_JUMPS, n_S=150, M=150,
                                        n_time_steps=200).price
        print(f"S0={S0}: COS={p_cos:.4f}  MOL={p_mol:.4f}  "
              f"diff={100*(p_cos-p_mol)/p_mol:+.4f}%")

    rule("Limiting case: short maturity T=0.02 (near-immediate)")
    for S0 in (80.0, 100.0, 120.0):
        spec = OptionSpec(S0=S0, K=K, H=H, r=r, q=q, T=0.02, american=True,
                           knock_in=False, monitor_dates=None)
        p_cos = ca.price_barrier_call(spec, BASE_HP, NO_JUMPS, n_exercise=50,
                                       N=512, M=200, richardson=True).price
        intrinsic = max(S0 - K, 0.0)
        print(f"S0={S0}: COS={p_cos:.4f}  intrinsic={intrinsic:.4f}")


# ---------------------------------------------------------------------
# 3. Stress sweep: multiple parameter sets, COS vs MOL.
# ---------------------------------------------------------------------
def stress_sweep():
    rule("Stress sweep: COS vs MOL across parameter regimes (continuous "
         "monitoring, American, S0=100 at-the-money unless noted)")
    cases = [
        ("base (Table 1)", BASE_HP, 100.0, H),
        ("Feller violated (2*kappa*theta < sigma^2)",
         HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5, v0=0.04), 100.0, H),
        ("Feller satisfied, high vol-of-vol",
         HestonParams(kappa=4.0, theta=0.09, sigma=0.4, rho=-0.5, v0=0.09), 100.0, H),
        ("high negative correlation rho=-0.9",
         HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=-0.9, v0=0.1), 100.0, H),
        ("high positive correlation rho=+0.9",
         HestonParams(kappa=2.0, theta=0.1, sigma=0.1, rho=0.9, v0=0.1), 100.0, H),
        ("barrier near strike H=105",
         BASE_HP, 100.0, 105.0),
        ("barrier near spot H=102, S0=100",
         BASE_HP, 100.0, 102.0),
        ("deep OTM S0=60", BASE_HP, 60.0, H),
        ("deep ITM S0=125 (near barrier)", BASE_HP, 125.0, H),
        ("long maturity T=2 (barrier widened)", BASE_HP, 100.0, 160.0),
    ]
    for label, hp, S0, Hlevel in cases:
        Tlocal = 2.0 if "long maturity" in label else T
        spec = OptionSpec(S0=S0, K=K, H=Hlevel, r=r, q=q, T=Tlocal, american=True,
                           knock_in=False, monitor_dates=None)
        res_cos = ca.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=100,
                                         N=512, M=200, richardson=True)
        res_mol = mol.price_barrier_call(spec, hp, NO_JUMPS, n_S=150, M=150,
                                          n_time_steps=200)
        feller = hp.feller_ratio()
        diff = 100 * (res_cos.price - res_mol.price) / res_mol.price
        print(f"{label:45s} Feller={feller:5.2f}  COS={res_cos.price:9.4f}  "
              f"MOL={res_mol.price:9.4f}  diff={diff:+7.3f}%  "
              f"COS runtime={res_cos.runtime:5.2f}s  MOL runtime={res_mol.runtime:6.2f}s")


# ---------------------------------------------------------------------
# 4. Convergence follow-up on the four largest stress-sweep gaps: refine
#    BOTH engines well beyond the base resolution to see which one (if
#    either) was under-resolved, rather than report the base-resolution
#    gap unexplained. This is what Section "Additional validation"'s
#    stress-sweep discussion (JAM_Paper.tex) is based on.
# ---------------------------------------------------------------------
def stress_convergence_followup():
    rule("Convergence follow-up: Feller-violated / high-vol-of-vol "
         "(does MOL or COS need finer resolution?)")
    cases = [
        ("Feller violated", HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5, v0=0.04)),
        ("high vol-of-vol, Feller satisfied", HestonParams(kappa=4.0, theta=0.09, sigma=0.4, rho=-0.5, v0=0.09)),
    ]
    for label, hp in cases:
        spec = OptionSpec(S0=100.0, K=K, H=H, r=r, q=q, T=T, american=True,
                           knock_in=False, monitor_dates=None)
        print(f"--- {label} (Feller={hp.feller_ratio():.2f}) ---")
        for (N, nex, M) in [(512, 100, 200), (1024, 200, 400), (1024, 300, 600)]:
            res = ca.price_barrier_call(spec, hp, NO_JUMPS, n_exercise=nex, N=N, M=M, richardson=True)
            print(f"  COS N={N} n_ex={nex} M={M}: price={res.price:.4f} runtime={res.runtime:.2f}s")
        for (nS, Mm, nt) in [(150, 150, 200), (250, 300, 400)]:
            res = mol.price_barrier_call(spec, hp, NO_JUMPS, n_S=nS, M=Mm, n_time_steps=nt)
            print(f"  MOL n_S={nS} M={Mm} n_t={nt}: price={res.price:.4f} runtime={res.runtime:.2f}s")

    rule("Convergence follow-up: tight barriers (does MOL or COS need finer resolution?)")
    tight_cases = [
        ("barrier near strike H=105", 100.0, 105.0),
        ("barrier near spot H=102, S0=100", 100.0, 102.0),
    ]
    for label, S0, Hlevel in tight_cases:
        spec = OptionSpec(S0=S0, K=K, H=Hlevel, r=r, q=q, T=T, american=True,
                           knock_in=False, monitor_dates=None)
        print(f"--- {label} ---")
        for (N, nex, M) in [(512, 100, 200), (1024, 300, 300), (2048, 600, 400)]:
            res = ca.price_barrier_call(spec, BASE_HP, NO_JUMPS, n_exercise=nex, N=N, M=M, richardson=True)
            print(f"  COS N={N} n_ex={nex} M={M}: price={res.price:.4f} runtime={res.runtime:.2f}s")
        for (nS, Mm, nt) in [(150, 150, 200), (300, 250, 400)]:
            res = mol.price_barrier_call(spec, BASE_HP, NO_JUMPS, n_S=nS, M=Mm, n_time_steps=nt)
            print(f"  MOL n_S={nS} M={Mm} n_t={nt}: price={res.price:.4f} runtime={res.runtime:.2f}s")


if __name__ == "__main__":
    martingale_check()
    limiting_cases()
    stress_sweep()
    stress_convergence_followup()
