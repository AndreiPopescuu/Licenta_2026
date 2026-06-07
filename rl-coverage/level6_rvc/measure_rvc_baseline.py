"""Random-agent baseline on the RVC-extended action space.

Mirrors level5_real_rtl/measure_l5_baselines.py exactly so numbers are
directly comparable. Writes per-episode curves to rvc_random_baseline.npz.

Also reports a breakdown of compressed_decoder / decoder / controller /
cs_registers coverage, so we can see which RTL submodules the RVC extension
unlocked.
"""

import time
from pathlib import Path

import numpy as np

from env_rvc import IbexRVCEnv
from codec_rvc import N_OPS, IMM_BUCKETS

EP_STEPS = 1024
N_EPISODES = 30
OUT_NPZ = Path(__file__).parent / "rvc_random_baseline.npz"

KEY_MODULES = [
    "ibex_compressed_decoder",
    "ibex_decoder",
    "ibex_controller",
    "ibex_cs_registers",
    "ibex_alu",
    "ibex_multdiv_fast",
    "ibex_id_stage",
    "ibex_if_stage",
    "ibex_load_store_unit",
    "ibex_register_file_ff",
]


def main():
    env = IbexRVCEnv(episode_steps=EP_STEPS, seed=42, kind="toggle")
    rng = np.random.default_rng(42)

    print(f"RVC-extended action space: {N_OPS} ops × 32³ × {IMM_BUCKETS} imm buckets")
    print(f"Episode = {EP_STEPS} actions ({EP_STEPS} × 4 bytes program).\n")
    print(f"{'ep':>3} | {'ep_tog%':>7} | {'cum_tog%':>8} | {'branch%':>7} | {'line%':>6} | {'wall':>5}")
    print("-" * 60)

    ep_pcts = np.zeros(N_EPISODES)
    cum_pcts = np.zeros(N_EPISODES)
    branch_pcts = np.zeros(N_EPISODES)
    line_pcts = np.zeros(N_EPISODES)

    for ep in range(N_EPISODES):
        env.reset()
        t0 = time.time()
        for _ in range(EP_STEPS):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(IMM_BUCKETS)]
            _, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        dt = time.time() - t0
        ep_pct = info.get("ep_pct", 0.0)
        cum_pct = 100.0 * info.get("cum_covered", 0) / max(info.get("ep_total", 1), 1)
        ep_pcts[ep] = ep_pct
        cum_pcts[ep] = cum_pct
        branch_pcts[ep] = info.get("branch_pct", 0.0)
        line_pcts[ep] = info.get("line_pct", 0.0)
        print(f"{ep+1:>3} | {ep_pct:>6.2f}% | {cum_pct:>7.2f}% | "
              f"{branch_pcts[ep]:>6.2f}% | {line_pcts[ep]:>5.2f}% | {dt:>4.1f}s")

    np.savez(OUT_NPZ, ep=np.arange(1, N_EPISODES + 1), ep_pct=ep_pcts,
             cum_pct=cum_pcts, branch_pct=branch_pcts, line_pct=line_pcts)
    print(f"\nSaved curve to {OUT_NPZ.name}")

    # ---- Final summary: per-module coverage on the last run ----
    print("\nFinal single-run module breakdown (last episode's coverage.dat):")
    import cov_parser
    summary = cov_parser.parse("../../cpu/coverage.dat")
    rows = []
    for page, (c, t) in summary.by_page.items():
        kind, _, module = page.partition("/")
        if kind != "v_toggle": continue
        if any(km in module for km in KEY_MODULES):
            rows.append((module, c, t))
    for module, c, t in sorted(rows):
        pct = 100 * c / t if t else 0
        print(f"  {module:<50s} {c:>4}/{t:<4} = {pct:5.1f}%")


if __name__ == "__main__":
    main()
