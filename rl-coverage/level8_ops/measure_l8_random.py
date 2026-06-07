"""Random-agent baseline on the L8 70-op env.

Directly comparable to L7's measure_l7_random: same 1024-step episodes,
same seed, same 30 episodes. Only difference: codec extended with
LUI / JALR / FENCE / CSRRWI / CSRRSI / CSRRCI (64 → 70 ops) and the
test_run_for_l8 cocotb driver (longer register-init prologue).
"""

import time, sys
from pathlib import Path
import numpy as np

from env_l8 import IbexL8Env
from codec_l8 import N_OPS, IMM_BUCKETS

EP_STEPS = 1024
N_EPISODES = 30
OUT_NPZ = Path(__file__).parent / "l8_random_baseline.npz"


def main():
    env = IbexL8Env(episode_steps=EP_STEPS, seed=42, kind="toggle")
    rng = np.random.default_rng(42)
    print(f"L8 action space: {N_OPS} ops × 32³ × {IMM_BUCKETS} imm", flush=True)
    print(f"{'ep':>3} | {'ep%':>6} | {'cum%':>6} | {'branch%':>7} | {'line%':>6} | {'wall':>5}")
    print("-" * 55)

    ep_pcts    = np.zeros(N_EPISODES)
    cum_pcts   = np.zeros(N_EPISODES)
    branch_pcts = np.zeros(N_EPISODES)
    line_pcts  = np.zeros(N_EPISODES)

    for ep in range(N_EPISODES):
        env.reset()
        t0 = time.time()
        for _ in range(EP_STEPS):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(IMM_BUCKETS)]
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        dt = time.time() - t0
        ep_pcts[ep]    = info.get("ep_pct", 0.0)
        cum_pcts[ep]   = 100.0 * info.get("cum_covered", 0) / max(info.get("ep_total", 1), 1)
        branch_pcts[ep] = info.get("branch_pct", 0.0)
        line_pcts[ep]  = info.get("line_pct", 0.0)
        print(f"{ep+1:>3} | {ep_pcts[ep]:>5.2f}% | {cum_pcts[ep]:>5.2f}% | "
              f"{branch_pcts[ep]:>6.2f}% | {line_pcts[ep]:>5.2f}% | {dt:>4.1f}s", flush=True)

    np.savez(OUT_NPZ, ep=np.arange(1, N_EPISODES + 1),
             ep_pct=ep_pcts, cum_pct=cum_pcts,
             branch_pct=branch_pcts, line_pct=line_pcts)

    print(f"\nSaved {OUT_NPZ.name}")
    print(f"Final cum toggle: {cum_pcts[-1]:.2f}%")
    print(f"vs L7 random (30 eps):  66.27%")
    print(f"vs L6 random (150 eps): 57.48%")


if __name__ == "__main__":
    main()
