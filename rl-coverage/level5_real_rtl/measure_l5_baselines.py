"""Measure random baseline on the EXTENDED Level 5 action space.

Reports per-episode and cumulative toggle / branch / line coverage across N
episodes. Each episode is independent (CPU resets), so the cumulative number
is what we'd measure if we kept all coverage from a series of test runs.
"""

import time
import numpy as np

from env_l5 import IbexL5Env
from codec_l5 import N_OPS, IMM_BUCKETS

EP_STEPS = 1024
N_EPISODES = 30


def main():
    env = IbexL5Env(episode_steps=EP_STEPS, seed=42, kind="toggle")
    rng = np.random.default_rng(42)
    print(f"Action space: {N_OPS} ops × 32 × 32 × 32 × {IMM_BUCKETS} imm buckets")
    print(f"Episode = {EP_STEPS} instructions.\n")
    print(f"{'ep':>3} | {'ep_toggle%':>10} | {'cum_toggle%':>11} | {'branch%':>7} | {'line%':>5} | {'wall':>5}")
    print("-" * 60)
    for ep in range(N_EPISODES):
        env.reset()
        t0 = time.time()
        for _ in range(EP_STEPS):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(IMM_BUCKETS)]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        dt = time.time() - t0
        ep_pct = info.get("ep_pct", 0)
        cum_pct = 100.0 * info.get("cum_covered", 0) / info.get("ep_total", 1)
        b_pct = info.get("branch_pct", 0)
        l_pct = info.get("line_pct", 0)
        print(f"{ep+1:>3} | {ep_pct:>9.2f}% | {cum_pct:>10.2f}% | {b_pct:>6.2f}% | {l_pct:>4.2f}% | {dt:>4.1f}s")


if __name__ == "__main__":
    main()
