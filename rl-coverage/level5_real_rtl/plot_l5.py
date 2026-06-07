"""Plot Random vs PPO cumulative toggle coverage across episodes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess, sys
from pathlib import Path

# Re-run a 100-episode random baseline to make the comparison apples-to-apples.
# Skips if /tmp/random_l5_curve.npz already exists.
RANDOM_PATH = Path("/tmp/random_l5_curve.npz")
if not RANDOM_PATH.exists():
    print("(no /tmp/random_l5_curve.npz found — generating one with 100 random episodes)")
    sys.path.insert(0, ".")
    import time, numpy as np
    from env_l5 import IbexL5Env
    from codec_l5 import N_OPS, IMM_BUCKETS
    env = IbexL5Env(episode_steps=1024, seed=42, kind="toggle", reward_mode="total")
    rng = np.random.default_rng(42)
    cum_pcts, ep_pcts = [], []
    for ep in range(100):
        env.reset()
        for _ in range(1024):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(IMM_BUCKETS)]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        ep_pcts.append(info["ep_pct"])
        cum_pcts.append(100.0 * info["cum_covered"] / info["ep_total"])
        print(f"  rand ep {ep+1:>3}: cum {cum_pcts[-1]:.2f}%")
    np.savez(RANDOM_PATH, ep_pct=np.array(ep_pcts), cum_pct=np.array(cum_pcts))

rand = np.load(RANDOM_PATH)
ppo  = np.load("ppo_l5_curve.npz")
ppo_n = np.load("ppo_l5_novelty.npz") if Path("ppo_l5_novelty.npz").exists() else None

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(np.arange(1, len(rand["cum_pct"]) + 1), rand["cum_pct"],
        label="Random (uniform over 45 ops)", color="tab:blue", lw=1.5)
ax.plot(ppo["ep"], ppo["cum_pct"],
        label="PPO (reward = total per ep)", color="tab:orange", lw=1.5)
if ppo_n is not None:
    ax.plot(ppo_n["ep"], ppo_n["cum_pct"],
            label="PPO (reward = novelty)", color="tab:green", lw=1.5)
ax.set_xlabel("Episodes (1024 instructions each)")
ax.set_ylabel("Cumulative toggle coverage on real Verilator Ibex (%)")
ax.set_title("Level 5 — real RTL toggle coverage on minimal Ibex (45-op action space)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("curves_l5.png", dpi=140)
print("Wrote curves_l5.png")

# Summary numbers
print(f"\n{'Curve':<28} {'final cum %':>12}")
print("-" * 42)
print(f"{'Random':<28} {rand['cum_pct'][-1]:>11.2f}%")
print(f"{'PPO (total reward)':<28} {ppo['cum_pct'][-1]:>11.2f}%")
if ppo_n is not None:
    print(f"{'PPO (novelty reward)':<28} {ppo_n['cum_pct'][-1]:>11.2f}%")
