"""Plot the chain-benchmark curves and compute the sample-efficiency gap."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow_cpu_chains import N_BINS

data = np.load("curves_chains.npz")
rand_short = data["random"]
ppo = data["ppo"]

rand_long = None
try:
    rand_long = np.load("/tmp/random_long.npy")
except FileNotFoundError:
    pass

rand = rand_long if rand_long is not None else rand_short
x_r = np.arange(1, len(rand) + 1)
x_p = np.arange(1, len(ppo) + 1)

print(f"\n=== Samples needed to reach N% of the ceiling ({N_BINS}) ===")
print(f"{'target':>8} | {'random':>12} | {'PPO':>10} | {'ratio':>8}")
print("-" * 46)
for frac in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    t = int(frac * N_BINS)
    r_i = np.argmax(rand >= t)
    p_i = np.argmax(ppo >= t)
    if rand[r_i] < t:
        r_label = f">{len(rand):,}"
    else:
        r_label = f"{r_i+1:,}"
    if ppo[p_i] < t:
        p_label = f">{len(ppo):,}"
    else:
        p_label = f"{p_i+1:,}"
    if rand[r_i] >= t and ppo[p_i] >= t and p_i > 0:
        ratio = f"{(r_i + 1) / (p_i + 1):.1f}x"
    else:
        ratio = "—"
    print(f"  {100*frac:>4.0f}%  | {r_label:>12} | {p_label:>10} | {ratio:>8}")

# Verilator wall-time view
print("\n=== In real-Verilator wall time (at ~630 insn/sec on this box) ===")
print(f"{'target':>8} | {'random':>16} | {'PPO':>14}")
print("-" * 44)
for frac in [0.75, 0.90, 0.95, 0.99]:
    t = int(frac * N_BINS)
    r_i = np.argmax(rand >= t)
    p_i = np.argmax(ppo >= t)
    if rand[r_i] < t: r_wall = "—"
    else: r_wall = f"{(r_i+1)/630:.0f}s" if r_i < 630*60 else f"{(r_i+1)/630/60:.1f}min"
    if ppo[p_i] < t: p_wall = "—"
    else: p_wall = f"{(p_i+1)/630:.0f}s"
    print(f"  {100*frac:>4.0f}%  | {r_wall:>16} | {p_wall:>14}")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(x_r, 100 * rand / N_BINS, label="Random (structured)", color="tab:blue", lw=1.5)
ax.plot(x_p, 100 * ppo / N_BINS, label="PPO (structured)", color="tab:orange", lw=1.5)
ax.axhline(100, color="red", ls="--", lw=1, label="Ceiling")
ax.set_xlabel("Samples")
ax.set_ylabel("Coverage (% of 1739 bins)")
ax.set_title("Level 3 — 1739-bin benchmark (49 easy + 390 RAW 1/2/3 + 1300 K=3 chains)")
ax.set_xscale("log")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("curves_chains.png", dpi=140)
print("\nWrote curves_chains.png")
