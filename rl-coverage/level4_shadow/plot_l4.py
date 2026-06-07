"""Plot Level 4 benchmark — random vs PPO on 5615 bins."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow_cpu_l4 import N_BINS

data = np.load("curves_l4.npz")
ppo = data["ppo"]
rand_short = data["random"]

try:
    rand = np.load("/tmp/random_l4_long.npy")
except FileNotFoundError:
    rand = rand_short

x_r = np.arange(1, len(rand) + 1)
x_p = np.arange(1, len(ppo) + 1)

print(f"\n=== Samples to reach N% of ceiling ({N_BINS}) ===")
print(f"{'target':>8} | {'random':>14} | {'PPO':>10} | {'ratio':>8}")
print("-" * 48)
for frac in [0.25, 0.50, 0.75, 0.90, 0.95]:
    t = int(frac * N_BINS)
    r_i = np.argmax(rand >= t)
    p_i = np.argmax(ppo >= t)
    r_hit = rand[r_i] >= t; p_hit = ppo[p_i] >= t
    r_label = f"{r_i+1:,}" if r_hit else f">{len(rand):,}"
    p_label = f"{p_i+1:,}" if p_hit else f">{len(ppo):,}"
    ratio = f"{(r_i+1)/(p_i+1):.1f}x" if (r_hit and p_hit and p_i > 0) else "—"
    print(f"  {100*frac:>4.0f}%  | {r_label:>14} | {p_label:>10} | {ratio:>8}")

print("\n=== At ~630 insn/sec real-Verilator wall time ===")
print(f"{'target':>8} | {'random':>16} | {'PPO':>14}")
print("-" * 44)
for frac in [0.50, 0.75, 0.90, 0.95]:
    t = int(frac * N_BINS)
    r_i = np.argmax(rand >= t); p_i = np.argmax(ppo >= t)
    r_hit = rand[r_i] >= t; p_hit = ppo[p_i] >= t
    def fmt(n_steps, hit):
        if not hit: return ">53 min"
        sec = n_steps / 630
        if sec < 60: return f"{sec:.0f}s"
        return f"{sec/60:.1f} min"
    print(f"  {100*frac:>4.0f}%  | {fmt(r_i+1, r_hit):>16} | {fmt(p_i+1, p_hit):>14}")

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(x_r, 100 * rand / N_BINS, label="Random (structured)", color="tab:blue", lw=1.5)
ax.plot(x_p, 100 * ppo / N_BINS, label="PPO (structured)", color="tab:orange", lw=1.5)
ax.axhline(100, color="red", ls="--", lw=1, label="Ceiling")
ax.set_xlabel("Samples")
ax.set_ylabel("Coverage (% of 5615 bins)")
ax.set_title("Level 4 rich functional coverage — 5615 bins (30-op action space)")
ax.set_xscale("log")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("curves_l4.png", dpi=140)
print("\nWrote curves_l4.png")
