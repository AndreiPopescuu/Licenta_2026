"""Plot coverage-vs-samples curves from curves_cpu.npz (196-bin CPU env)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow_cpu import N_BINS

data = np.load("curves_cpu.npz")
x = np.arange(1, len(data["random"]) + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, 100 * data["random"] / N_BINS, label="Random (structured)", color="tab:blue", lw=1.5)
ax.plot(x, 100 * data["ppo"] / N_BINS, label="PPO (structured)", color="tab:orange", lw=1.5)
ax.axhline(100, color="red", ls="--", lw=1, label="Ceiling (100%)")
ax.set_xlabel("Samples (simulation steps)")
ax.set_ylabel("Coverage (% of 196 bins)")
ax.set_title("Ibex CPU 196-bin coverage -- random vs PPO (sequential, RAW hazards)")
ax.set_xscale("log")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("curves_cpu.png", dpi=140)
print("Wrote curves_cpu.png")
