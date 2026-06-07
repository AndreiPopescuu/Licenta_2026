"""Plot coverage-vs-samples curves from curves.npz."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow_decoder import N_BINS, max_reachable_bins

data = np.load("curves.npz")
ceiling = max_reachable_bins()

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(1, len(data["random"]) + 1)
ax.plot(x, 100 * data["raw32"] / N_BINS, label="Random (raw 32-bit)", color="gray", lw=1.5)
ax.plot(x, 100 * data["random"] / N_BINS, label="Random (structured)", color="tab:blue", lw=1.5)
ax.plot(x, 100 * data["ppo"] / N_BINS, label="PPO (structured)", color="tab:orange", lw=1.5)
ax.axhline(100 * ceiling / N_BINS, color="red", ls="--", lw=1, label=f"ISA ceiling ({100*ceiling/N_BINS:.1f}%)")
ax.set_xlabel("Samples (simulation steps)")
ax.set_ylabel("Coverage (% of 2107 bins)")
ax.set_title("Ibex decoder coverage -- random vs PPO (shadow decoder)")
ax.set_xscale("log")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("curves.png", dpi=140)
print("Wrote curves.png")
