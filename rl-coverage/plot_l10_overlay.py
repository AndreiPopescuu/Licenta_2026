"""Overlay: Random baseline vs PPO Pipeline on the same axes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "."

# --- Data ---
rnd    = np.load(f"{base}/level10_ops/random_l10_full_curve.npz")
rnd_ep  = rnd["ep"]
rnd_cum = rnd["cum_pct"]

l8_cum = np.load(f"{base}/level8_ops/l8_dynamic_ppo_v2_curve.npz")["cum_pct"]
l9_cum = np.load(f"{base}/level9_ops/l9_v2_checkpoint_history.npz")["cum_pct"]
L10_FINAL = 71.99

l8_x  = np.arange(1, len(l8_cum) + 1)
l9_x  = np.arange(len(l8_cum) + 1, len(l8_cum) + len(l9_cum) + 1)
l10_x = np.array([len(l8_cum) + len(l9_cum) + 1,
                  len(l8_cum) + len(l9_cum) + 36])

fig, ax = plt.subplots(figsize=(11, 5.5))

# PPO pipeline regions
ax.axvspan(1, len(l8_cum), alpha=0.06, color='tab:blue')
ax.axvspan(len(l8_cum)+1, len(l8_cum)+len(l9_cum), alpha=0.06, color='tab:orange')
ax.axvspan(len(l8_cum)+len(l9_cum)+1, len(l8_cum)+len(l9_cum)+36, alpha=0.06, color='tab:green')

# PPO curves
ax.plot(l8_x,  l8_cum, color='tab:blue',   lw=2, label='L8 PPO (1200 ep)')
ax.plot(l9_x,  l9_cum, color='tab:orange', lw=2, label='L9 PPO (1700 ep)')
ax.plot(l10_x, [l9_cum[-1], L10_FINAL], color='tab:green', lw=2,
        marker='o', markersize=5, label='L10 PPO (+4 bins)')

# Random baseline
ax.plot(rnd_ep, rnd_cum, color='tab:gray', lw=2, ls='--', label='Random baseline (2936 ep)')

# Separators
ax.axvline(len(l8_cum), color='gray', ls=':', lw=1)
ax.axvline(len(l8_cum)+len(l9_cum), color='gray', ls=':', lw=1)

# Structural ceiling
ax.axhline(72.59, color='red', ls='--', lw=1.2, label='Structural ceiling (72.59%)')

# Level labels
ax.text(len(l8_cum)/2,               42, 'L8',  ha='center', fontsize=11, color='tab:blue',   fontweight='bold')
ax.text(len(l8_cum) + len(l9_cum)/2, 42, 'L9',  ha='center', fontsize=11, color='tab:orange', fontweight='bold')
ax.text(len(l8_cum)+len(l9_cum)+18,  42, 'L10', ha='center', fontsize=11, color='tab:green',  fontweight='bold')

ax.set_xlim(-100, len(l8_cum) + len(l9_cum) + 36)
ax.set_ylim(40, 78)
ax.set_xlabel("Cumulative episodes", fontsize=12)
ax.set_ylabel("Cumulative toggle coverage (%)", fontsize=12)
ax.set_title("Random Baseline vs PPO Pipeline — Ibex Toggle Coverage", fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = "graf_l10_overlay.png"
fig.savefig(out, dpi=140)
print(f"Wrote {out}")
