"""Plot pipeline L8 -> L9 -> L10 cumulative coverage curve."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

base = "."

# Load data
l8 = np.load(f"{base}/level8_ops/l8_dynamic_ppo_v2_curve.npz")["cum_pct"]
l9 = np.load(f"{base}/level9_ops/l9_v2_checkpoint_history.npz")["cum_pct"]

# L10 adds ~4 bins — approximate as a small step after L9
L10_FINAL = 71.99

# Concatenate L8 + L9 on a single x axis
l8_eps = np.arange(1, len(l8) + 1)
l9_eps = np.arange(len(l8) + 1, len(l8) + len(l9) + 1)
l10_eps = np.array([len(l8) + len(l9) + 1, len(l8) + len(l9) + 36])

all_x   = np.concatenate([l8_eps, l9_eps, l10_eps])
all_pct = np.concatenate([l8, l9, [l9[-1], L10_FINAL]])

fig, ax = plt.subplots(figsize=(11, 5.5))

# L8 region
ax.axvspan(1, len(l8), alpha=0.06, color='tab:blue', label='_nolegend_')
ax.plot(l8_eps, l8, color='tab:blue', lw=2, label='L8 PPO (70 ops, 1200 ep)')

# L9 region
ax.axvspan(len(l8)+1, len(l8)+len(l9), alpha=0.06, color='tab:orange', label='_nolegend_')
ax.plot(l9_eps, l9, color='tab:orange', lw=2, label='L9 PPO (83 ops, 1700 ep)')

# L10 step
ax.axvspan(len(l8)+len(l9)+1, len(l8)+len(l9)+36, alpha=0.06, color='tab:green', label='_nolegend_')
ax.plot(l10_eps, [l9[-1], L10_FINAL], color='tab:green', lw=2,
        marker='o', markersize=5, label='L10 PPO (87 ops, +4 bins)')

# Structural ceiling
ax.axhline(72.59, color='red', ls='--', lw=1.2, label='Structural ceiling (72.59%)')

# Vertical separators
ax.axvline(len(l8), color='gray', ls=':', lw=1)
ax.axvline(len(l8)+len(l9), color='gray', ls=':', lw=1)

# Level labels
ax.text(len(l8)/2, 63, 'L8', ha='center', fontsize=11,
        color='tab:blue', fontweight='bold')
ax.text(len(l8) + len(l9)/2, 63, 'L9', ha='center', fontsize=11,
        color='tab:orange', fontweight='bold')
ax.text(len(l8)+len(l9)+18, 63, 'L10', ha='center', fontsize=11,
        color='tab:green', fontweight='bold')

# Annotate final values
ax.annotate(f'{l8[-1]:.2f}%', xy=(len(l8), l8[-1]),
            xytext=(-60, 8), textcoords='offset points',
            fontsize=10, color='tab:blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1))
ax.annotate(f'{l9[-1]:.2f}%', xy=(len(l8)+len(l9), l9[-1]),
            xytext=(-70, 8), textcoords='offset points',
            fontsize=10, color='tab:orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='tab:orange', lw=1))
ax.annotate(f'{L10_FINAL:.2f}%', xy=(len(l8)+len(l9)+36, L10_FINAL),
            xytext=(8, -15), textcoords='offset points',
            fontsize=10, color='tab:green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='tab:green', lw=1))

ax.set_xlabel("Cumulative episodes", fontsize=12)
ax.set_ylabel("Cumulative toggle coverage (%)", fontsize=12)
ax.set_title("L8 → L9 → L10 Pipeline — Ibex Toggle Coverage", fontsize=13)
ax.set_ylim(60, 78)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("graf_pipeline_l8_l9_l10.png", dpi=140)
print("Wrote graf_pipeline_l8_l9_l10.png")
print(f"  L8 final: {l8[-1]:.2f}%  ({len(l8)} ep)")
print(f"  L9 final: {l9[-1]:.2f}%  ({len(l9)} ep)")
print(f"  L10 final: {L10_FINAL:.2f}%")
print(f"  Total episodes: {len(l8)+len(l9)+36}")
