"""Side-by-side: L10 Random Baseline (left) | L8->L9->L10 PPO Pipeline (right)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "."

# --- Data ---
rnd   = np.load(f"{base}/level10_ops/random_l10_full_curve.npz")
rnd_ep     = rnd["ep"]
rnd_cum    = rnd["cum_pct"]

l8_cum = np.load(f"{base}/level8_ops/l8_dynamic_ppo_v2_curve.npz")["cum_pct"]
l9_cum = np.load(f"{base}/level9_ops/l9_v2_checkpoint_history.npz")["cum_pct"]
L10_FINAL  = 71.99

l8_x = np.arange(1, len(l8_cum) + 1)
l9_x = np.arange(len(l8_cum) + 1, len(l8_cum) + len(l9_cum) + 1)
l10_x = np.array([len(l8_cum) + len(l9_cum) + 1,
                  len(l8_cum) + len(l9_cum) + 36])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Ibex Toggle Coverage — Random Baseline vs PPO Pipeline", fontsize=13)

# ── LEFT: Random L10 baseline ──────────────────────────────────────────────
ax1.plot(rnd_ep, rnd_cum, color='tab:gray', lw=2, label='Random baseline')
ax1.axhline(L10_FINAL, color='tab:green', ls='--', lw=1.5,
            label=f'PPO L10 final ({L10_FINAL}%)')
ax1.axhline(72.59, color='red', ls='--', lw=1.2, label='Structural ceiling (72.59%)')
ax1.annotate(f'{rnd_cum[-1]:.2f}%', xy=(rnd_ep[-1], rnd_cum[-1]),
             xytext=(-70, 8), textcoords='offset points',
             fontsize=10, color='tab:gray', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='tab:gray', lw=1))
ax1.set_xlabel("Episode", fontsize=11)
ax1.set_ylabel("Cumulative toggle coverage (%)", fontsize=11)
ax1.set_title(f"L10 — Random Baseline ({len(rnd_ep)} ep)", fontsize=12)
ax1.set_ylim(40, 78)
ax1.set_xlim(-100, len(l8_cum) + len(l9_cum) + 36)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── RIGHT: L8 -> L9 -> L10 PPO Pipeline ───────────────────────────────────
ax2.axvspan(1, len(l8_cum), alpha=0.06, color='tab:blue')
ax2.plot(l8_x, l8_cum, color='tab:blue', lw=2, label='L8 PPO (1200 ep)')

ax2.axvspan(len(l8_cum)+1, len(l8_cum)+len(l9_cum), alpha=0.06, color='tab:orange')
ax2.plot(l9_x, l9_cum, color='tab:orange', lw=2, label='L9 PPO (1700 ep)')

ax2.axvspan(len(l8_cum)+len(l9_cum)+1, len(l8_cum)+len(l9_cum)+36,
            alpha=0.06, color='tab:green')
ax2.plot(l10_x, [l9_cum[-1], L10_FINAL], color='tab:green', lw=2,
         marker='o', markersize=5, label='L10 PPO (+4 bins)')

ax2.axhline(72.59, color='red', ls='--', lw=1.2, label='Structural ceiling (72.59%)')

ax2.axvline(len(l8_cum), color='gray', ls=':', lw=1)
ax2.axvline(len(l8_cum)+len(l9_cum), color='gray', ls=':', lw=1)

mid_l8 = len(l8_cum) / 2
mid_l9 = len(l8_cum) + len(l9_cum) / 2
mid_l10 = len(l8_cum) + len(l9_cum) + 18
ax2.text(mid_l8,  62, 'L8',  ha='center', fontsize=11, color='tab:blue',   fontweight='bold')
ax2.text(mid_l9,  62, 'L9',  ha='center', fontsize=11, color='tab:orange', fontweight='bold')
ax2.text(mid_l10, 62, 'L10', ha='center', fontsize=11, color='tab:green',  fontweight='bold')

ax2.annotate(f'{l8_cum[-1]:.2f}%', xy=(len(l8_cum), l8_cum[-1]),
             xytext=(-60, 8), textcoords='offset points',
             fontsize=9, color='tab:blue', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1))
ax2.annotate(f'{l9_cum[-1]:.2f}%', xy=(len(l8_cum)+len(l9_cum), l9_cum[-1]),
             xytext=(-70, 8), textcoords='offset points',
             fontsize=9, color='tab:orange', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='tab:orange', lw=1))
ax2.annotate(f'{L10_FINAL:.2f}%', xy=(len(l8_cum)+len(l9_cum)+36, L10_FINAL),
             xytext=(8, -15), textcoords='offset points',
             fontsize=9, color='tab:green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='tab:green', lw=1))

ax2.set_xlabel("Cumulative episodes", fontsize=11)
ax2.set_ylabel("Cumulative toggle coverage (%)", fontsize=11)
ax2.set_title("L8 → L9 → L10 PPO Pipeline", fontsize=12)
ax2.set_ylim(40, 78)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out = "graf_l10_random_full_vs_pipeline.png"
fig.savefig(out, dpi=140)
print(f"Wrote {out}")
print(f"  Random L10 final: {rnd_cum[-1]:.2f}%  ({len(rnd_ep)} ep)")
print(f"  PPO L10 final:    {L10_FINAL:.2f}%")
print(f"  L8 final: {l8_cum[-1]:.2f}%  |  L9 final: {l9_cum[-1]:.2f}%")
