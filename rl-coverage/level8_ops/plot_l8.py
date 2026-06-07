"""L7/L8 comparison chart — 2 subplots: random vs PPO."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
L7   = THIS.parent / "level7_stimulus"


def _load(p: Path):
    try: return dict(np.load(p))
    except Exception: return {}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    l7_rand = _load(L7 / "l7_random_baseline.npz")
    l7_ppo  = _load(L7 / "l7_ppo_curve.npz")
    l8_rand = _load(THIS / "l8_random_baseline.npz")
    l8_dyn  = _load(THIS / "l8_dynamic_ppo_scratch.npz")
    l8_v2   = _load(THIS / "l8_dynamic_ppo_v2_curve.npz")
    l8_v3   = _load(THIS / "l8_v3_ppo_curve.npz")

    YLIM = (58, 78)

    # ── Subplot 1: Constrained Random ────────────────────────────────────
    ax1.set_title("Constrained Random: L7 vs L8", fontsize=12, fontweight="bold")

    if l7_rand:
        ax1.plot(l7_rand["ep"], l7_rand["cum_pct"],
                 label=f"L7 random\n64 ops (trap + CSR + mem)\n→ {l7_rand['cum_pct'][-1]:.2f}%",
                 color="#2ca02c", linewidth=2.5, marker="o", markersize=5)

    if l8_rand:
        ax1.plot(l8_rand["ep"], l8_rand["cum_pct"],
                 label=f"L8 random\n70 ops (+LUI/JALR/FENCE/CSRi)\n→ {l8_rand['cum_pct'][-1]:.2f}%",
                 color="#17becf", linewidth=2.5, marker="s", markersize=5)

    ax1.axhline(75.9, color="#2ca02c", linestyle="--", alpha=0.5, linewidth=1.2)
    ax1.text(1, 76.2, "ceiling ≥ 75.9%", fontsize=8.5, color="#2ca02c", alpha=0.85)

    ax1.set_xlabel("episode", fontsize=11)
    ax1.set_ylabel("cumulative toggle coverage (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim(YLIM)

    # ── Subplot 2: PPO (x = total steps pentru comparație corectă) ─────────
    ax2.set_title("PPO: L7 vs L8 v1 vs L8 v2 vs L8 v3\n(x = total steps simulați)", fontsize=12, fontweight="bold")

    if l7_ppo:
        steps = l7_ppo["ep"] * 1024
        ax2.plot(steps / 1000, l7_ppo["cum_pct"],
                 label=f"L7 PPO vanilla\n(300 ep × 1024 steps)\n→ {l7_ppo['cum_pct'][-1]:.2f}%",
                 color="#ff7f0e", linewidth=2.5, marker="o", markersize=3)

    if l8_dyn:
        steps = l8_dyn["ep"] * 1024
        ax2.plot(steps / 1000, l8_dyn["cum_pct"],
                 label=f"L8 PPO dynamic weights\n(300 ep × 1024 steps)\n→ {l8_dyn['cum_pct'][-1]:.2f}%",
                 color="#e377c2", linewidth=2.5, marker="s", markersize=3)

    if l8_v3:
        steps = l8_v3["ep"] * 256
        ax2.plot(steps / 1000, l8_v3["cum_pct"],
                 label=f"L8 PPO v3\n(obs 32 dims + action hist)\n→ {l8_v3['cum_pct'][-1]:.2f}%",
                 color="#d62728", linewidth=2.5, marker="D", markersize=3)

    if l8_v2:
        steps = l8_v2["ep"] * 256
        ax2.plot(steps / 1000, l8_v2["cum_pct"],
                 label=f"L8 PPO dynamic weights v2\n(1200 ep × 256 steps)\n→ {l8_v2['cum_pct'][-1]:.2f}%",
                 color="#9467bd", linewidth=2.5, marker="^", markersize=3)

    ax2.axhline(75.9, color="#888888", linestyle="--", alpha=0.5, linewidth=1.2)
    ax2.text(10, 76.2, "ceiling ≥ 75.9%", fontsize=8.5, color="#888888", alpha=0.85)

    ax2.set_xlabel("total steps (×1000)", fontsize=11)
    ax2.set_ylabel("cumulative toggle coverage (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_ylim(YLIM)

    fig.suptitle("Ibex Toggle Coverage — L7 vs L8", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = THIS / "l8_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
