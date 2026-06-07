"""Plot the Level 6 (RVC-extended) results alongside the Level 5 baselines.

Two panels:
  (a) cumulative toggle % over episodes — RVC random, RVC PPO, L5 PPO rich
  (b) per-module coverage "unlocked" by RVC — bar chart of compressed_decoder
      and a few adjacent modules, before vs after.

Saves to rvc_comparison.png.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
L5_DIR = THIS.parent / "level5_real_rtl"


def _load(p: Path) -> dict:
    try:
        return dict(np.load(p))
    except Exception as e:
        print(f"(skipping {p.name}: {e})")
        return {}


def main():
    rvc_rand_150 = _load(THIS / "rvc_random_baseline_150.npz")
    rvc_rand_30  = _load(THIS / "rvc_random_baseline.npz")
    rvc_ppo  = _load(THIS / "ppo_rvc_rich.npz")
    l5_ppo_rich = _load(L5_DIR / "ppo_l5_rich.npz")
    l5_ppo_curve = _load(L5_DIR / "ppo_l5_curve.npz")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    if l5_ppo_rich:
        ax.plot(l5_ppo_rich["ep"], l5_ppo_rich["cum_pct"],
                label=f"L5 PPO rich (45 ops, 300 eps) → {l5_ppo_rich['cum_pct'][-1]:.2f}%",
                color="#888888", linestyle="--", linewidth=1.2)
    if l5_ppo_curve:
        ax.plot(l5_ppo_curve["ep"], l5_ppo_curve["cum_pct"],
                label=f"L5 PPO (45 ops, 100 eps) → {l5_ppo_curve['cum_pct'][-1]:.2f}%",
                color="#cccccc", linestyle=":", linewidth=1)
    if rvc_rand_150:
        ax.plot(rvc_rand_150["ep"], rvc_rand_150["cum_pct"],
                label=f"L6 RVC random (61 ops, 150 eps) → {rvc_rand_150['cum_pct'][-1]:.2f}%",
                color="#d62728", linewidth=2.2)
    if rvc_ppo:
        ax.plot(rvc_ppo["ep"], rvc_ppo["cum_pct"],
                label=f"L6 RVC PPO rich (61 ops, 150 eps) → {rvc_ppo['cum_pct'][-1]:.2f}%",
                color="#1f77b4", linewidth=2.5)

    ax.axhline(69.6, color="#2ca02c", linestyle=":", alpha=0.6, linewidth=1.2,
               label="reachable ceiling ≈ 69.6% (tied-off excluded)")
    ax.axhline(56.20, color="#888888", linestyle="-", alpha=0.3, linewidth=1)
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative toggle coverage (%)")
    ax.set_title("Ibex toggle coverage: Level 5 (45 base ops) vs Level 6 (+ 16 RVC ops)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(20, 72)
    ax.text(5, 56.5, "L5 PPO ceiling (56.2%)", fontsize=8, color="#666666")

    out = THIS / "rvc_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
