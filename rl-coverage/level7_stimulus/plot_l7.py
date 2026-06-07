"""L5/L6/L7 progression chart."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
L5 = THIS.parent / "level5_real_rtl"
L6 = THIS.parent / "level6_rvc"


def _load(p: Path):
    try: return dict(np.load(p))
    except Exception: return {}


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    l5_ppo_rich = _load(L5 / "ppo_l5_rich.npz")
    l6_rand_150 = _load(L6 / "rvc_random_baseline_150.npz")
    l6_ppo      = _load(L6 / "ppo_rvc_rich.npz")
    l7_rand     = _load(THIS / "l7_random_baseline.npz")
    l7_ppo      = _load(THIS / "l7_ppo_curve.npz")

    if l5_ppo_rich:
        ax.plot(l5_ppo_rich["ep"], l5_ppo_rich["cum_pct"],
                label=f"L5 PPO rich (45 ops, 300 eps) → {l5_ppo_rich['cum_pct'][-1]:.2f}%",
                color="#888888", linestyle="--", linewidth=1.2)
    if l6_rand_150:
        ax.plot(l6_rand_150["ep"], l6_rand_150["cum_pct"],
                label=f"L6 RVC random (61 ops, 150 eps) → {l6_rand_150['cum_pct'][-1]:.2f}%",
                color="#d62728", linewidth=1.6)
    if l6_ppo:
        ax.plot(l6_ppo["ep"], l6_ppo["cum_pct"],
                label=f"L6 RVC PPO (61 ops, 150 eps) → {l6_ppo['cum_pct'][-1]:.2f}%",
                color="#1f77b4", linewidth=1.6)
    if l7_rand:
        ax.plot(l7_rand["ep"], l7_rand["cum_pct"],
                label=f"L7 stimulus random (64 ops + mem prepop + trap, 30 eps) → {l7_rand['cum_pct'][-1]:.2f}%",
                color="#2ca02c", linewidth=2.8)
    if l7_ppo:
        n_eps = len(l7_ppo["ep"])
        ax.plot(l7_ppo["ep"], l7_ppo["cum_pct"],
                label=f"L7 stimulus PPO (64 ops + mem prepop + trap, {n_eps} eps) → {l7_ppo['cum_pct'][-1]:.2f}%",
                color="#ff7f0e", linewidth=2.0)

    ax.axhline(75.86, color="#2ca02c", linestyle=":", alpha=0.5, linewidth=1.2,
               label="L7 reachable ceiling ≈ 75.9% (tied-off excluded)")
    ax.axhline(56.20, color="#888888", linestyle="-", alpha=0.3, linewidth=1)
    ax.text(5, 56.7, "L5 ceiling 56.2%", fontsize=8, color="#666666")

    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative toggle coverage (%)")
    ax.set_title("Ibex toggle coverage: stimulus engineering beats algorithm tuning")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(25, 80)

    out = THIS / "l7_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
