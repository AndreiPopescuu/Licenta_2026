"""Train PPO on the Level 5 real-Verilator env with toggle coverage as reward.

Each episode:
  1. PPO emits 1024 actions
  2. We run the program through real Ibex and read coverage.dat
  3. Reward = number of toggle points covered this episode
  4. PPO learns to emit programs that maximize toggle

Single env (no vectorization) because each env step's terminal action invokes
Vtop as a subprocess and they'd race on coverage.dat. Each episode takes ~2 s,
so 200 episodes ~ 7 minutes wall.

Logs cumulative toggle coverage across episodes, which is the comparable metric
against the random baseline (also cumulative) and against published numbers.
"""

import argparse, time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env_l5 import IbexL5Env


class CovCallback(BaseCallback):
    """Logs cumulative toggle and per-episode coverage after each episode."""

    def __init__(self):
        super().__init__()
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if not info or "ep_pct" not in info:
            return True
        ep = len(self.history) + 1
        cum_total = info.get("ep_total", 1)
        cum_pct = 100.0 * info.get("cum_covered", 0) / cum_total
        self.history.append({
            "ep": ep,
            "ep_pct": info["ep_pct"],
            "cum_pct": cum_pct,
            "branch_pct": info.get("branch_pct", 0.0),
            "line_pct":   info.get("line_pct", 0.0),
        })
        if ep % 1 == 0:
            print(f"  ep {ep:>4} | ep_toggle {info['ep_pct']:>6.2f}% | "
                  f"cum_toggle {cum_pct:>6.2f}% | branch {info.get('branch_pct',0):>5.2f}%")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--episode-steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ppo_l5_curve.npz")
    ap.add_argument("--reward", choices=["total", "novelty"], default="novelty")
    args = ap.parse_args()

    env = IbexL5Env(episode_steps=args.episode_steps, seed=args.seed,
                    kind="toggle", reward_mode=args.reward)
    print(f"Reward mode: {args.reward}\n")

    # Per-step PPO config — n_steps = episode_steps so one rollout = one episode.
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.episode_steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,            # episodic, very long horizon for toggle credit
        ent_coef=0.05,          # high entropy — toggle wants diverse stimulus
        verbose=0, seed=args.seed, device="cpu",
    )

    cb = CovCallback()
    total_steps = args.episodes * args.episode_steps
    print(f"Training PPO for {args.episodes} episodes "
          f"({total_steps:,} env steps)...\n")
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
    dt = time.time() - t0
    print(f"\nTrained in {dt:.0f}s  ({args.episodes/(dt/60):.1f} eps/min)")

    eps = np.array([h["ep"] for h in cb.history])
    ep_pct = np.array([h["ep_pct"] for h in cb.history])
    cum_pct = np.array([h["cum_pct"] for h in cb.history])
    branch_pct = np.array([h["branch_pct"] for h in cb.history])
    np.savez(args.out, ep=eps, ep_pct=ep_pct, cum_pct=cum_pct, branch_pct=branch_pct)
    print(f"Saved {args.out}")
    print(f"\nFinal cumulative toggle: {cum_pct[-1]:.2f}%   "
          f"(random plateau across same N eps: ~52.65%)")


if __name__ == "__main__":
    main()
