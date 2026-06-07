"""Train PPO on the L7 rich-obs env.

Direct comparison with the L7 random baseline (l7_random_baseline.npz):
same stimulus space (64 ops, mem prepop, trap handler), same episode length,
only difference is PPO policy vs uniform random.

Usage:
    python train_l7_ppo.py                  # 300 episodes, saves l7_ppo_curve.npz
    python train_l7_ppo.py --episodes 100   # quick run
"""

import argparse, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env_l7_rich import IbexL7RichEnv


class Log(BaseCallback):
    def __init__(self): super().__init__(); self.history = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True
        ep = len(self.history) + 1
        self.history.append({
            "ep": ep,
            "ep_pct": info["ep_pct"],
            "cum_pct": info["cum_pct"],
            "new_hits": info.get("new_hits", 0),
            "branch_pct": info.get("branch_pct", 0.0),
        })
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | "
              f"cum {info['cum_pct']:>5.2f}% | new {info['new_hits']:>4} | "
              f"branch {info['branch_pct']:>5.2f}%", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--episode-steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="l7_ppo_curve.npz")
    args = ap.parse_args()

    env = IbexL7RichEnv(episode_steps=args.episode_steps, seed=args.seed,
                        reward_mode="novelty")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.episode_steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0, seed=args.seed, device="cpu",
    )
    print(f"Training PPO on L7 env for {args.episodes} episodes...")
    print(f"Random baseline (30 eps): 66.27%  |  L5 PPO rich (300 eps): 56.20%\n")
    cb = Log()
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.episode_steps, callback=cb)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    if cb.history:
        eps = np.array([h["ep"] for h in cb.history])
        cum = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"] for h in cb.history])
        branch = np.array([h["branch_pct"] for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct, branch_pct=branch)
        print(f"Saved {args.out}")
        print(f"Final cum toggle: {cum[-1]:.2f}%")
        print(f"Random baseline (30 eps):  66.27%")
        print(f"Random baseline (300 eps): ~{66.27:.2f}% (estimated saturation)")


if __name__ == "__main__":
    main()
