"""Train PPO on the rich-obs Level 5 env.

Fair test: PPO sees per-module coverage, learns which modules still need
stimulus, gets novelty reward, trains for 300 episodes. If it still can't
beat random, the action space is truly saturated.
"""

import argparse, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env_l5_rich import IbexL5RichEnv


class Log(BaseCallback):
    def __init__(self): super().__init__(); self.history = []
    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info: return True
        ep = len(self.history) + 1
        self.history.append({
            "ep": ep, "ep_pct": info["ep_pct"], "cum_pct": info["cum_pct"],
            "new_hits": info.get("new_hits", 0),
            "branch_pct": info.get("branch_pct", 0.0),
        })
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | "
              f"cum {info['cum_pct']:>5.2f}% | new {info['new_hits']:>4} | "
              f"branch {info['branch_pct']:>5.2f}%")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--episode-steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ppo_l5_rich.npz")
    args = ap.parse_args()

    env = IbexL5RichEnv(episode_steps=args.episode_steps, seed=args.seed,
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
    print(f"Training rich-obs PPO for {args.episodes} episodes...\n")
    cb = Log()
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.episode_steps, callback=cb)
    print(f"\nDone in {time.time()-t0:.0f}s")

    eps = np.array([h["ep"] for h in cb.history])
    cum = np.array([h["cum_pct"] for h in cb.history])
    ep_pct = np.array([h["ep_pct"] for h in cb.history])
    branch = np.array([h["branch_pct"] for h in cb.history])
    np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct, branch_pct=branch)
    print(f"Final cum toggle: {cum[-1]:.2f}%    (random at same N eps: ~54.49%)")


if __name__ == "__main__":
    main()
