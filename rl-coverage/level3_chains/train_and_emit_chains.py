"""Train PPO on the 1739-bin chain shadow, roll out a program, save to
/tmp/rl_chains_program.json for real-RTL validation.

No JAL — all 13 ops (ADD..SW) encode directly to R-type / S-type words.
"""
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cpu_env_chains import IbexChainsEnv
from shadow_cpu_chains import Op, bins_for_step, advance_history, ChainHistory, BIN_NAMES, N_BINS


def rollout_policy(model, n: int, seed: int):
    env = IbexChainsEnv(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    seq, covered = [], set()
    hist = ChainHistory()
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        op_idx = int(action[0]); rd = int(action[1]); rs1 = int(action[2]); rs2 = int(action[3])
        for b in bins_for_step(op_idx, rd, rs1, rs2, hist):
            covered.add(b)
        advance_history(op_idx, rd, rs1, rs2, hist)
        seq.append((op_idx, rd, rs1, rs2))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=seed + 1)
            hist.reset()
    return seq, sorted(BIN_NAMES[b] for b in covered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=2_000_000)
    ap.add_argument("--rollout-n", type=int, default=200_000)
    ap.add_argument("--episode-steps", type=int, default=2048)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--out", default="/tmp/rl_chains_program.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | envs: {args.n_envs}")

    def make_env(): return IbexChainsEnv(episode_steps=args.episode_steps, seed=args.seed)
    try:
        vec = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    except Exception:
        vec = DummyVecEnv([make_env for _ in range(args.n_envs)])
    model = PPO("MlpPolicy", vec,
                learning_rate=3e-4, n_steps=512, batch_size=2048, n_epochs=4,
                gamma=0.995, ent_coef=0.02,
                policy_kwargs=dict(net_arch=[512, 512]),
                verbose=0, seed=args.seed, device=device)

    print(f"Training PPO (chains, 1739 bins) for {args.ppo_steps:,} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Rolling out {args.rollout_n:,} instructions...")
    seq, shadow_hit = rollout_policy(model, args.rollout_n, seed=args.seed)
    print(f"  shadow predicts {len(shadow_hit)}/{N_BINS} bins")

    with open(args.out, "w") as f:
        json.dump({"n_instructions": len(seq), "seed": args.seed,
                   "sequence": seq, "shadow_hit_bins": shadow_hit}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
