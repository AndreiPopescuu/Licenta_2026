"""Train + evaluate on the 5615-bin Level 6 benchmark."""

import argparse, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cpu_env_l4 import IbexL4Env, N_OPS
from shadow_cpu_l4 import N_BINS, IMM_BUCKETS


def random_rollout(n: int, seed: int = 0):
    env = IbexL4Env(episode_steps=n, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    curve = np.empty(n, dtype=np.int32)
    for t in range(n):
        a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
             rng.integers(32), rng.integers(IMM_BUCKETS)]
        _, _, term, _, info = env.step(a)
        curve[t] = info["covered"]
        if term: curve[t:] = info["covered"]; break
    return curve


def ppo_rollout(model, n: int, seed: int = 0):
    env = IbexL4Env(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    curve = np.empty(n, dtype=np.int32)
    for t in range(n):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, term, _, info = env.step(action)
        curve[t] = info["covered"]
        if term: curve[t:] = info["covered"]; break
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=3_000_000)
    ap.add_argument("--eval-samples", type=int, default=200_000)
    ap.add_argument("--episode-steps", type=int, default=4096)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"OBS_DIM per env: {IbexL4Env().observation_space.shape[0]}")

    print(f"\n=== Random: {args.eval_samples:,} samples ===")
    t0 = time.time()
    rand = random_rollout(args.eval_samples, seed=args.seed)
    print(f"  final: {rand[-1]}/{N_BINS} = {100*rand[-1]/N_BINS:.2f}%  ({time.time()-t0:.1f}s)")

    print(f"\n=== PPO training: {args.ppo_steps:,} steps, {args.n_envs} envs ===")
    def make_env():
        return IbexL4Env(episode_steps=args.episode_steps, seed=args.seed)
    try:
        vec = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    except Exception:
        vec = DummyVecEnv([make_env for _ in range(args.n_envs)])
    model = PPO(
        "MlpPolicy", vec,
        learning_rate=3e-4, n_steps=512, batch_size=2048, n_epochs=4,
        gamma=0.995, ent_coef=0.02,
        policy_kwargs=dict(net_arch=[512, 512]),
        verbose=0, seed=args.seed, device=device,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    dt = time.time() - t0
    print(f"  trained in {dt:.1f}s  ({args.ppo_steps/dt:,.0f} steps/sec)")

    print(f"\n=== PPO rollout: {args.eval_samples:,} samples ===")
    t0 = time.time()
    ppo = ppo_rollout(model, args.eval_samples, seed=args.seed)
    print(f"  final: {ppo[-1]}/{N_BINS} = {100*ppo[-1]/N_BINS:.2f}%  ({time.time()-t0:.1f}s)")

    np.savez("curves_l4.npz", random=rand, ppo=ppo)
    model.save("ppo_l4.zip")
    print("Saved model to ppo_l4.zip")

    print(f"\n{'samples':>10} | {'random':>10} | {'PPO':>10}")
    print("-" * 38)
    for n in [1000, 5000, 10000, 50000, 100000, 200000]:
        if n <= len(rand):
            print(f"{n:>10,} | {100*rand[n-1]/N_BINS:>8.2f}%  | {100*ppo[n-1]/N_BINS:>8.2f}%")


if __name__ == "__main__":
    main()
