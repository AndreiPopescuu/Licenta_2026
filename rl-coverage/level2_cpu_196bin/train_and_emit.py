"""Train PPO on the full env (with JAL), roll out a program, save to /tmp/rl_program.json
for real-RTL validation.

Forward JAL  → encoded as jal(rd, +4): jumps to next instruction, RAW chain preserved.
Backward JAL → encoded with two BEQ trampolines (BEQ is not in our 14 ops, fires no
               coverage bins). The BEQ breaks the RAW hazard chain, so backward JAL
               cannot be a RAW writer — shadow models this by clearing prev_w after it.
               All 196 bins are reachable: the 13 jal-writer RAW hazard bins are covered
               by forward JAL, and jal_br_backwards is covered by backward JAL.
"""
import argparse, json, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cpu_env import IbexCpuEnv
from shadow_cpu import Op, bins_for_step, WRITERS, BIN_NAMES


def rollout_policy(model, n: int, seed: int):
    env = IbexCpuEnv(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    seq, covered = [], set()
    prev_w, prev_r = None, None
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        op_idx = int(action[0]); rd = int(action[1]); rs1 = int(action[2]); rs2 = int(action[3])
        op_enum = Op(op_idx)
        if op_enum == Op.JAL:
            # imm_bucket=0 → backward, anything else → forward (avoid zero-offset self-jump)
            imm_sign = -1 if int(action[4]) == 0 else +1
        else:
            imm_sign = 0
        for b in bins_for_step(int(op_enum), rd, rs1, rs2, imm_sign, prev_w, prev_r):
            covered.add(b)
        seq.append((int(op_enum), rd, rs1, rs2, imm_sign))
        if op_enum in WRITERS:
            # Backward JAL uses a BEQ trampoline in RTL which breaks the RAW hazard
            # chain — model that here so shadow and RTL agree.
            if op_enum == Op.JAL and imm_sign < 0:
                prev_w, prev_r = None, None
            else:
                prev_w, prev_r = op_enum, rd
        else:
            prev_w, prev_r = None, None
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=seed+1); prev_w = prev_r = None
    return seq, sorted(BIN_NAMES[b] for b in covered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=300_000)
    ap.add_argument("--rollout-n", type=int, default=10_000)
    ap.add_argument("--episode-steps", type=int, default=512)
    ap.add_argument("--out", default="/tmp/rl_program.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def make_env(): return IbexCpuEnv(episode_steps=args.episode_steps, seed=args.seed)
    vec = DummyVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", vec,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.99, ent_coef=0.03, verbose=0, seed=args.seed, device="cpu")
    print(f"Training PPO (with JAL) for {args.ppo_steps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Rolling out {args.rollout_n} instructions...")
    seq, shadow_hit = rollout_policy(model, args.rollout_n, seed=args.seed)
    print(f"  shadow predicts {len(shadow_hit)}/196 bins")

    with open(args.out, "w") as f:
        json.dump({"n_instructions": len(seq), "seed": args.seed,
                   "sequence": seq, "shadow_hit_bins": shadow_hit}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
