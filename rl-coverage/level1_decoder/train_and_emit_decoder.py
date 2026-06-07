"""Train PPO on the decoder shadow, roll out a program, save to
/tmp/rl_decoder_program.json for real-RTL validation.

alu_imm_sub is excluded: it would fire the illegal_instruction bin, but emitting
a truly illegal instruction halts the CPU (exception → trap vector → WFI), which
would abort the rest of the sequence.  The ceiling without it is 2041/2107.
"""
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from decoder_env import IbexDecoderEnv
from shadow_decoder import bins_for_action, BIN_NAMES, N_BINS, N_OP_TYPES, OP_TYPES

# op_idx for ("alu_imm", "sub") — illegal in RISC-V, skipped in RTL program
_ALU_IMM_SUB = next(i for i, (k, n) in enumerate(OP_TYPES) if k == "alu_imm" and n == "sub")


def rollout_policy(model, n: int, seed: int):
    env = IbexDecoderEnv(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    seq, covered = [], set()
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        op_idx = int(action[0]); rd = int(action[1]); rs1 = int(action[2]); rs2 = int(action[3])
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=seed + 1)
        if op_idx == _ALU_IMM_SUB:
            continue  # skip: can't emit without halting CPU
        for b in bins_for_action(op_idx, rd, rs1, rs2):
            covered.add(b)
        seq.append((op_idx, rd, rs1, rs2))
    return seq, sorted(BIN_NAMES[b] for b in covered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=300_000)
    ap.add_argument("--rollout-n", type=int, default=10_000)
    ap.add_argument("--episode-steps", type=int, default=512)
    ap.add_argument("--out", default="/tmp/rl_decoder_program.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def make_env(): return IbexDecoderEnv(episode_steps=args.episode_steps, seed=args.seed)
    vec = DummyVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", vec,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.99, ent_coef=0.02, verbose=0, seed=args.seed, device="cpu")

    print(f"Training PPO (decoder) for {args.ppo_steps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Rolling out {args.rollout_n} instructions...")
    seq, shadow_hit = rollout_policy(model, args.rollout_n, seed=args.seed)
    print(f"  shadow predicts {len(shadow_hit)}/{N_BINS} bins (ceiling 2041 without alu_imm_sub)")

    with open(args.out, "w") as f:
        json.dump({"n_instructions": len(seq), "seed": args.seed,
                   "sequence": seq, "shadow_hit_bins": shadow_hit}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
