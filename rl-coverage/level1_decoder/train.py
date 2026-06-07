"""
Train a PPO agent on the Ibex decoder coverage environment and compare to random.

Logs a coverage-vs-samples curve for both agents and prints the final numbers.
Designed to finish in ~1-2 minutes on CPU so you can run it live in a meeting.
"""

import argparse
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from decoder_env import IbexDecoderEnv
from shadow_decoder import N_BINS, N_OP_TYPES


def random_baseline_structured(n_samples: int, seed: int = 0) -> np.ndarray:
    """Pick (op, rd, rs1, rs2) uniformly over the *valid* structured action space.
    Strong baseline -- matches what the agent gets "for free" from the action encoding."""
    env = IbexDecoderEnv(episode_steps=n_samples, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    curve = np.empty(n_samples, dtype=np.int32)
    for t in range(n_samples):
        a = [rng.integers(N_OP_TYPES), rng.integers(32), rng.integers(32), rng.integers(32)]
        _, _, term, trunc, info = env.step(a)
        curve[t] = info["covered"]
        if term:
            curve[t:] = info["covered"]
            break
    return curve


def random_baseline_raw32(n_samples: int, seed: int = 0) -> np.ndarray:
    """Pick a 32-bit integer uniformly and decode it. Matches the LLM4DV paper's
    baseline -- most values are illegal instructions, so coverage climbs slowly."""
    # We only need to know which (op, rd, rs1, rs2) a raw word maps to. Decode the
    # same way our shadow decoder would (opcode + funct3 + funct7 bit 30).
    env = IbexDecoderEnv(episode_steps=n_samples, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    curve = np.empty(n_samples, dtype=np.int32)
    ALU_F3_TO_OP = {
        (0, 0): "add", (0, 1): "sub", (4, 0): "xor", (6, 0): "or", (7, 0): "and",
        (1, 0): "sll", (5, 0): "srl", (5, 1): "sra", (2, 0): "slt", (3, 0): "sltu",
    }
    ALUI_F3_TO_OP = {0: "add", 4: "xor", 6: "or", 7: "and", 1: "sll", 5: "srl",  # SUBI is illegal
                     2: "slt", 3: "sltu"}
    LOAD_F3 = {0: "byte", 1: "half-word", 2: "word"}
    STORE_F3 = {0: "byte", 1: "half-word", 2: "word"}
    from shadow_decoder import OP_TYPES
    OP_KEY_TO_IDX = {(kind, name): i for i, (kind, name) in enumerate(OP_TYPES)}

    for t in range(n_samples):
        w = int(rng.integers(0, 1 << 32))
        opcode = w & 0x7F
        funct3 = (w >> 12) & 0x7
        funct7_b30 = (w >> 30) & 0x1
        rd = (w >> 7) & 0x1F
        rs1 = (w >> 15) & 0x1F
        rs2 = (w >> 20) & 0x1F
        op_idx = None
        if opcode == 0b0110011:                                            # R-type ALU
            op = ALU_F3_TO_OP.get((funct3, funct7_b30))
            if op:
                op_idx = OP_KEY_TO_IDX[("alu", op)]
        elif opcode == 0b0010011:                                          # I-type ALU
            op = ALUI_F3_TO_OP.get(funct3)
            if op:
                op_idx = OP_KEY_TO_IDX[("alu_imm", op)]
        elif opcode == 0b0000011 and funct3 in LOAD_F3:                    # loads
            op_idx = OP_KEY_TO_IDX[("load", LOAD_F3[funct3])]
        elif opcode == 0b0100011 and funct3 in STORE_F3:                   # stores
            op_idx = OP_KEY_TO_IDX[("store", STORE_F3[funct3])]
        if op_idx is None:
            # illegal -- just step with a no-op action that hits nothing.
            # We still consume a sample, so the curve includes illegal waste.
            curve[t] = env.covered.sum()
            continue
        _, _, term, trunc, info = env.step([op_idx, rd, rs1, rs2])
        curve[t] = info["covered"]
        if term:
            curve[t:] = info["covered"]; break
    return curve


def ppo_evaluate(model, n_samples: int, seed: int = 0) -> np.ndarray:
    """Roll out the trained policy for n_samples steps (single long episode)."""
    env = IbexDecoderEnv(episode_steps=n_samples, seed=seed)
    obs, _ = env.reset(seed=seed)
    curve = np.empty(n_samples, dtype=np.int32)
    for t in range(n_samples):
        action, _ = model.predict(obs, deterministic=False)  # keep stochasticity for exploration
        obs, _, term, trunc, info = env.step(action)
        curve[t] = info["covered"]
        if term:
            curve[t:] = info["covered"]
            break
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=150_000, help="Total env steps for PPO training")
    ap.add_argument("--eval-samples", type=int, default=5_000, help="Samples for the final coverage curve")
    ap.add_argument("--episode-steps", type=int, default=256, help="Steps per training episode")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Random (raw 32-bit, paper baseline): {args.eval_samples} samples ===")
    t0 = time.time()
    raw_curve = random_baseline_raw32(args.eval_samples, seed=args.seed)
    print(f"  final: {raw_curve[-1]}/{N_BINS} = {100*raw_curve[-1]/N_BINS:.2f}%  "
          f"({time.time()-t0:.1f}s)")

    print(f"\n=== Random (structured action): {args.eval_samples} samples ===")
    t0 = time.time()
    rand_curve = random_baseline_structured(args.eval_samples, seed=args.seed)
    print(f"  final: {rand_curve[-1]}/{N_BINS} = {100*rand_curve[-1]/N_BINS:.2f}%  "
          f"({time.time()-t0:.1f}s)")

    print(f"\n=== Training PPO: {args.ppo_steps} env steps ===")
    def make_env():
        return IbexDecoderEnv(episode_steps=args.episode_steps, seed=args.seed)
    vec_env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel envs
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.02,        # higher entropy -> more exploration, helps for coverage
        verbose=0,
        seed=args.seed,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s")

    print(f"\n=== PPO evaluation: {args.eval_samples} samples ===")
    t0 = time.time()
    ppo_curve = ppo_evaluate(model, args.eval_samples, seed=args.seed)
    print(f"  final: {ppo_curve[-1]}/{N_BINS} = {100*ppo_curve[-1]/N_BINS:.2f}%  "
          f"({time.time()-t0:.1f}s)")

    np.savez("curves.npz", raw32=raw_curve, random=rand_curve, ppo=ppo_curve)
    print(f"\nSaved curves.npz (shape {rand_curve.shape}).")

    # quick ASCII summary of a few checkpoints
    checkpoints = [100, 500, 1000, 2500, 5000]
    print(f"\n{'samples':>8} | {'raw32':>10} | {'rand-struct':>12} | {'PPO':>10}")
    print("-" * 52)
    for n in checkpoints:
        if n <= len(rand_curve):
            raw = 100 * raw_curve[n-1] / N_BINS
            r = 100 * rand_curve[n-1] / N_BINS
            p = 100 * ppo_curve[n-1] / N_BINS
            print(f"{n:>8} | {raw:>8.2f}%  | {r:>10.2f}%  | {p:>8.2f}%")


if __name__ == "__main__":
    main() 
