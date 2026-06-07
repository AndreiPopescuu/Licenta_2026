"""Generate an L6 program (random or PPO-rolled-out), compute the shadow's
predicted coverage, encode the sequence to 32-bit RISC-V, and dump JSON for
real-RTL validation.

The real-RTL test loads the machine code into Ibex memory, executes it, runs
our shadow-backed monitor on each RVFI retirement, and compares the resulting
coverage set against `shadow_hit_bins`.
"""

import argparse, json
import numpy as np

from shadow_cpu_l4 import (
    Op, N_BINS, BIN_NAMES, L6History,
    bins_for_step, advance_history,
    IMM_BUCKETS as N_IMM_BUCKETS,
)
from codec_l4 import encode

N_OPS = len(Op)


def rollout_random(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    hist = L6History()
    covered = set()
    sequence = []
    for _ in range(n):
        op = int(rng.integers(N_OPS))
        rd = int(rng.integers(32))
        rs1 = int(rng.integers(32))
        rs2 = int(rng.integers(32))
        ib = int(rng.integers(N_IMM_BUCKETS))
        for b in bins_for_step(op, rd, rs1, rs2, ib, hist):
            covered.add(b)
        sequence.append((op, rd, rs1, rs2, ib))
        advance_history(op, rd, rs1, rs2, hist)
    return sequence, covered


def rollout_ppo(model_path: str, n: int, seed: int = 42):
    from stable_baselines3 import PPO
    from cpu_env_l4 import IbexL4Env
    model = PPO.load(model_path, device="cpu")
    env = IbexL4Env(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    hist = L6History()
    covered = set()
    sequence = []
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        op = int(action[0]); rd = int(action[1])
        rs1 = int(action[2]); rs2 = int(action[3]); ib = int(action[4])
        for b in bins_for_step(op, rd, rs1, rs2, ib, hist):
            covered.add(b)
        sequence.append((op, rd, rs1, rs2, ib))
        advance_history(op, rd, rs1, rs2, hist)
        obs, _, _, _, _ = env.step(action)
    return sequence, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agent", choices=["random", "ppo"], default="random")
    ap.add_argument("--ppo-model", default=None, help="path to saved PPO model (for --agent ppo)")
    ap.add_argument("--out", default="/tmp/rl_l4_program.json")
    args = ap.parse_args()

    if args.agent == "random":
        seq, covered = rollout_random(args.n, seed=args.seed)
    else:
        if not args.ppo_model:
            raise SystemExit("--ppo-model required when --agent ppo")
        seq, covered = rollout_ppo(args.ppo_model, args.n, seed=args.seed)

    machine = [encode(*t) for t in seq]
    shadow_hit = sorted(BIN_NAMES[b] for b in covered)

    print(f"Generated {len(seq)} L6 instructions with agent={args.agent}")
    print(f"Shadow coverage: {len(shadow_hit)}/{N_BINS} = {100*len(shadow_hit)/N_BINS:.2f}%")

    with open(args.out, "w") as f:
        json.dump({
            "n": len(seq),
            "agent": args.agent,
            "seed": args.seed,
            "sequence": seq,                              # [(op,rd,rs1,rs2,ib), ...]
            "machine_code": [int(m) for m in machine],    # list of 32-bit words
            "shadow_hit_bins": shadow_hit,
        }, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
