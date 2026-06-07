"""random_baseline_l9.py — Constrained random baseline pe L9 (83 ops).

Rulează N episoade de program random (uniform sampling din action space),
acumulează toggle coverage și compară cu RL-ul de 71.97%.

Usage:
    python random_baseline_l9.py --episodes 500
    python random_baseline_l9.py --episodes 500 --steps 256 --seed 0
    python random_baseline_l9.py --episodes 500 --out random_curve.npz
"""

import argparse, pickle, time
from pathlib import Path
import sys
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l9_v2 import IbexL9V2Env, N_OBS
from codec_l9 import N_OPS, IMM_BUCKETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3945)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="random_baseline_curve.npz")
    ap.add_argument("--save-hits", default="random_baseline_hits.pkl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Action space dims: [N_OPS=83, rd=32, rs1=32, rs2=32, imm_bucket=5]
    action_dims = [N_OPS, 32, 32, 32, IMM_BUCKETS]

    print("=" * 64)
    print(f"Constrained Random Baseline — L9 ({N_OPS} ops)")
    print(f"Episodes: {args.episodes}  Steps/ep: {args.steps}  Seed: {args.seed}")
    print(f"Action space: MultiDiscrete({action_dims})")
    print("=" * 64)
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δ vs RL':>10}")
    print("-" * 45)

    RL_REFERENCE = 71.97  # target de batut

    env = IbexL9V2Env(episode_steps=args.steps, seed=args.seed)
    env.reset()

    history = []
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()

        # Sampling uniform random din fiecare dimensiune a action space
        for _ in range(args.steps):
            action = [int(rng.integers(0, d)) for d in action_dims]
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                break

        if "cum_pct" not in info:
            continue

        cum   = info["cum_pct"]
        ep_p  = info["ep_pct"]
        new   = info.get("new_hits_vs_cum", 0)
        delta = cum - RL_REFERENCE

        history.append({"ep": ep, "cum_pct": cum, "ep_pct": ep_p, "new_hits": new})
        print(f"  {ep:>4} | {ep_p:>5.2f}% | {cum:>5.2f}% | {new:>5} | {delta:>+7.2f}pp",
              flush=True)

    elapsed = time.time() - t0
    print(f"\nDone în {elapsed/60:.1f} min")

    if history:
        best  = max(h["cum_pct"] for h in history)
        final = history[-1]["cum_pct"]
        print(f"\nRezultate:")
        print(f"  Random best:   {best:.2f}%")
        print(f"  Random final:  {final:.2f}%")
        print(f"  RL L9v2 ref:   {RL_REFERENCE:.2f}%")
        print(f"  Delta (random - RL): {best - RL_REFERENCE:+.2f}pp")

        eps    = np.array([h["ep"]      for h in history])
        cum    = np.array([h["cum_pct"] for h in history])
        ep_pct = np.array([h["ep_pct"]  for h in history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")

        with open(args.save_hits, "wb") as f:
            pickle.dump(env._cum_hits, f)
        print(f"  Hits → {args.save_hits}")


if __name__ == "__main__":
    main()
