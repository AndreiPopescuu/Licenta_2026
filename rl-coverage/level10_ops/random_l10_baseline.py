"""random_l10_baseline.py — Random agent cu action space L10 (87 ops + csr_bucket).

Compară cu random L9 (71.99%) pentru a vedea dacă instrucțiunile extra din L10
(ILLEGAL_INSN, LW_MISALIGN, SW_MISALIGN, LH_MISALIGN + csr_bucket) aduc mai mult.

Usage:
    python random_l10_baseline.py --episodes 500 --steps 256
"""

import argparse, pickle, time, sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l10 import IbexL10Env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="random_l10_curve.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    env = IbexL10Env(episode_steps=args.steps, seed=args.seed)

    print(f"Random agent L10 — {args.episodes} ep × {args.steps} pași")
    print(f"Action space: {env.action_space.nvec.tolist()}")
    print(f"  (L9 referinta: 71.99%)")
    print(f"{'ep':>5} | {'cum%':>6} | {'ep%':>6} | {'tog+':>5} | {'elapsed':>8}")
    print("-" * 50)

    history = []
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if "cum_pct" in info:
            cum  = info["cum_pct"]
            ep_p = info["ep_pct"]
            new  = info.get("new_hits_vs_cum", 0)
            history.append({"ep": ep, "cum_pct": cum, "ep_pct": ep_p})

            if ep % 50 == 0 or ep <= 10 or new > 0:
                elapsed = time.time() - t0
                print(f"  {ep:>4} | {cum:>6.3f}% | {ep_p:>6.3f}% | +{new:<4} | {elapsed:>6.1f}s",
                      flush=True)

    elapsed = time.time() - t0
    if history:
        best  = max(h["cum_pct"] for h in history)
        final = history[-1]["cum_pct"]
        print(f"\n{'='*50}")
        print(f"Random L10 best:  {best:.4f}%")
        print(f"Random L10 final: {final:.4f}%")
        print(f"Random L9 ref:    71.99%")
        diff = best - 71.99
        print(f"Diferenta:        {diff:+.4f} pp")
        print(f"Timp total:       {elapsed/60:.1f} min")

        eps    = np.array([h["ep"]      for h in history])
        cum    = np.array([h["cum_pct"] for h in history])
        ep_pct = np.array([h["ep_pct"]  for h in history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"Salvat → {args.out}")

if __name__ == "__main__":
    main()
