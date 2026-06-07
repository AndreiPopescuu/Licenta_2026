"""train_l8_dynamic_ppo_v2.py — PPO cu weights dinamice, episoade de 256 pași.

Față de v1 (train_l8_dynamic_ppo.py):
  - episode_steps = 256 (în loc de 1024)
  - ~4x mai multe episoade în același timp de simulare
  - n_steps=256, batch_size=64 ajustate la episoadele scurte
  - default 1200 episoade (≈ același walltime ca 300 × 1024)

Usage:
    python train_l8_dynamic_ppo_v2.py
    python train_l8_dynamic_ppo_v2.py --episodes 1200
    python train_l8_dynamic_ppo_v2.py --episodes 1200 --hits l8_pipeline_hits.pkl
"""

import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8_dynamic import IbexL8DynamicEnv, TRACKED_MODULES
from utils_l8 import run_raw

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


class Log(BaseCallback):
    def __init__(self, baseline_pct: float, saturation_window: int = 50,
                 save_hits_path: str | None = None):
        super().__init__()
        self.baseline_pct      = baseline_pct
        self.saturation_window = saturation_window
        self.save_hits_path    = save_hits_path
        self.history           = []
        self._zero_streak      = 0

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True
        ep      = len(self.history) + 1
        cum     = info["cum_pct"]
        new     = info.get("new_hits_vs_cum", 0)
        worst   = info.get("worst_mod", "?")
        worst_p = info.get("worst_pct", 0.0)
        self.history.append({
            "ep": ep, "cum_pct": cum,
            "ep_pct": info["ep_pct"], "new_hits": new,
        })
        delta = cum - self.baseline_pct
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
              f"new {new:>4} | Δ {delta:>+6.2f}pp | worst: {worst} {worst_p:.1f}%",
              flush=True)

        # Saturație: N episoade consecutive fără niciun hit nou
        self._zero_streak = self._zero_streak + 1 if new == 0 else 0
        if self._zero_streak >= self.saturation_window:
            print(f"\n[!] Saturație detectată: {self.saturation_window} ep fără hits noi. Opresc.", flush=True)
            self._save_hits()
            return False

        return True

    def _save_hits(self):
        if not self.save_hits_path:
            return
        env = self.training_env.envs[0]
        with open(self.save_hits_path, "wb") as f:
            pickle.dump(env._cum_hits, f)
        cum = self.history[-1]["cum_pct"] if self.history else 0
        print(f"  Hits salvate → {self.save_hits_path}  ({cum:.2f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200,
                    help="Episoade PPO (default 1200 ≈ 300×1024 în walltime)")
    ap.add_argument("--steps",    type=int, default=256,
                    help="Instrucțiuni per episod")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits pre-acumulate (opțional)")
    ap.add_argument("--out",       default="l8_dynamic_ppo_v2_curve.npz")
    ap.add_argument("--save-hits", default="l8_final_hits.pkl",
                    help="Salvează hits finale în acest fișier (pentru pipeline L9)")
    ap.add_argument("--saturation", type=int, default=50,
                    help="Oprește după N episoade consecutive fără hits noi")
    args = ap.parse_args()

    print("=" * 64)
    print("L8 Dynamic PPO v2 — 256 pași/episod, weights dinamice")
    print("=" * 64)

    initial_hits = set()
    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"\n  Hits pre-încărcate: {len(initial_hits):,} (din {args.hits})")

    env = IbexL8DynamicEnv(
        episode_steps=args.steps,
        seed=args.seed,
        initial_hits=initial_hits,
    )

    s = run_raw([0x00000013] * 8)
    total_tog    = s.by_kind["toggle"][1] if s else 1
    baseline_pct = 100. * len(initial_hits) / max(total_tog, 1)
    env._total_tog = total_tog

    print(f"  Episoade:      {args.episodes}")
    print(f"  Pași/episod:   {args.steps}")
    print(f"  Total steps:   {args.episodes * args.steps:,}")
    print(f"  Baseline:      {baseline_pct:.2f}%")
    print(f"  Ceiling L8:   ~74.72%")
    print(f"  Gap:           {74.72 - baseline_pct:.2f} pp")
    print(f"\nPPO dynamic v2 — {args.episodes} ep × {args.steps} pași")
    print(f"{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10} | worst module")
    print("-" * 72)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.steps,        # rollout buffer = 1 episod exact
        batch_size=64,             # mai mic față de v1 (256) pentru n_steps=256
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0, seed=args.seed, device="cpu",
    )

    cb = Log(baseline_pct,
             saturation_window=args.saturation,
             save_hits_path=args.save_hits)
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    if cb.history:
        best  = max(h["cum_pct"] for h in cb.history)
        final = cb.history[-1]["cum_pct"]
        print(f"\nRezultate finale:")
        print(f"  Baseline:                  {baseline_pct:.2f}%")
        print(f"  Dynamic PPO v2 best:       {best:.2f}%")
        print(f"  Dynamic PPO v2 final:      {final:.2f}%")
        print(f"  Câștig față de baseline:  {final - baseline_pct:+.2f} pp")
        print(f"  Ceiling L8:               ~74.72%")

        eps    = np.array([h["ep"]      for h in cb.history])
        cum    = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"]  for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")
        cb._save_hits()


if __name__ == "__main__":
    main()
