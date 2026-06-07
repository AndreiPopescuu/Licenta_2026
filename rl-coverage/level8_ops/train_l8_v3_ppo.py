"""train_l8_v3_ppo.py — PPO with extended observation space (32 dims).

Changes vs train_l8_dynamic_ppo_v2.py:
  - IbexL8V3Env: 25 modules + n_ep_frac + action history (32 dims total)
  - larger net_arch: [512, 256]
  - ent_coef increased to 0.08 for more aggressive exploration

Usage:
    python train_l8_v3_ppo.py
    python train_l8_v3_ppo.py --episodes 1200 --steps 256
    python train_l8_v3_ppo.py --episodes 1200 --steps 64
    python train_l8_v3_ppo.py --episodes 1200 --hits l8_pipeline_hits.pkl
"""

import argparse, pickle, re, sys, time
from pathlib import Path
import numpy as np

_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def norm(key):
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8_v3 import IbexL8V3Env, MODULES
from utils_l8 import run_raw

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


class Log(BaseCallback):
    def __init__(self, baseline_pct: float, saturation_window: int = 50,
                 save_hits_path: str | None = None,
                 corpus_path: str | None = None):
        super().__init__()
        self.baseline_pct      = baseline_pct
        self.saturation_window = saturation_window
        self.save_hits_path    = save_hits_path
        self.corpus_path       = corpus_path
        self.history           = []
        self._zero_streak      = 0
        self._corpus           = []

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

        words = info.get("ep_words")
        if words and self.corpus_path:
            self._corpus.append({
                "ep": ep, "words": words,
                "new_hits": new, "cum_pct": round(cum, 3),
            })
            self._save_corpus(cum)

        self._zero_streak = self._zero_streak + 1 if new == 0 else 0
        if self._zero_streak >= self.saturation_window:
            print(f"\n[!] Saturație: {self.saturation_window} ep fără hits noi. Opresc.", flush=True)
            self._save_hits()
            return False

        return True

    def _save_corpus(self, cum_pct: float):
        import json
        with open(self.corpus_path, "w") as f:
            json.dump({
                "total_programs": len(self._corpus),
                "final_cum_pct":  round(cum_pct, 3),
                "programs":       self._corpus,
            }, f, indent=2)

    def _load_corpus(self):
        import json
        from pathlib import Path
        if self.corpus_path and Path(self.corpus_path).exists():
            with open(self.corpus_path) as f:
                data = json.load(f)
            self._corpus = data.get("programs", [])

    def _save_hits(self):
        if not self.save_hits_path:
            return
        env = self.training_env.envs[0].unwrapped
        normalized = {norm(k) for k in env._cum_hits}
        with open(self.save_hits_path, "wb") as f:
            pickle.dump(normalized, f)
        cum = self.history[-1]["cum_pct"] if self.history else 0
        print(f"  Hits salvate → {self.save_hits_path}  ({cum:.2f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--steps",    type=int, default=256,
                    help="Instrucțiuni per episod (încearcă și 64, 128)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits pre-acumulate (opțional)")
    ap.add_argument("--out",        default="l8_v3_ppo_curve.npz")
    ap.add_argument("--save-hits",  default="l8_final_hits.pkl",
                    help="Salvează hits finale pentru pipeline L9")
    ap.add_argument("--save-model", default="l8_v3_ppo_model.zip",
                    help="Salvează modelul PPO antrenat")
    ap.add_argument("--saturation", type=int, default=50,
                    help="Oprește după N ep consecutive fără hits noi")
    ap.add_argument("--corpus-out", default="../corpus_all.json",
                    help="Salvează episoadele cu bins noi în JSON")
    args = ap.parse_args()

    print("=" * 64)
    print(f"L8 V3 PPO — obs 32 dims, {args.steps} steps/episode")
    print("=" * 64)

    initial_hits = set()
    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"\n  Hits pre-încărcate: {len(initial_hits):,} (din {args.hits})")

    env = IbexL8V3Env(
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
    print(f"  Obs dims:      35  (vs 10 în v2)")
    print(f"  Baseline:      {baseline_pct:.2f}%")
    print(f"  Ceiling L8:   ~74.72%")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10} | worst module")
    print("-" * 72)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.steps,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.08,
        policy_kwargs=dict(net_arch=[512, 256]),
        verbose=0, seed=args.seed, device="cpu",
    )

    cb = Log(baseline_pct,
             saturation_window=args.saturation,
             save_hits_path=args.save_hits,
             corpus_path=str(THIS / args.corpus_out))
    cb._load_corpus()
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    if cb.history:
        best  = max(h["cum_pct"] for h in cb.history)
        final = cb.history[-1]["cum_pct"]
        print(f"\nRezultate finale:")
        print(f"  Baseline:          {baseline_pct:.2f}%")
        print(f"  V3 PPO best:       {best:.2f}%")
        print(f"  V3 PPO final:      {final:.2f}%")
        print(f"  Câștig:           {final - baseline_pct:+.2f} pp")
        print(f"  Ceiling L8:       ~74.72%")

        eps    = np.array([h["ep"]      for h in cb.history])
        cum    = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"]  for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")
        cb._save_hits()

    model.save(args.save_model)
    print(f"  Model salvat → {args.save_model}")


if __name__ == "__main__":
    main()
