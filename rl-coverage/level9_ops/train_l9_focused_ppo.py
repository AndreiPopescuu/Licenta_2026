"""train_l9_focused_ppo.py — PPO pe L9 focalizat.

Față de train_l9_v2_ppo:
  - IbexL9FocusedEnv: 30 ops (în loc de 83), csr_bucket explicit (6 dims)
  - Reward include branch coverage (0.3×) + episode_base

Usage:
    python train_l9_focused_ppo.py --hits ../level8_ops/l8_pipeline_hits.pkl --episodes 3600
    python train_l9_focused_ppo.py --resume --episodes 3600
"""

import argparse, pickle, signal, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l9_focused import IbexL9FocusedEnv, MODULES, N_OBS

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)

CKPT_MODEL   = THIS / "l9_focused_checkpoint_model"
CKPT_HITS    = THIS / "l9_focused_checkpoint_hits.pkl"
CKPT_HISTORY = THIS / "l9_focused_checkpoint_history.npz"

_stop_requested = False

def _sigint_handler(sig, frame):
    global _stop_requested
    print("\n[!] Stop cerut — salvez la finalul episodului curent...", flush=True)
    _stop_requested = True

signal.signal(signal.SIGINT, _sigint_handler)


class Log(BaseCallback):
    def __init__(self, baseline_pct: float, checkpoint_every: int = 100,
                 corpus_path: str | None = None):
        super().__init__()
        self.baseline_pct     = baseline_pct
        self.checkpoint_every = checkpoint_every
        self.corpus_path      = corpus_path
        self.history          = []
        self._corpus          = []

    def _on_step(self):
        global _stop_requested
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True

        ep         = len(self.history) + 1
        cum        = info["cum_pct"]
        new_tog    = info.get("new_hits_vs_cum", 0)
        new_branch = info.get("new_branch_hits", 0)
        worst      = info.get("worst_mod", "?")
        worst_p    = info.get("worst_pct", 0.0)
        self.history.append({
            "ep": ep, "cum_pct": cum,
            "ep_pct": info["ep_pct"], "new_hits": new_tog,
        })
        delta = cum - self.baseline_pct
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
              f"tog+{new_tog:<4} br+{new_branch:<3} | Δ {delta:>+6.2f}pp | "
              f"worst: {worst} {worst_p:.1f}%",
              flush=True)

        words = info.get("ep_words")
        if words and self.corpus_path:
            self._corpus.append({
                "ep": ep, "words": words,
                "new_hits": new_tog, "cum_pct": round(cum, 3),
            })
            self._save_corpus(cum)

        if ep % self.checkpoint_every == 0:
            self._save(ep, reason="checkpoint")

        if _stop_requested:
            self._save(ep, reason="interrupt")
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

    def _save(self, ep: int, reason: str = "checkpoint"):
        self.model.save(str(CKPT_MODEL))
        env = self.training_env.envs[0]
        if hasattr(env, "env"):
            env = env.env
        with open(CKPT_HITS, "wb") as f:
            pickle.dump(env._cum_hits, f)
        if self.history:
            eps    = np.array([h["ep"]      for h in self.history])
            cum    = np.array([h["cum_pct"] for h in self.history])
            ep_pct = np.array([h["ep_pct"]  for h in self.history])
            np.savez(CKPT_HISTORY, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  [{reason}] ep={ep} salvat → {CKPT_MODEL}.zip "
              f"({len(env._cum_hits):,} hits)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3600)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="l9_focused_ppo_curve.npz")
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--hits",     default=None,
                    help="Hits din L8 (l8_pipeline_hits.pkl) sau L9 v2")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--corpus-out", default="../corpus_all.json",
                    help="Fișier JSON pentru episoadele cu bins noi (append dacă există)")
    args = ap.parse_args()

    from codec_l9_focused import N_OPS, N_CSR_BUCKETS, IMM_BUCKETS
    print("=" * 70)
    print(f"L9 Focused PPO — {N_OPS} ops + csr_bucket, obs {N_OBS} dims, {args.steps} pași/ep")
    print(f"  Action space: [{N_OPS} ops, 32 rd, 32 rs1, 32 rs2, {IMM_BUCKETS} imm, {N_CSR_BUCKETS} csr_bucket]")
    print(f"  Reward: episode_base + toggle (weighted) + {0.3}× branch")
    print("=" * 70)

    initial_hits  = set()
    episodes_done = 0
    history_prev  = []

    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"  Hits pre-încărcate din {args.hits}: {len(initial_hits):,}")

    if args.resume and CKPT_HITS.exists() and CKPT_MODEL.with_suffix(".zip").exists():
        with open(CKPT_HITS, "rb") as f:
            initial_hits = pickle.load(f)
        if CKPT_HISTORY.exists():
            d = np.load(CKPT_HISTORY)
            episodes_done = int(d["ep"][-1])
            for i in range(len(d["ep"])):
                history_prev.append({
                    "ep": int(d["ep"][i]), "cum_pct": float(d["cum_pct"][i]),
                    "ep_pct": float(d["ep_pct"][i]), "new_hits": 0,
                })
        print(f"  Resume din ep {episodes_done} — {len(initial_hits):,} hits")
    else:
        if args.resume:
            print("  [!] Niciun checkpoint găsit — pornesc de la zero")

    remaining = args.episodes - episodes_done
    if remaining <= 0:
        print(f"  Deja {episodes_done} episoade, nimic de rulat."); return

    total_tog    = 20248
    baseline_pct = 100. * len(initial_hits) / total_tog if initial_hits else 0.0
    ent_coef     = args.ent_coef if args.ent_coef is not None else 0.08

    print(f"  Episoade totale: {args.episodes}  (rămase: {remaining})")
    print(f"  Pași/episod:     {args.steps}")
    print(f"  Net arch:        [512, 256]  ent_coef={ent_coef}")
    print(f"  Baseline:        {baseline_pct:.2f}%")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'tog+':>5} {'br+':>4} | {'Δbaseline':>10} | worst module")
    print("-" * 75)

    env = IbexL9FocusedEnv(
        episode_steps=args.steps,
        seed=args.seed,
        initial_hits=initial_hits,
    )

    if args.resume and CKPT_MODEL.with_suffix(".zip").exists():
        try:
            model = PPO.load(str(CKPT_MODEL), env=env)
            if args.ent_coef is not None:
                model.ent_coef = args.ent_coef
            print(f"  Model încărcat din {CKPT_MODEL}.zip")
        except Exception as e:
            print(f"  [!] Checkpoint incompatibil ({e}) — model nou")
            model = None
    else:
        model = None

    if model is None:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=args.steps,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=[512, 256]),
            verbose=0, seed=args.seed, device="cpu",
        )

    cb = Log(baseline_pct=baseline_pct, checkpoint_every=args.checkpoint_every,
             corpus_path=str(THIS / args.corpus_out))
    cb._load_corpus()
    cb.history = history_prev

    t0 = time.time()
    model.learn(total_timesteps=remaining * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    all_history = cb.history
    if all_history:
        best  = max(h["cum_pct"] for h in all_history)
        final = all_history[-1]["cum_pct"]
        print(f"\nRezultate finale:")
        print(f"  L9 focused best:  {best:.2f}%")
        print(f"  L9 focused final: {final:.2f}%")
        print(f"  L8 ceiling:      ~74.72%  (referință)")

        eps    = np.array([h["ep"]      for h in all_history])
        cum    = np.array([h["cum_pct"] for h in all_history])
        ep_pct = np.array([h["ep_pct"]  for h in all_history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
