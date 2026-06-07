"""build_corpus.py — rulează env L8 și salvează programele care aduc bins noi.

Usage:
    python build_corpus.py --episodes 300 --out corpus_l8.json
    python build_corpus.py --episodes 300 --model <model.zip> --out corpus_l8.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from codec_l8 import emit_program
from env_l8_v3 import IbexL8V3Env, run_program
from utils_l8 import run_raw

_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def norm(key):
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--model",    default=None, help="Fișier .zip PPO (opțional)")
    ap.add_argument("--out",      default="../corpus_all.json")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    env = IbexL8V3Env(episode_steps=args.steps)

    s = run_raw([0x00000013] * 8)
    TOTAL = s.by_kind["toggle"][1] if s else 20248
    print(f"Total bins toggle detectate: {TOTAL}")

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, env=env, device="cpu")
        print(f"Model încărcat: {args.model}")
    else:
        print("Mod: random agent")

    cum_hits = set()
    corpus   = []
    out_path = THIS / args.out

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        corpus = existing.get("programs", [])
        print(f"Append la corpus existent: {len(corpus)} programe deja salvate")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info.get("vtop_failed"):
            print(f"  ep {ep:>4}: Vtop failed, skip")
            continue

        # rulam episodul izolat ca sa obtinem hits-urile lui proprii
        summary = run_program(env._actions)
        if summary is None:
            print(f"  ep {ep:>4}: run_program failed, skip")
            continue

        ep_hits  = {norm(k) for k, v in summary.points.items()
                    if v > 0 and "\x01page\x02v_toggle/" in ("\x01" + k)}
        new_hits = ep_hits - cum_hits
        cum_hits |= ep_hits

        cum_pct = 100.0 * len(cum_hits) / TOTAL
        salvat  = "DA" if new_hits else "nu"
        print(f"  ep {ep:>4}: +{len(new_hits):>4} bins noi  |  cum {cum_pct:.2f}%  |  salvat: {salvat}")

        if new_hits:
            words = emit_program(env._actions)
            corpus.append({
                "ep":       ep,
                "words":    [int(w) for w in words],
                "new_hits": len(new_hits),
                "cum_pct":  round(cum_pct, 3),
            })
            with open(out_path, "w") as f:
                json.dump({
                    "total_programs": len(corpus),
                    "final_cum_pct":  round(cum_pct, 3),
                    "programs":       corpus,
                }, f, indent=2)

    with open(out_path, "w") as f:
        json.dump({
            "total_programs": len(corpus),
            "final_cum_pct":  round(100.0 * len(cum_hits) / TOTAL, 3),
            "programs":       corpus,
        }, f, indent=2)

    print(f"\nCorpus salvat: {out_path}")
    print(f"  Programe utile: {len(corpus)} / {args.episodes}")
    print(f"  Coverage final: {100.0 * len(cum_hits) / TOTAL:.2f}%")


if __name__ == "__main__":
    main()
