"""replay_corpus.py — replays corpus L8 și raportează coverage final.

Regression test: dacă coverage-ul replayat atinge pragul așteptat, testul trece.

Usage:
    python replay_corpus.py --corpus corpus_l8.json
    python replay_corpus.py --corpus corpus_l8.json --expected 70.0
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

THIS   = Path(__file__).resolve().parent
ML4DV  = (THIS.parent.parent / "cpu").resolve()
VTOP   = ML4DV / "sim_build" / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_replay.json"

sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser

_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def norm(key):
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k


def run_words(words: list[int]):
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(words), "agent": "l8",
                   "machine_code": [int(w) for w in words]}, f)

    env = os.environ.copy()
    cocotb_libs = "/home/andrei/ibex_env/lib/python3.12/site-packages/cocotb/libs"
    env["LD_LIBRARY_PATH"] = (
        cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = (
        str(ML4DV) + ":/home/andrei/ibex_env/lib/python3.12/site-packages"
        + ":" + env.get("PYTHONPATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON

    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",   required=True)
    ap.add_argument("--expected", type=float, default=None,
                    help="Coverage minim așteptat (%). Test eșuează dacă nu e atins.")
    args = ap.parse_args()

    with open(args.corpus) as f:
        corpus = json.load(f)

    programs  = corpus["programs"]
    total_prg = len(programs)

    s = run_words([0x00000013] * 8)
    TOTAL = s.by_kind["toggle"][1] if s else 20248
    print(f"Total bins toggle detectate: {TOTAL}")

    print(f"Corpus: {total_prg} programe")
    print(f"Coverage așteptat din corpus: {corpus['final_cum_pct']}%\n")

    cum_hits = set()
    failed   = 0

    for i, prog in enumerate(programs):
        summary = run_words(prog["words"])

        if summary is None:
            print(f"  [{i+1:>3}/{total_prg}] ep={prog['ep']:>4}: Vtop FAILED")
            failed += 1
            continue

        ep_hits  = {norm(k) for k, v in summary.points.items()
                    if v > 0 and "\x01page\x02v_toggle/" in ("\x01" + k)}
        new_hits = ep_hits - cum_hits
        cum_hits |= ep_hits

        cum_pct = 100.0 * len(cum_hits) / TOTAL
        print(f"  [{i+1:>3}/{total_prg}] ep={prog['ep']:>4}: "
              f"+{len(new_hits):>4} bins  |  cum {cum_pct:.2f}%")

    final_pct = 100.0 * len(cum_hits) / TOTAL
    print(f"\n{'='*50}")
    print(f"Coverage final replayat:  {final_pct:.2f}%  ({len(cum_hits):,} / {TOTAL:,} bins)")
    print(f"Coverage din corpus:      {corpus['final_cum_pct']}%")
    print(f"Programe eșuate:          {failed} / {total_prg}")

    if args.expected is not None:
        if final_pct >= args.expected:
            print(f"\n[PASS] {final_pct:.2f}% >= {args.expected}% așteptat")
            sys.exit(0)
        else:
            print(f"\n[FAIL] {final_pct:.2f}% < {args.expected}% așteptat")
            sys.exit(1)


if __name__ == "__main__":
    main()
