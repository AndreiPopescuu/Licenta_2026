"""starter.py — Entry point for LLM-based coverage starting from L10 baseline.

This script:
  1. Loads the L10 baseline hits (14,404 bins = 71.99% coverage)
  2. Provides run_and_check() — run a program and see which NEW bins it covers

Usage:
    python starter.py                    # run a quick sanity check
    python starter.py --list             # list all accessible bins
"""

import sys, re, pickle, argparse, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
L5   = THIS.parent / "level5_real_rtl"
L9   = THIS.parent / "level9_ops"
sys.path.insert(0, str(L5))
sys.path.insert(0, str(L9))

import cov_parser
from env_l9_v2 import run_program

# ── Op constants ──────────────────────────────────────────────────────────────
ADD=0;  SUB=1;  SLL=2;  SLT=3;  SLTU=4; XOR=5;  SRL=6;  SRA=7;  OR=8;  AND=9
ADDI=10; SLTI=11; SLTIU=12; XORI=13; ORI=14; ANDI=15
SLLI=16; SRLI=17; SRAI=18
LB=19; LH=20; LW=21; LBU=22; LHU=23
SB=24; SH=25; SW=26
CSRRW=27; CSRRS=28; CSRRC=29
MUL=30; MULH=31; MULHSU=32; MULHU=33
DIV=34; DIVU=35; REM=36; REMU=37
BEQ=38; BNE=39; BLT=40; BGE=41; BLTU=42; BGEU=43
JAL=44; AUIPC=61; ECALL=62; EBREAK=63; LUI=64; JALR=65
MRET=70; WFI=71; FENCE_I=72
NOP = (ADDI, 0, 0, 0, 0)

# ── Stable ID ─────────────────────────────────────────────────────────────────
_PAGE_RE = re.compile(r"page\x02([^\x01]+)")
_SIG_RE  = re.compile(r"\x01o\x02([^\x01]+)")
_LINE_RE = re.compile(r"\x01l\x02([^\x01]+)")

def stable_id(k):
    pm = _PAGE_RE.search(k); sm = _SIG_RE.search(k); lm = _LINE_RE.search(k)
    return (f"{pm.group(1) if pm else '?'}"
            f"|{sm.group(1) if sm else '?'}"
            f"|{lm.group(1) if lm else '?'}")


# ── Load baseline ─────────────────────────────────────────────────────────────
BASELINE_PKL = THIS / "baseline_hits.pkl"
COVDAT       = THIS.parent.parent / "cpu" / "coverage.dat"

def load_baseline():
    with open(BASELINE_PKL, "rb") as f:
        hits = pickle.load(f)
    print(f"Baseline loaded: {len(hits):,} hits (71.99%)")
    return hits

def load_accessible_bins():
    bins = set()
    with open(THIS / "accessible_bins_for_llm.txt") as f:
        for line in f:
            m = re.match(r"\s{2}(\S+)\s{2,}(\S+)", line)
            if m:
                bins.add(m.group(2))
    return bins


# ── Core function: run a program and check new hits ───────────────────────────
def run_and_check(actions: list, baseline_ids: set, covdat_map: dict,
                  target_bins: set = None) -> dict:
    """
    Run a program and return newly covered bins.

    Args:
        actions:      list of (op, rd, rs1, rs2, imm_bucket) tuples
        baseline_ids: set of stable_ids already covered (from baseline pkl)
        covdat_map:   dict stable_id → signal_name (from coverage.dat)
        target_bins:  optional set of signal names to focus on

    Returns:
        dict with:
          'new_total':  count of new bins vs baseline
          'new_target': new bins that are in the accessible list
          'signals':    list of newly covered signal names
          'target_signals': list of newly covered target signal names
    """
    summary = run_program(actions)
    if summary is None:
        return {'new_total': 0, 'new_target': 0, 'signals': [], 'target_signals': []}

    run_hits_ids = {stable_id(k)
                    for k, v in summary.points.items()
                    if v > 0 and "v_toggle/" in k}

    new_ids = run_hits_ids - baseline_ids

    new_signals = []
    for sid in new_ids:
        if sid in covdat_map:
            new_signals.append(covdat_map[sid])

    new_target = [s for s in new_signals if s in target_bins] if target_bins else new_signals

    return {
        'new_total':      len(new_ids),
        'new_target':     len(new_target),
        'signals':        sorted(new_signals),
        'target_signals': sorted(new_target),
    }


# ── Setup helper ──────────────────────────────────────────────────────────────
def setup(covdat_path: str = None):
    """Load baseline and build covdat_map. Call once before run_and_check()."""
    baseline_hits = load_baseline()
    baseline_ids  = {stable_id(k) for k in baseline_hits}

    path = covdat_path or str(COVDAT)
    s = cov_parser.parse(path)
    prefix = "page\x02v_toggle/"
    covdat_map = {}
    for k in s.points:
        if prefix in k:
            sm = _SIG_RE.search(k)
            if sm:
                covdat_map[stable_id(k)] = sm.group(1)

    target_bins = load_accessible_bins()
    print(f"Target bins: {len(target_bins)}")
    return baseline_ids, covdat_map, target_bins


# ── Main (sanity check) ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list",   action="store_true", help="List accessible bins")
    ap.add_argument("--covdat", default=str(COVDAT))
    args = ap.parse_args()

    if args.list:
        bins = load_accessible_bins()
        print(f"Accessible bins ({len(bins)}):")
        for b in sorted(bins):
            print(f"  {b}")
        return

    print("=" * 60)
    print("LLM Coverage — starting from L10 baseline (71.99%)")
    print("=" * 60)

    baseline_ids, covdat_map, target_bins = setup(args.covdat)

    # Quick sanity check: run a NOP program
    actions = [NOP] * 10
    result  = run_and_check(actions, baseline_ids, covdat_map, target_bins)
    print(f"\nSanity check (10x NOP):")
    print(f"  New bins (total):  {result['new_total']}")
    print(f"  New bins (target): {result['new_target']}")
    print("\nEverything works. Implement your LLM loop in a new file.")
    print("Import: from starter import run_and_check, setup, NOP, LUI, CSRRW, ...")


if __name__ == "__main__":
    main()
