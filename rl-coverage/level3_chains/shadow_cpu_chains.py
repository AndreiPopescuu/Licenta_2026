"""
Coverage model with K=3 RAW dependency chains.

A K=3 chain is three consecutive instructions where the middle one reads from
the first's write AND writes a new value that the third reads:

    i-2: W_outer writes X
    i-1: W_mid reads X, writes Y            (must be R-type: reader AND writer)
    i  : R reads Y

The middle instruction must be R-type (only R-type reads *and* writes rd).
The outer can be any writer (R-type or JAL). The tail can be any reader
(R-type or S-type).

Bin count (482 easy + 1300 chain = 1782 total):
  SEEN, ZERO_DST, ZERO_SRC, SAME_SRC, BR, RAW_DIST_{1,2,3}   482  (as before)
  K3_CHAIN: 10 W_outer (R-type, non-JAL here) × 10 W_mid × 13 R  1300

(We keep no-JAL throughout so the model validates on real RTL without
control-flow desynchronization; K=3 chain bins with JAL as W_outer would
require an assembler that handles branch targets.)

Why this is a better benchmark than the 482-bin one:
- At 100K rollout steps, random covers ~15% of the 1300 chain bins
  (each chain fires with P ≈ 1.7e-6 per step offset; expected hits = 0.17;
   Poisson miss prob = 0.84). Random plateau ≈ 15%.
- At 1M steps, ~81% — still not saturated.
- PPO that deliberately chains writes CAN hit every specific triple
  probabilistically per episode. Expected PPO advantage: multiple x at
  realistic training budgets.
"""

from enum import IntEnum
from collections import deque


class Op(IntEnum):
    ADD = 0; SUB = 1; SLL = 2; SLT = 3; SLTU = 4
    XOR = 5; SRL = 6; SRA = 7; OR = 8; AND = 9
    SB = 10; SH = 11; SW = 12


OP_NAMES = [op.name.lower() for op in Op]
R_TYPE = {Op.ADD, Op.SUB, Op.SLL, Op.SLT, Op.SLTU, Op.XOR, Op.SRL, Op.SRA, Op.OR, Op.AND}
S_TYPE = {Op.SB, Op.SH, Op.SW}

WRITERS = sorted(R_TYPE)          # 10 (no JAL in this benchmark)
READERS = sorted(R_TYPE | S_TYPE) # 13
CHAIN_MIDDLES = sorted(R_TYPE)    # 10 — must be reader AND writer
RAW_MAX_DIST = 3


def _build_bin_index():
    bins: list[str] = []
    for op in Op:           bins.append(f"{op.name.lower()}_seen")
    for op in WRITERS:      bins.append(f"{op.name.lower()}_zero_dst")
    for op in READERS:      bins.append(f"{op.name.lower()}_zero_src")
    for op in READERS:      bins.append(f"{op.name.lower()}_same_src")
    for d in range(1, RAW_MAX_DIST + 1):
        for w in WRITERS:
            for r in READERS:
                bins.append(f"{w.name.lower()}->{r.name.lower()}_raw_dist{d}")
    for o in WRITERS:
        for m in CHAIN_MIDDLES:
            for r in READERS:
                bins.append(f"{o.name.lower()}->{m.name.lower()}->{r.name.lower()}_k3_chain")
    return bins


BIN_NAMES = _build_bin_index()
N_BINS = len(BIN_NAMES)
BIN_IDX = {n: i for i, n in enumerate(BIN_NAMES)}
EXPECTED = 13 + 10 + 13 + 13 + 3 * 10 * 13 + 10 * 10 * 13  # 13+10+13+13+390+1300 = 1739
assert N_BINS == EXPECTED, f"expected {EXPECTED} bins, got {N_BINS}"

_SEEN_IDX = {op: BIN_IDX[f"{op.name.lower()}_seen"] for op in Op}
_ZERO_DST_IDX = {op: BIN_IDX[f"{op.name.lower()}_zero_dst"] for op in WRITERS}
_ZERO_SRC_IDX = {op: BIN_IDX[f"{op.name.lower()}_zero_src"] for op in READERS}
_SAME_SRC_IDX = {op: BIN_IDX[f"{op.name.lower()}_same_src"] for op in READERS}
_RAW_IDX = {
    (d, w, r): BIN_IDX[f"{w.name.lower()}->{r.name.lower()}_raw_dist{d}"]
    for d in range(1, RAW_MAX_DIST + 1) for w in WRITERS for r in READERS
}
_K3_IDX = {
    (o, m, r): BIN_IDX[f"{o.name.lower()}->{m.name.lower()}->{r.name.lower()}_k3_chain"]
    for o in WRITERS for m in CHAIN_MIDDLES for r in READERS
}


class ChainHistory:
    """Tracks per-register last-write info AND the last 2 instruction tuples
    (needed to detect K=3 chains)."""

    def __init__(self):
        # per-register: (writer, age) for RAW_DIST bins
        self.writer: list[Op | None] = [None] * 32
        self.age: list[int] = [0] * 32
        # Last 2 instructions (for chain detection). Each: (op, rd, rs1, rs2)
        self.last_two: deque = deque(maxlen=2)

    def reset(self):
        for i in range(32):
            self.writer[i] = None
            self.age[i] = 0
        self.last_two.clear()


def bins_for_step(op: int, rd: int, rs1: int, rs2: int,
                  hist: ChainHistory) -> list[int]:
    op_e = Op(op)
    hits: list[int] = [_SEEN_IDX[op_e]]

    if op_e in WRITERS and rd == 0:
        hits.append(_ZERO_DST_IDX[op_e])
    if op_e in READERS and (rs1 == 0 or rs2 == 0):
        hits.append(_ZERO_SRC_IDX[op_e])
    if op_e in READERS and rs1 == rs2:
        hits.append(_SAME_SRC_IDX[op_e])

    # RAW hazards at distance 1..3
    if op_e in READERS:
        for src in (rs1, rs2):
            w, age = hist.writer[src], hist.age[src]
            if w is not None and 1 <= age <= RAW_MAX_DIST:
                hits.append(_RAW_IDX[(age, w, op_e)])

    # K=3 chain check: requires last 2 instructions recorded
    if op_e in READERS and len(hist.last_two) == 2:
        i1 = hist.last_two[-1]   # i-1 (nearest)
        i2 = hist.last_two[-2]   # i-2 (oldest)
        i1_op, i1_rd, i1_rs1, i1_rs2 = i1
        i2_op, i2_rd, i2_rs1, i2_rs2 = i2
        # i-2 must be a writer; i-1 must be R-type (reader+writer); current must read i-1's rd
        if (i2_op in WRITERS and i1_op in CHAIN_MIDDLES
                and (i1_rs1 == i2_rd or i1_rs2 == i2_rd)
                and (rs1 == i1_rd or rs2 == i1_rd)):
            hits.append(_K3_IDX[(i2_op, i1_op, op_e)])

    return hits


def advance_history(op: int, rd: int, rs1: int, rs2: int, hist: ChainHistory):
    """Call AFTER bins_for_step, to update state for the next instruction."""
    op_e = Op(op)
    # Age all per-register writes
    for i in range(32):
        if hist.age[i] > 0:
            hist.age[i] += 1
            if hist.age[i] > RAW_MAX_DIST:
                hist.writer[i] = None
                hist.age[i] = 0
    # Record new write
    if op_e in WRITERS:
        hist.writer[rd] = op_e
        hist.age[rd] = 1
    # Push onto last_two ring
    hist.last_two.append((op_e, rd, rs1, rs2))


if __name__ == "__main__":
    print(f"Total bins: {N_BINS}")
    print(f"  Easy (SEEN/ZERO/SAME): {13+10+13+13} = 49")
    print(f"  RAW dist 1/2/3:        {3*10*13} = 390")
    print(f"  K=3 chains:            {10*10*13} = 1300")
    print(f"Theoretical random-hit probability per K=3 chain bin per step: ~1.7e-6")
