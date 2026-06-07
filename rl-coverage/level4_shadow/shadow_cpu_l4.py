"""
Level 6 shadow — 30-op action space with rich functional coverage, targeting
the brief's "5K-10K Python-defined bins" ambition.

Ops (30 total, no JAL for shadow<->real parity):
  R-type ALU    (10) — ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
  I-type ALU     (9) — ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
  Loads          (5) — LB, LH, LW, LBU, LHU
  Stores         (3) — SB, SH, SW
  CSR            (3) — CSRRW, CSRRS, CSRRC

Coverage bins (5,885 total):
  SEEN              30  — each op executed at least once
  ZERO_DST          24  — writers with rd=x0 (R+I+Loads+CSR, not stores)
  ZERO_SRC          27  — readers with rs1=x0 or rs2=x0 (all readers)
  SAME_SRC          13  — readers with 2 sources and rs1==rs2 (R+S types)
  IMM_BUCKETS       85  — 17 imm-ops × 5 imm buckets (neg-max/small-neg/0/small-pos/pos-max)
  RAW_DIST_1/2/3 2,700  — 30 writers × 30 readers × 3 distances
  K3_CHAINS      3,000  — R × R × any_reader (the structurally hard bins)

                Total:  5,879 (see assertion below)

Writer/reader taxonomy:
  Writes rd:  R-type, I-type ALU, Loads, CSR            (27 ops)
  Reads rs1:  all except stores & CSR-immediate         (27 ops)
  Reads rs2:  R-type, Stores                             (13 ops)
  Reader+writer (eligible for K=3 middle): R, I, Loads, CSR (27 ops)

Why K=3 chains stay hard for random at this scale:
  With 30 ops, P(specific W_outer) = 1/30, P(specific W_mid) = 1/30,
  P(specific R_tail) = 1/30. Register match P ~ 0.061 per source.
  Per-chain P = 1/30^3 × 0.061^2 ≈ 1.4e-7 per step.
  At 100K steps: expected hits per chain bin ≈ 0.014 → random coverage ~1.4%.
  At 1M steps: ~13%. At 10M steps: ~75%. At real-Verilator speeds (~630
  insn/sec) 10M steps = 4.4h wall — random cannot close this in a working day.
"""

from enum import IntEnum
from collections import deque


class Op(IntEnum):
    # R-type ALU (10)
    ADD = 0;   SUB = 1;   SLL = 2;   SLT = 3;   SLTU = 4
    XOR = 5;   SRL = 6;   SRA = 7;   OR = 8;    AND = 9
    # I-type ALU (9)
    ADDI = 10; SLTI = 11; SLTIU = 12; XORI = 13; ORI = 14; ANDI = 15
    SLLI = 16; SRLI = 17; SRAI = 18
    # Loads (5)
    LB = 19;   LH = 20;   LW = 21;   LBU = 22;  LHU = 23
    # Stores (3)
    SB = 24;   SH = 25;   SW = 26
    # CSR (3)
    CSRRW = 27; CSRRS = 28; CSRRC = 29


R_TYPE   = {Op.ADD, Op.SUB, Op.SLL, Op.SLT, Op.SLTU, Op.XOR, Op.SRL, Op.SRA, Op.OR, Op.AND}
I_ALU    = {Op.ADDI, Op.SLTI, Op.SLTIU, Op.XORI, Op.ORI, Op.ANDI, Op.SLLI, Op.SRLI, Op.SRAI}
LOADS    = {Op.LB, Op.LH, Op.LW, Op.LBU, Op.LHU}
STORES   = {Op.SB, Op.SH, Op.SW}
CSR_OPS  = {Op.CSRRW, Op.CSRRS, Op.CSRRC}

WRITERS        = sorted(R_TYPE | I_ALU | LOADS | CSR_OPS)      # 27
READERS        = sorted(R_TYPE | I_ALU | LOADS | STORES | CSR_OPS)  # 30
READERS_TWO    = sorted(R_TYPE | STORES)                       # 13 (read both rs1 and rs2)
READERS_ONE    = sorted(I_ALU | LOADS | CSR_OPS)               # 17 (read only rs1)
IMM_OPS        = sorted(I_ALU | LOADS | STORES)                # 17 (have imm field)
CHAIN_R_MID    = sorted(R_TYPE)                                # 10 (use both sources, fast middle)
CHAIN_R_OUT    = sorted(R_TYPE)                                # 10

RAW_MAX_DIST = 3
IMM_BUCKETS = 5   # 0=neg-max, 1=small-neg, 2=zero, 3=small-pos, 4=pos-max


def reads_rs1(op: Op) -> bool:
    return op in R_TYPE or op in I_ALU or op in LOADS or op in STORES or op in CSR_OPS

def reads_rs2(op: Op) -> bool:
    return op in R_TYPE or op in STORES

def writes_rd(op: Op) -> bool:
    return op in R_TYPE or op in I_ALU or op in LOADS or op in CSR_OPS

def has_imm(op: Op) -> bool:
    return op in I_ALU or op in LOADS or op in STORES


def _build_bins():
    bins: list[str] = []
    # SEEN
    for op in Op:                          bins.append(f"{op.name.lower()}_seen")
    # ZERO_DST (writers)
    for op in WRITERS:                     bins.append(f"{op.name.lower()}_zero_dst")
    # ZERO_SRC (all readers)
    for op in READERS:                     bins.append(f"{op.name.lower()}_zero_src")
    # SAME_SRC (two-source readers)
    for op in READERS_TWO:                 bins.append(f"{op.name.lower()}_same_src")
    # IMM_BUCKETS
    for op in IMM_OPS:
        for b in range(IMM_BUCKETS):       bins.append(f"{op.name.lower()}_imm{b}")
    # RAW_DIST 1..3 × writers × readers
    for d in range(1, RAW_MAX_DIST + 1):
        for w in WRITERS:
            for r in READERS:
                bins.append(f"{w.name.lower()}->{r.name.lower()}_raw_dist{d}")
    # K=3 R-type chains: R outer × R middle × any reader tail
    for o in CHAIN_R_OUT:
        for m in CHAIN_R_MID:
            for r in READERS:
                bins.append(f"{o.name.lower()}->{m.name.lower()}->{r.name.lower()}_k3chain")
    return bins


BIN_NAMES = _build_bins()
N_BINS = len(BIN_NAMES)
BIN_IDX = {n: i for i, n in enumerate(BIN_NAMES)}
EXPECTED = 30 + 27 + 30 + 13 + 17 * 5 + 27 * 30 * 3 + 10 * 10 * 30
assert N_BINS == EXPECTED, f"expected {EXPECTED}, got {N_BINS}"

_SEEN_IDX = {op: BIN_IDX[f"{op.name.lower()}_seen"] for op in Op}
_ZERO_DST_IDX = {op: BIN_IDX[f"{op.name.lower()}_zero_dst"] for op in WRITERS}
_ZERO_SRC_IDX = {op: BIN_IDX[f"{op.name.lower()}_zero_src"] for op in READERS}
_SAME_SRC_IDX = {op: BIN_IDX[f"{op.name.lower()}_same_src"] for op in READERS_TWO}
_IMM_IDX = {(op, b): BIN_IDX[f"{op.name.lower()}_imm{b}"] for op in IMM_OPS for b in range(IMM_BUCKETS)}
_RAW_IDX = {(d, w, r): BIN_IDX[f"{w.name.lower()}->{r.name.lower()}_raw_dist{d}"]
            for d in range(1, RAW_MAX_DIST + 1) for w in WRITERS for r in READERS}
_K3_IDX = {(o, m, r): BIN_IDX[f"{o.name.lower()}->{m.name.lower()}->{r.name.lower()}_k3chain"]
           for o in CHAIN_R_OUT for m in CHAIN_R_MID for r in READERS}


class L6History:
    """Tracks, per register, the most recent writer + age; plus last 2 (op, rd, rs1, rs2) for chains."""

    def __init__(self):
        self.writer: list[Op | None] = [None] * 32
        self.age: list[int] = [0] * 32
        self.last_two: deque = deque(maxlen=2)

    def reset(self):
        for i in range(32):
            self.writer[i] = None
            self.age[i] = 0
        self.last_two.clear()


def bins_for_step(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int,
                  hist: L6History) -> list[int]:
    op = Op(op_i)
    hits: list[int] = [_SEEN_IDX[op]]

    if writes_rd(op) and rd == 0:
        hits.append(_ZERO_DST_IDX[op])
    # ZERO_SRC: fires if the op reads an x0 source
    if op in READERS:
        if (reads_rs1(op) and rs1 == 0) or (reads_rs2(op) and rs2 == 0):
            hits.append(_ZERO_SRC_IDX[op])
    if op in READERS_TWO and rs1 == rs2:
        hits.append(_SAME_SRC_IDX[op])
    if has_imm(op):
        hits.append(_IMM_IDX[(op, imm_bucket)])

    # RAW hazards at distance 1..3 on any source register that matches
    if op in READERS:
        srcs = []
        if reads_rs1(op): srcs.append(rs1)
        if reads_rs2(op): srcs.append(rs2)
        for src in srcs:
            w, age = hist.writer[src], hist.age[src]
            if w is not None and 1 <= age <= RAW_MAX_DIST:
                hits.append(_RAW_IDX[(age, w, op)])

    # K=3 chains: R outer → R middle → reader tail with register pass-through
    if op in READERS and len(hist.last_two) == 2:
        i1 = hist.last_two[-1]       # (op, rd, rs1, rs2)
        i2 = hist.last_two[-2]
        i1_op, i1_rd, i1_rs1, i1_rs2 = i1
        i2_op, i2_rd, _, _ = i2
        if (i2_op in R_TYPE and i1_op in R_TYPE
                and (i1_rs1 == i2_rd or i1_rs2 == i2_rd)):
            # does current read i1's rd?
            srcs = []
            if reads_rs1(op): srcs.append(rs1)
            if reads_rs2(op): srcs.append(rs2)
            if i1_rd in srcs:
                hits.append(_K3_IDX[(i2_op, i1_op, op)])

    return hits


def advance_history(op_i: int, rd: int, rs1: int, rs2: int, hist: L6History):
    op = Op(op_i)
    for i in range(32):
        if hist.age[i] > 0:
            hist.age[i] += 1
            if hist.age[i] > RAW_MAX_DIST:
                hist.writer[i] = None
                hist.age[i] = 0
    if writes_rd(op):
        hist.writer[rd] = op
        hist.age[rd] = 1
    hist.last_two.append((op, rd, rs1, rs2))


if __name__ == "__main__":
    print(f"Total bins: {N_BINS}")
    print(f"  SEEN={30}  ZERO_DST={27}  ZERO_SRC={30}  SAME_SRC={13}")
    print(f"  IMM_BUCKETS={17*5}={17*5}   RAW_DIST_*={27*30*3}   K=3_chains={10*10*30}")
    print(f"\nWriters: {len(WRITERS)}, Readers: {len(READERS)}")
    print(f"K=3 chain-eligible middle: {len(CHAIN_R_MID)} (R-type)")
    p_chain = (1/30)**3 * 0.061**2
    print(f"\nRandom K=3 hit-prob per chain bin per step: {p_chain:.2e}")
    print(f"  at 100K samples: expected coverage ≈ {100*(1 - 2.71828**(-p_chain*100000)):.1f}% of chains")
    print(f"  at 1M samples:   expected coverage ≈ {100*(1 - 2.71828**(-p_chain*1_000_000)):.1f}%")
    print(f"  at 10M samples:  expected coverage ≈ {100*(1 - 2.71828**(-p_chain*10_000_000)):.1f}%")
