"""
Python shadow of the LLM4DV CPU-level 196-bin coverage model.

14 instructions (10 R-type, 3 stores, 1 JAL). Coverage types:
  SEEN         14 bins (one per instruction)
  ZERO_DST     11 bins (R-type + JAL -- ops that write rd)
  ZERO_SRC     13 bins (R-type + S-type -- ops that read source regs)
  SAME_SRC     13 bins (R-type + S-type -- need rs1==rs2)
  BR_FORWARDS  1 bin   (JAL with positive immediate)
  BR_BACKWARDS 1 bin   (JAL with negative immediate)
  RAW_HAZARD   143 bins -- (prev_writer, curr_reader) pairs where a source
               register of the current instr matches the rd of the previous.

Total: 14 + 11 + 13 + 13 + 1 + 1 + 143 = 196.

The RAW_HAZARD bins are what make this harder than the decoder: the
environment has *sequential* state (what the previous instruction wrote),
and the agent must deliberately chain writer->reader to trigger the bin.
Pure random has to rely on 1/32 register collisions.
"""

from enum import IntEnum


class Op(IntEnum):
    ADD = 0; SUB = 1; SLL = 2; SLT = 3; SLTU = 4
    XOR = 5; SRL = 6; SRA = 7; OR = 8; AND = 9
    SB = 10; SH = 11; SW = 12; JAL = 13


OP_NAMES = [op.name.lower() for op in Op]
R_TYPE = {Op.ADD, Op.SUB, Op.SLL, Op.SLT, Op.SLTU, Op.XOR, Op.SRL, Op.SRA, Op.OR, Op.AND}
S_TYPE = {Op.SB, Op.SH, Op.SW}
J_TYPE = {Op.JAL}

# Who writes rd?  R-type + JAL -> 11 writers
WRITERS = sorted(R_TYPE | J_TYPE)
# Who reads source registers?  R-type reads rs1+rs2, S-type reads rs1 (addr) + rs2 (data) -> 13 readers
READERS = sorted(R_TYPE | S_TYPE)


def _build_bin_index():
    bins: list[str] = []
    # SEEN for every op
    for op in Op:
        bins.append(f"{op.name.lower()}_seen")
    # ZERO_DST for writers (11)
    for op in WRITERS:
        bins.append(f"{op.name.lower()}_zero_dst")
    # ZERO_SRC for readers (13)
    for op in READERS:
        bins.append(f"{op.name.lower()}_zero_src")
    # SAME_SRC for readers (13)
    for op in READERS:
        bins.append(f"{op.name.lower()}_same_src")
    # BR for JAL (2)
    bins.append("jal_br_forwards")
    bins.append("jal_br_backwards")
    # RAW_HAZARD crosses: 11 writers * 13 readers = 143
    for w in WRITERS:
        for r in READERS:
            bins.append(f"{w.name.lower()}->{r.name.lower()}_raw_hazard")
    return bins


BIN_NAMES = _build_bin_index()
N_BINS = len(BIN_NAMES)
BIN_IDX = {n: i for i, n in enumerate(BIN_NAMES)}
assert N_BINS == 196, f"Expected 196 bins, got {N_BINS}"

# Precomputed indices for fast lookup
_SEEN_IDX = {op: BIN_IDX[f"{op.name.lower()}_seen"] for op in Op}
_ZERO_DST_IDX = {op: BIN_IDX[f"{op.name.lower()}_zero_dst"] for op in WRITERS}
_ZERO_SRC_IDX = {op: BIN_IDX[f"{op.name.lower()}_zero_src"] for op in READERS}
_SAME_SRC_IDX = {op: BIN_IDX[f"{op.name.lower()}_same_src"] for op in READERS}
_RAW_IDX = {(w, r): BIN_IDX[f"{w.name.lower()}->{r.name.lower()}_raw_hazard"]
            for w in WRITERS for r in READERS}
_JAL_FWD = BIN_IDX["jal_br_forwards"]
_JAL_BACK = BIN_IDX["jal_br_backwards"]


def bins_for_step(op: int, rd: int, rs1: int, rs2: int, imm_sign: int,
                  prev_writer: int | None, prev_rd: int | None) -> list[int]:
    """Return bin indices this (op, rd, rs1, rs2, imm_sign) fires, given prior state.

    - imm_sign: +1 / -1 / 0 (only used for JAL to split forwards/backwards).
    - prev_writer: Op enum of the previous instruction if it wrote rd, else None.
    - prev_rd: rd the previous instruction wrote (if any).

    The environment is responsible for updating (prev_writer, prev_rd) after calling this.
    """
    op = Op(op)
    hits = [_SEEN_IDX[op]]

    if op in WRITERS and rd == 0:
        hits.append(_ZERO_DST_IDX[op])
    if op in READERS and (rs1 == 0 or rs2 == 0):
        hits.append(_ZERO_SRC_IDX[op])
    if op in READERS and rs1 == rs2:
        hits.append(_SAME_SRC_IDX[op])
    if op == Op.JAL:
        if imm_sign > 0:
            hits.append(_JAL_FWD)
        elif imm_sign < 0:
            hits.append(_JAL_BACK)

    # RAW hazard: the encoding bits of the current instr's rs1/rs2 match the
    # encoding bits of the previous instr's rd. The LLM4DV monitor fires this
    # even when rd==0 (a write to x0 is architecturally a no-op but the RVFI
    # still reports the same bit pattern). We mirror that literal behavior.
    if op in READERS and prev_writer is not None and prev_rd is not None:
        if rs1 == prev_rd or rs2 == prev_rd:
            hits.append(_RAW_IDX[(prev_writer, op)])

    return hits


def max_reachable_bins() -> int:
    """All 196 bins are reachable by construction -- we're the ground truth."""
    return N_BINS


if __name__ == "__main__":
    print(f"Total bins: {N_BINS}")
    print(f"Writers (11): {[op.name for op in WRITERS]}")
    print(f"Readers (13): {[op.name for op in READERS]}")
    print(f"RAW hazard crosses: {len(WRITERS) * len(READERS)}")
