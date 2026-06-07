"""
Pure-Python model of the Ibex decoder + LLM4DV coverage database (2107 bins).

Skips Verilator/cocotb entirely. Given (op_name, rd, rs1, rs2) it records the
exact same bins the real hardware+monitor would record. Validated against
LLM4DV's ID coverage plan (same bin names, same sizes).

This lets us iterate millions of steps per minute for RL training, then we
can validate a final policy against the real cocotb decoder if we want to.
"""

import numpy as np

ALU_OP_NAMES = ["add", "sub", "or", "xor", "and", "sll", "srl", "sra", "slt", "sltu"]
MEM_SIZE_NAMES = ["word", "half-word", "byte"]

# The 26 "ops" that actually produce coverage (R-type ALU, I-type ALU, loads, stores).
# Plus a no-op / illegal slot that hits nothing.
OP_TYPES = (
    [("alu", op) for op in ALU_OP_NAMES]            # 10 R-type ALU
    + [("alu_imm", op) for op in ALU_OP_NAMES]      # 10 I-type ALU (SUBI unreachable but we keep it)
    + [("load", sz) for sz in MEM_SIZE_NAMES]       # 3 loads
    + [("store", sz) for sz in MEM_SIZE_NAMES]      # 3 stores
)
N_OP_TYPES = len(OP_TYPES)  # 26


def build_bin_index():
    """Return an ordered list of bin names, matching LLM4DV's get_coverage_plan_ID()."""
    bins = []
    # Type 1 -- "SEEN" bins
    for op in ALU_OP_NAMES:                 bins.append(f"ALU_{op.upper()}")
    for op in ALU_OP_NAMES:                 bins.append(f"ALUI_{op.upper()}I")
    bins.append("illegal_instruction")
    for sz in MEM_SIZE_NAMES:               bins.append(f"L{sz[0].upper()}")
    for sz in MEM_SIZE_NAMES:               bins.append(f"S{sz[0].upper()}")
    # Type 2 -- register ports
    for i in range(32):                     bins.append(f"read_A_reg_{i}")
    for i in range(32):                     bins.append(f"read_B_reg_{i}")
    for i in range(32):                     bins.append(f"write_reg_{i}")
    # Type 3 -- crosses
    for op in ALU_OP_NAMES:
        for i in range(32):                 bins.append(f"{op.upper()}_x_read_A_reg_{i}")
    for op in ALU_OP_NAMES:
        for i in range(32):                 bins.append(f"{op.upper()}_x_read_B_reg_{i}")
    for op in ALU_OP_NAMES:
        for i in range(32):                 bins.append(f"{op.upper()}_x_write_reg_{i}")
    for op in ALU_OP_NAMES:
        for i in range(32):                 bins.append(f"{op.upper()}I_x_read_A_reg_{i}")
    for op in ALU_OP_NAMES:
        for i in range(32):                 bins.append(f"{op.upper()}I_x_write_reg_{i}")
    for sz in MEM_SIZE_NAMES:
        for i in range(32):                 bins.append(f"L{sz[0].upper()}_x_read_A_reg_{i}")
    for sz in MEM_SIZE_NAMES:
        for i in range(32):                 bins.append(f"L{sz[0].upper()}_x_write_reg_{i}")
    for sz in MEM_SIZE_NAMES:
        for i in range(32):                 bins.append(f"S{sz[0].upper()}_x_read_A_reg_{i}")
    for sz in MEM_SIZE_NAMES:
        for i in range(32):                 bins.append(f"S{sz[0].upper()}_x_read_B_reg_{i}")
    return bins


BIN_NAMES = build_bin_index()
N_BINS = len(BIN_NAMES)
BIN_IDX = {name: i for i, name in enumerate(BIN_NAMES)}
assert N_BINS == 2107, f"Expected 2107 bins, got {N_BINS}"


def bins_for_action(op_idx: int, rd: int, rs1: int, rs2: int) -> list[int]:
    """Given a structured action (op, rd, rs1, rs2), return the bin indices that fire.

    Mirrors CoverageDatabase.update() + the monitor logic in ibex_decoder_cocotb.py.
    Note: real hardware loads only write rd (no rs2 read) and stores read rs1+rs2
    (no rd write). We encode those rules faithfully.
    """
    kind, name = OP_TYPES[op_idx]
    hits: list[int] = []

    if kind == "alu":
        hits.append(BIN_IDX[f"ALU_{name.upper()}"])
        hits.append(BIN_IDX[f"read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"read_B_reg_{rs2}"])
        hits.append(BIN_IDX[f"write_reg_{rd}"])
        hits.append(BIN_IDX[f"{name.upper()}_x_read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"{name.upper()}_x_read_B_reg_{rs2}"])
        hits.append(BIN_IDX[f"{name.upper()}_x_write_reg_{rd}"])
    elif kind == "alu_imm":
        # RISC-V has no SUBI; real decoder emits illegal_instruction. Mirror that.
        if name == "sub":
            hits.append(BIN_IDX["illegal_instruction"])
            return hits
        hits.append(BIN_IDX[f"ALUI_{name.upper()}I"])
        hits.append(BIN_IDX[f"read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"write_reg_{rd}"])
        hits.append(BIN_IDX[f"{name.upper()}I_x_read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"{name.upper()}I_x_write_reg_{rd}"])
    elif kind == "load":
        hits.append(BIN_IDX[f"L{name[0].upper()}"])
        hits.append(BIN_IDX[f"read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"write_reg_{rd}"])
        hits.append(BIN_IDX[f"L{name[0].upper()}_x_read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"L{name[0].upper()}_x_write_reg_{rd}"])
    elif kind == "store":
        hits.append(BIN_IDX[f"S{name[0].upper()}"])
        hits.append(BIN_IDX[f"read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"read_B_reg_{rs2}"])
        hits.append(BIN_IDX[f"S{name[0].upper()}_x_read_A_reg_{rs1}"])
        hits.append(BIN_IDX[f"S{name[0].upper()}_x_read_B_reg_{rs2}"])
    return hits


def max_reachable_bins() -> int:
    """Systematic sweep -- the ISA-achievable ceiling."""
    covered = np.zeros(N_BINS, dtype=bool)
    for op in range(N_OP_TYPES):
        for rd in range(32):
            for rs1 in range(32):
                for rs2 in range(32):
                    for b in bins_for_action(op, rd, rs1, rs2):
                        covered[b] = True
    return int(covered.sum())


if __name__ == "__main__":
    ceiling = max_reachable_bins()
    print(f"Total bins: {N_BINS}")
    print(f"ISA-reachable ceiling: {ceiling}/{N_BINS} = {100*ceiling/N_BINS:.2f}%")
