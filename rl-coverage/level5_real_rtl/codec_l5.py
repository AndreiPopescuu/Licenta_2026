"""Extended codec for Level 5 — adds the op classes whose absence capped our
toggle ceiling at 50%:

  + MUL/DIV (8 ops, R-type funct7=0x01) — opens ibex_multdiv_fast paths
  + Branches (6 ops, B-type)             — opens controller / if_stage taken-branch paths
  + JAL                                  — opens jump-target / link-write paths
  + CSR with rotating CSR address        — opens cs_registers diversity

Control-flow safety: branches and JAL use *small forward* targets only, so
execution can't infinite-loop or run off the program. Every program is also
padded so that even maximal forward jumping lands inside the WFI sentinel
region.
"""

from enum import IntEnum


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
    # M extension — MUL/DIV (8 new)
    MUL = 30; MULH = 31; MULHSU = 32; MULHU = 33
    DIV = 34; DIVU = 35; REM = 36; REMU = 37
    # Branches (6 new) — forward-safe targets only
    BEQ = 38; BNE = 39; BLT = 40; BGE = 41; BLTU = 42; BGEU = 43
    # JAL (1 new) — forward-safe targets only
    JAL = 44


N_OPS = 45  # 30 base + 8 MUL/DIV + 6 branches + 1 JAL

R_TYPE_ALU = {Op.ADD, Op.SUB, Op.SLL, Op.SLT, Op.SLTU, Op.XOR, Op.SRL, Op.SRA, Op.OR, Op.AND}
I_ALU      = {Op.ADDI, Op.SLTI, Op.SLTIU, Op.XORI, Op.ORI, Op.ANDI, Op.SLLI, Op.SRLI, Op.SRAI}
LOADS      = {Op.LB, Op.LH, Op.LW, Op.LBU, Op.LHU}
STORES     = {Op.SB, Op.SH, Op.SW}
CSR_OPS    = {Op.CSRRW, Op.CSRRS, Op.CSRRC}
MUL_DIV    = {Op.MUL, Op.MULH, Op.MULHSU, Op.MULHU, Op.DIV, Op.DIVU, Op.REM, Op.REMU}
BRANCHES   = {Op.BEQ, Op.BNE, Op.BLT, Op.BGE, Op.BLTU, Op.BGEU}
JAL_OPS    = {Op.JAL}

R_TYPE_ALL = R_TYPE_ALU | MUL_DIV   # all R-type encodings

# Bucket interpretation:
#   ALU/Load/Store imm:  signed 12-bit value index
#   Shift shamt:         5-bit value index
#   Branch target:       all forward, all small (no infinite loops)
#   JAL target:          all forward, all small
#   CSR address:         rotates through a safe CSR list
IMM_BUCKETS = 5

IMM_BUCKET_VALUES   = [-2048, -100, 0, 100, 2047]
SHAMT_BUCKET_VALUES = [0, 8, 16, 24, 31]
BRANCH_BUCKET_OFFSETS = [4, 8, 12, 16, 20]   # all forward, all multiples of 4
JAL_BUCKET_OFFSETS    = [4, 8, 12, 16, 24]   # all forward, all multiples of 4

# Safe CSRs that can be read/written without altering control flow or interrupts.
# mscratch is the safest (just a scratch register). The others are read-mostly
# but won't change CPU behaviour given we mostly write back what we read.
SAFE_CSRS = [
    0x340,  # mscratch  (R/W scratchpad)
    0xF14,  # mhartid   (read-only; CSR write becomes a no-op for the value)
    0xF11,  # mvendorid (read-only)
    0xF12,  # marchid   (read-only)
    0xF13,  # mimpid    (read-only)
]


# ---- R-type encoder helpers ----

R_F3F7 = {
    Op.ADD:  (0b000, 0b0000000), Op.SUB:  (0b000, 0b0100000),
    Op.SLL:  (0b001, 0b0000000), Op.SLT:  (0b010, 0b0000000),
    Op.SLTU: (0b011, 0b0000000), Op.XOR:  (0b100, 0b0000000),
    Op.SRL:  (0b101, 0b0000000), Op.SRA:  (0b101, 0b0100000),
    Op.OR:   (0b110, 0b0000000), Op.AND:  (0b111, 0b0000000),
    # M extension uses funct7 = 0b0000001
    Op.MUL:    (0b000, 0b0000001), Op.MULH:   (0b001, 0b0000001),
    Op.MULHSU: (0b010, 0b0000001), Op.MULHU:  (0b011, 0b0000001),
    Op.DIV:    (0b100, 0b0000001), Op.DIVU:   (0b101, 0b0000001),
    Op.REM:    (0b110, 0b0000001), Op.REMU:   (0b111, 0b0000001),
}
I_F3 = {
    Op.ADDI: 0b000, Op.SLTI: 0b010, Op.SLTIU: 0b011,
    Op.XORI: 0b100, Op.ORI: 0b110, Op.ANDI: 0b111,
}
SHIFT_F3F7 = {
    Op.SLLI: (0b001, 0b0000000),
    Op.SRLI: (0b101, 0b0000000),
    Op.SRAI: (0b101, 0b0100000),
}
LOAD_F3 = {Op.LB: 0b000, Op.LH: 0b001, Op.LW: 0b010, Op.LBU: 0b100, Op.LHU: 0b101}
STORE_F3 = {Op.SB: 0b000, Op.SH: 0b001, Op.SW: 0b010}
CSR_F3 = {Op.CSRRW: 0b001, Op.CSRRS: 0b010, Op.CSRRC: 0b011}
BRANCH_F3 = {
    Op.BEQ: 0b000, Op.BNE: 0b001,
    Op.BLT: 0b100, Op.BGE: 0b101,
    Op.BLTU: 0b110, Op.BGEU: 0b111,
}


def _imm12(imm: int) -> int:
    return imm & 0xFFF


def _b_imm_encode(offset: int) -> tuple[int, int]:
    """Branch immediate (B-type) splits across rd / funct7 fields.

    Returns (encoded_imm_hi_bits[31:25], encoded_imm_lo_bits[11:7]).
    """
    o = offset & 0x1FFE  # 13-bit, bit 0 always 0
    imm12   = (o >> 12) & 0x1
    imm10_5 = (o >> 5)  & 0x3F
    imm4_1  = (o >> 1)  & 0xF
    imm11   = (o >> 11) & 0x1
    hi = (imm12 << 6) | imm10_5         # bits [31:25]
    lo = (imm4_1 << 1) | imm11          # bits [11:7]
    return hi, lo


def _j_imm_encode(offset: int) -> int:
    """JAL immediate (J-type) — 20 bits scattered across [31:12]."""
    o = offset & 0x1FFFFE
    imm20    = (o >> 20) & 0x1
    imm10_1  = (o >> 1)  & 0x3FF
    imm11    = (o >> 11) & 0x1
    imm19_12 = (o >> 12) & 0xFF
    return (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12)


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    op = Op(op_i)
    rd &= 0x1F; rs1 &= 0x1F; rs2 &= 0x1F

    if op in R_TYPE_ALU or op in MUL_DIV:
        f3, f7 = R_F3F7[op]
        return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0b0110011
    if op in I_ALU:
        if op in SHIFT_F3F7:
            f3, f7 = SHIFT_F3F7[op]
            shamt = SHAMT_BUCKET_VALUES[imm_bucket] & 0x1F
            return (f7 << 25) | (shamt << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0b0010011
        return (_imm12(IMM_BUCKET_VALUES[imm_bucket]) << 20) | (rs1 << 15) | (I_F3[op] << 12) | (rd << 7) | 0b0010011
    if op in LOADS:
        return (_imm12(IMM_BUCKET_VALUES[imm_bucket]) << 20) | (rs1 << 15) | (LOAD_F3[op] << 12) | (rd << 7) | 0b0000011
    if op in STORES:
        imm = _imm12(IMM_BUCKET_VALUES[imm_bucket])
        return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (STORE_F3[op] << 12) | ((imm & 0x1F) << 7) | 0b0100011
    if op in CSR_OPS:
        csr = SAFE_CSRS[imm_bucket % len(SAFE_CSRS)]
        return (csr << 20) | (rs1 << 15) | (CSR_F3[op] << 12) | (rd << 7) | 0b1110011
    if op in BRANCHES:
        offset = BRANCH_BUCKET_OFFSETS[imm_bucket]
        hi, lo = _b_imm_encode(offset)
        return (hi << 25) | (rs2 << 20) | (rs1 << 15) | (BRANCH_F3[op] << 12) | (lo << 7) | 0b1100011
    if op in JAL_OPS:
        offset = JAL_BUCKET_OFFSETS[imm_bucket]
        return _j_imm_encode(offset) | (rd << 7) | 0b1101111
    raise ValueError(f"Unsupported op {op}")


def emit_program(actions: list[tuple[int, int, int, int, int]]) -> list[int]:
    """Encode a list of (op, rd, rs1, rs2, imm_bucket) tuples into machine code.

    Pads the tail with 16 NOPs (ADDI x0, x0, 0) so that any forward branch /
    JAL near the end can land on a valid instruction rather than running off
    into uninitialized memory.
    """
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)  # ADDI x0, x0, 0  (imm bucket 2 = 0)
    return [encode(*a) for a in actions] + [nop] * 16


if __name__ == "__main__":
    # Self-test: encode every op once with random fields, ensure no crashes.
    import random
    rng = random.Random(0)
    for op_i in range(N_OPS):
        word = encode(op_i, rng.randrange(32), rng.randrange(32),
                      rng.randrange(32), rng.randrange(IMM_BUCKETS))
        print(f"  op={Op(op_i).name:>8s}  ->  0x{word:08x}")
    print(f"\nN_OPS = {N_OPS}, IMM_BUCKETS = {IMM_BUCKETS}")
