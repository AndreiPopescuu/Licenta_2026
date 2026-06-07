"""Level 8 codec — L7 (64 ops) + LUI + JALR + CSRRWI/SI/CI + FENCE = 70 ops.

Each new op exercises toggle paths unreachable with the L7 action space:

  LUI    — U-type, different opcode from AUIPC.  Enables loading arbitrary
            32-bit constants into registers, which diversifies MUL/DIV operands
            and helps toggle imd_val_q_i[0][*] bits in ibex_multdiv_fast.
  JALR   — indirect branch; exercises controller FSM indirect-jump path and
            the PC-target mux in ibex_if_stage.
  CSRRWI/CSRRSI/CSRRCI — CSR immediate variants (funct3=101/110/111); use
            the 5-bit uimm field instead of rs1, exercising a different data
            path in ibex_cs_registers from the register-based CSRRW/S/C.
  FENCE  — MISC-MEM opcode; exercises the ibex_decoder FENCE branch and the
            associated controller flush path.

Total: 70 ops.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L7 = (Path(__file__).resolve().parent.parent / "level7_stimulus")
sys.path.insert(0, str(_L7))

from codec_l7 import (  # noqa: E402
    Op as L7Op,
    N_OPS as L7_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    SHAMT_BUCKET_VALUES,
    BRANCH_BUCKET_OFFSETS,
    JAL_BUCKET_OFFSETS,
    L7_SAFE_CSRS,
    encode as l7_encode,
)


class Op(IntEnum):
    # ── L7 ops unchanged (0-63) ───────────────────────────────────────────
    ADD = 0;   SUB = 1;   SLL = 2;   SLT = 3;   SLTU = 4
    XOR = 5;   SRL = 6;   SRA = 7;   OR = 8;    AND = 9
    ADDI = 10; SLTI = 11; SLTIU = 12; XORI = 13; ORI = 14; ANDI = 15
    SLLI = 16; SRLI = 17; SRAI = 18
    LB = 19;   LH = 20;   LW = 21;   LBU = 22;  LHU = 23
    SB = 24;   SH = 25;   SW = 26
    CSRRW = 27; CSRRS = 28; CSRRC = 29
    MUL = 30; MULH = 31; MULHSU = 32; MULHU = 33
    DIV = 34; DIVU = 35; REM = 36; REMU = 37
    BEQ = 38; BNE = 39; BLT = 40; BGE = 41; BLTU = 42; BGEU = 43
    JAL = 44
    C_ADDI = 45; C_LI = 46; C_LUI = 47; C_SLLI = 48
    C_MV = 49;   C_ADD = 50
    C_ADDI4SPN = 51; C_LW = 52; C_SW = 53
    C_AND = 54; C_OR = 55; C_XOR = 56; C_SUB = 57
    C_SRLI = 58; C_SRAI = 59; C_ANDI = 60
    AUIPC  = 61
    ECALL  = 62
    EBREAK = 63
    # ── Level 8 additions ────────────────────────────────────────────────
    LUI    = 64
    JALR   = 65
    CSRRWI = 66
    CSRRSI = 67
    CSRRCI = 68
    FENCE  = 69


N_OPS = 70

# LUI: same 20-bit upper-imm buckets as AUIPC — chosen to hit diverse high bits.
LUI_IMM_BUCKETS = [0x00001, 0x12345, 0xABCDE, 0xFFFFF, 0x80000]

# JALR: small byte offsets so a "JALR x0, 0(x1)" acts as a simple return.
# The L8 cocotb driver initialises x1..x31 to known values, so small offsets
# keep PC within the program region most of the time.
JALR_OFFSET_BUCKETS = [0, 4, 8, -4, 0]

# CSR immediate uimm[4:0] buckets (unsigned 5-bit).
CSRIMM_BUCKETS = [0, 1, 3, 7, 15]


def _encode_lui(rd: int, imm_bucket: int) -> int:
    imm20 = LUI_IMM_BUCKETS[imm_bucket % len(LUI_IMM_BUCKETS)] & 0xFFFFF
    return (imm20 << 12) | ((rd & 0x1F) << 7) | 0b0110111


def _encode_jalr(rd: int, rs1: int, imm_bucket: int) -> int:
    offset = JALR_OFFSET_BUCKETS[imm_bucket % len(JALR_OFFSET_BUCKETS)] & 0xFFF
    return (offset << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b1100111


def _encode_csrimm(funct3: int, rd: int, rs1: int, imm_bucket: int) -> int:
    uimm = CSRIMM_BUCKETS[imm_bucket % len(CSRIMM_BUCKETS)] & 0x1F
    csr_idx = (imm_bucket * 7 + (rs1 % 5)) % len(L7_SAFE_CSRS)
    csr = L7_SAFE_CSRS[csr_idx]
    return (csr << 20) | ((uimm & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_fence() -> int:
    return 0x0FF0000F  # FENCE iorw, iorw


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    if op_i == int(Op.LUI):
        return _encode_lui(rd, imm_bucket)
    if op_i == int(Op.JALR):
        return _encode_jalr(rd, rs1, imm_bucket)
    if op_i == int(Op.CSRRWI):
        return _encode_csrimm(0b101, rd, rs1, imm_bucket)
    if op_i == int(Op.CSRRSI):
        return _encode_csrimm(0b110, rd, rs1, imm_bucket)
    if op_i == int(Op.CSRRCI):
        return _encode_csrimm(0b111, rd, rs1, imm_bucket)
    if op_i == int(Op.FENCE):
        return _encode_fence()
    return l7_encode(op_i, rd, rs1, rs2, imm_bucket)


def emit_program(actions):
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} → 0x{w:08x}"
    assert (encode(int(Op.LUI),  3, 0, 0, 0) & 0x7F) == 0b0110111
    assert (encode(int(Op.JALR), 1, 2, 0, 0) & 0x7F) == 0b1100111
    assert encode(int(Op.FENCE), 0, 0, 0, 0) == 0x0FF0000F
    print(f"[OK] L8 self-test: {N_OPS} ops × {IMM_BUCKETS} buckets")
    print(f"  LUI    sample: 0x{encode(int(Op.LUI),   3, 0, 0, 1):08x}")
    print(f"  JALR   sample: 0x{encode(int(Op.JALR),  1, 2, 0, 0):08x}")
    print(f"  CSRRWI sample: 0x{encode(int(Op.CSRRWI),1, 1, 0, 2):08x}")
    print(f"  FENCE: 0x{encode(int(Op.FENCE), 0, 0, 0, 0):08x}")


if __name__ == "__main__":
    _self_test()
