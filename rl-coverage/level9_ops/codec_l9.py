"""Level 9 codec — L8 (70 ops) + 13 new ops = 83 ops.

Completează ISA-ul RV32IMC al lui Ibex. Fiecare op nouă targetează
un path specific neacoperit de cele 70 ops din L8:

  MRET        — exception return: ibex_controller FSM path mepc→PC,
                mstatus.mie restore. Fără MRET jumătate din FSM-ul de
                excepții rămâne neacoperit indiferent de câte ECALL.
  WFI         — wait-for-interrupt: ibex_controller WFI state +
                ibex_cs_registers WFI path.
  FENCE_I     — instruction fence (0x0000100F): ibex_if_stage flush
                diferit de FENCE (data fence 0x0FF0000F).
  C_JAL       — compressed JAL (RV32 only, quad1 funct3=001):
                ibex_compressed_decoder path neacoperit.
  C_J         — compressed jump (quad1 funct3=101).
  C_BEQZ      — compressed branch ==0 (quad1 funct3=110).
  C_BNEZ      — compressed branch !=0 (quad1 funct3=111).
  C_ADDI16SP  — c.addi16sp (quad1 funct3=011, rd=2): caz special din
                C_LUI, path separat în ibex_compressed_decoder.
  C_LWSP      — load word stack-relative (quad2 funct3=010).
  C_SWSP      — store word stack-relative (quad2 funct3=110).
  C_JR        — compressed jump register (quad2 funct3=100, bit12=0).
  C_JALR      — compressed JAL register (quad2 funct3=100, bit12=1).
  C_EBREAK    — compressed breakpoint (0x9002).

Instrucțiunile compressed de 16 biți sunt împachetate ca:
  (C_NOP << 16) | rvc_16bit
astfel că fiecare word de 32 biți conține C_NOP (upper) + instrucțiunea
(lower), în ordinea de execuție a CPU-ului (little-endian).

Total: 83 ops.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L8 = Path(__file__).resolve().parent.parent / "level8_ops"
_L7 = Path(__file__).resolve().parent.parent / "level7_stimulus"
sys.path.insert(0, str(_L8))
sys.path.insert(0, str(_L7))

from codec_l8 import (  # noqa: E402
    Op as L8Op,
    N_OPS as L8_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    encode as l8_encode,
)
from codec_l7 import L7_SAFE_CSRS  # noqa: E402

# Extended CSR list: L7 safe set + interrupt/trap control registers
L9_CSRS = L7_SAFE_CSRS + [
    0x300,  # mstatus  — MIE/MPIE/MPP fields
    0x304,  # mie      — machine interrupt enable
    0x305,  # mtvec    — trap vector base
]

_CSR_F3 = {27: 0b001, 28: 0b010, 29: 0b011}   # CSRRW, CSRRS, CSRRC
_CSRI_F3 = {66: 0b101, 67: 0b110, 68: 0b111}  # CSRRWI, CSRRSI, CSRRCI


def _encode_csr_l9(funct3: int, rd: int, rs1: int, imm_bucket: int) -> int:
    csr_idx = (imm_bucket * 7 + rs1) % len(L9_CSRS)
    csr = L9_CSRS[csr_idx]
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_csri_l9(funct3: int, rd: int, rs1: int, imm_bucket: int) -> int:
    uimm = (imm_bucket * 3 + (rs1 % 5)) & 0x1F
    csr_idx = (imm_bucket * 7 + (rs1 % 5)) % len(L9_CSRS)
    csr = L9_CSRS[csr_idx]
    return (csr << 20) | ((uimm & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011

_C_NOP = 0x0001  # c.nop = c.addi x0, 0


class Op(IntEnum):
    # L8 ops unchanged (0-69)
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
    LUI    = 64
    JALR   = 65
    CSRRWI = 66
    CSRRSI = 67
    CSRRCI = 68
    FENCE  = 69
    # L9 additions
    MRET       = 70
    WFI        = 71
    FENCE_I    = 72
    C_JAL      = 73
    C_J        = 74
    C_BEQZ     = 75
    C_BNEZ     = 76
    C_ADDI16SP = 77
    C_LWSP     = 78
    C_SWSP     = 79
    C_JR       = 80
    C_JALR     = 81
    C_EBREAK   = 82


N_OPS = 83

_MRET    = 0x30200073
_WFI     = 0x10500073
_FENCE_I = 0x0000100F
_C_EBREAK_16 = 0x9002

# Buckets pentru instrucțiunile noi
_C_JUMP_OFFSETS    = [4, 8, 16, 32, 64]   # C_J / C_JAL: salturi forward
_C_BRANCH_OFFSETS  = [4, 8, 16, 32, -4]   # C_BEQZ / C_BNEZ
_C_ADDI16SP_OFFS   = [16, 32, -16, 48, -32]  # multiple de 16, non-zero
_C_LSTK_OFFSETS    = [0, 4, 8, 12, 16]    # C_LWSP / C_SWSP: word-aligned


def _pack_rvc(rvc16: int) -> int:
    """Împachetează o instrucțiune compressed de 16b ca word de 32b."""
    return (_C_NOP << 16) | (rvc16 & 0xFFFF)


# ── C.J / C.JAL immediate encoding ──────────────────────────────────────────

def _c_jump_imm_bits(off: int) -> int:
    """Calculează câmpul immediate de 11 biți pentru C.J/C.JAL (inst[12:2])."""
    off &= ~1  # trebuie să fie par
    b11 = (off >> 11) & 1
    b4  = (off >> 4)  & 1
    b9  = (off >> 9)  & 1
    b8  = (off >> 8)  & 1
    b10 = (off >> 10) & 1
    b6  = (off >> 6)  & 1
    b7  = (off >> 7)  & 1
    b3  = (off >> 3)  & 1
    b2  = (off >> 2)  & 1
    b1  = (off >> 1)  & 1
    b5  = (off >> 5)  & 1
    return (b11 << 12) | (b4 << 11) | (b9 << 10) | (b8 << 9) | (b10 << 8) \
         | (b6 << 7)  | (b7 << 6)  | (b3 << 5)  | (b2 << 4) | (b1 << 3)  \
         | (b5 << 2)


def _encode_c_j(imm_bucket: int) -> int:
    off = _C_JUMP_OFFSETS[imm_bucket % len(_C_JUMP_OFFSETS)]
    return (0b101 << 13) | _c_jump_imm_bits(off) | 0b01


def _encode_c_jal(imm_bucket: int) -> int:
    off = _C_JUMP_OFFSETS[imm_bucket % len(_C_JUMP_OFFSETS)]
    return (0b001 << 13) | _c_jump_imm_bits(off) | 0b01


# ── C.BEQZ / C.BNEZ ──────────────────────────────────────────────────────────

def _encode_c_branch(funct3: int, rs1: int, imm_bucket: int) -> int:
    rs1p = rs1 & 7  # 3-bit: selectează x8..x15
    off  = _C_BRANCH_OFFSETS[imm_bucket % len(_C_BRANCH_OFFSETS)]
    off &= ~1
    b8  = (off >> 8) & 1
    b4  = (off >> 4) & 1
    b3  = (off >> 3) & 1
    b7  = (off >> 7) & 1
    b6  = (off >> 6) & 1
    b2  = (off >> 2) & 1
    b1  = (off >> 1) & 1
    b5  = (off >> 5) & 1
    return (funct3 << 13) | (b8 << 12) | (b4 << 11) | (b3 << 10) \
         | (rs1p << 7) | (b7 << 6) | (b6 << 5) | (b2 << 4) \
         | (b1 << 3) | (b5 << 2) | 0b01


# ── C.ADDI16SP ───────────────────────────────────────────────────────────────

def _encode_c_addi16sp(imm_bucket: int) -> int:
    off   = _C_ADDI16SP_OFFS[imm_bucket % len(_C_ADDI16SP_OFFS)]
    nzimm = off & 0x3FF  # 10-bit two's complement
    b9   = (nzimm >> 9) & 1
    b4   = (nzimm >> 4) & 1
    b6   = (nzimm >> 6) & 1
    b8_7 = (nzimm >> 7) & 3
    b5   = (nzimm >> 5) & 1
    return (0b011 << 13) | (b9 << 12) | (0b00010 << 7) \
         | (b4 << 6) | (b6 << 5) | (b8_7 << 3) | (b5 << 2) | 0b01


# ── C.LWSP / C.SWSP ──────────────────────────────────────────────────────────

def _encode_c_lwsp(rd: int, imm_bucket: int) -> int:
    rd  = max(1, rd & 0x1F)  # rd != 0
    off = _C_LSTK_OFFSETS[imm_bucket % len(_C_LSTK_OFFSETS)]
    b5  = (off >> 5) & 1
    b4  = (off >> 4) & 1
    b3  = (off >> 3) & 1
    b2  = (off >> 2) & 1
    b7  = (off >> 7) & 1
    b6  = (off >> 6) & 1
    return (0b010 << 13) | (b5 << 12) | (rd << 7) \
         | (b4 << 6) | (b3 << 5) | (b2 << 4) | (b7 << 3) | (b6 << 2) | 0b10


def _encode_c_swsp(rs2: int, imm_bucket: int) -> int:
    off = _C_LSTK_OFFSETS[imm_bucket % len(_C_LSTK_OFFSETS)]
    b5  = (off >> 5) & 1
    b4  = (off >> 4) & 1
    b3  = (off >> 3) & 1
    b2  = (off >> 2) & 1
    b7  = (off >> 7) & 1
    b6  = (off >> 6) & 1
    return (0b110 << 13) | (b5 << 12) | (b4 << 11) | (b3 << 10) \
         | (b2 << 9) | (b7 << 8) | (b6 << 7) | ((rs2 & 0x1F) << 2) | 0b10


# ── C.JR / C.JALR ────────────────────────────────────────────────────────────

def _encode_c_jr(rs1: int) -> int:
    rs1 = max(1, rs1 & 0x1F)  # rs1 != 0
    return (0b1000 << 12) | (rs1 << 7) | 0b10


def _encode_c_jalr(rs1: int) -> int:
    rs1 = max(1, rs1 & 0x1F)  # rs1 != 0
    return (0b1001 << 12) | (rs1 << 7) | 0b10


# ── Encoder principal ─────────────────────────────────────────────────────────

def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    # CSR register ops — use extended L9_CSRS list
    if op_i in _CSR_F3:
        return _encode_csr_l9(_CSR_F3[op_i], rd, rs1, imm_bucket)
    # CSR immediate ops — use extended L9_CSRS list
    if op_i in _CSRI_F3:
        return _encode_csri_l9(_CSRI_F3[op_i], rd, rs1, imm_bucket)
    if op_i == int(Op.MRET):
        return _MRET
    if op_i == int(Op.WFI):
        return _WFI
    if op_i == int(Op.FENCE_I):
        return _FENCE_I
    if op_i == int(Op.C_JAL):
        return _pack_rvc(_encode_c_jal(imm_bucket))
    if op_i == int(Op.C_J):
        return _pack_rvc(_encode_c_j(imm_bucket))
    if op_i == int(Op.C_BEQZ):
        return _pack_rvc(_encode_c_branch(0b110, rs1, imm_bucket))
    if op_i == int(Op.C_BNEZ):
        return _pack_rvc(_encode_c_branch(0b111, rs1, imm_bucket))
    if op_i == int(Op.C_ADDI16SP):
        return _pack_rvc(_encode_c_addi16sp(imm_bucket))
    if op_i == int(Op.C_LWSP):
        return _pack_rvc(_encode_c_lwsp(rd, imm_bucket))
    if op_i == int(Op.C_SWSP):
        return _pack_rvc(_encode_c_swsp(rs2, imm_bucket))
    if op_i == int(Op.C_JR):
        return _pack_rvc(_encode_c_jr(rs1))
    if op_i == int(Op.C_JALR):
        return _pack_rvc(_encode_c_jalr(rs1))
    if op_i == int(Op.C_EBREAK):
        return _pack_rvc(_C_EBREAK_16)
    return l8_encode(op_i, rd, rs1, rs2, imm_bucket)


def emit_program(actions):
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} → 0x{w:08x}"

    assert encode(int(Op.MRET),    0, 0, 0, 0) == _MRET
    assert encode(int(Op.WFI),     0, 0, 0, 0) == _WFI
    assert encode(int(Op.FENCE_I), 0, 0, 0, 0) == _FENCE_I
    assert encode(int(Op.C_EBREAK), 0, 0, 0, 0) == (_C_NOP << 16) | _C_EBREAK_16

    # C_J cu offset=4: lower 16 biți trebuie să aibă quadrant=01 și funct3=101
    c_j_w = encode(int(Op.C_J), 0, 0, 0, 0) & 0xFFFF
    assert (c_j_w & 0b11) == 0b01 and (c_j_w >> 13) == 0b101, \
        f"C_J encoding greșit: 0x{c_j_w:04x}"

    # C_JAL: quadrant=01, funct3=001
    c_jal_w = encode(int(Op.C_JAL), 0, 0, 0, 0) & 0xFFFF
    assert (c_jal_w & 0b11) == 0b01 and (c_jal_w >> 13) == 0b001, \
        f"C_JAL encoding greșit: 0x{c_jal_w:04x}"

    # C_BEQZ: quadrant=01, funct3=110
    c_beqz_w = encode(int(Op.C_BEQZ), 0, 0, 0, 0) & 0xFFFF
    assert (c_beqz_w & 0b11) == 0b01 and (c_beqz_w >> 13) == 0b110, \
        f"C_BEQZ encoding greșit: 0x{c_beqz_w:04x}"

    # C_BNEZ: quadrant=01, funct3=111
    c_bnez_w = encode(int(Op.C_BNEZ), 0, 0, 0, 0) & 0xFFFF
    assert (c_bnez_w & 0b11) == 0b01 and (c_bnez_w >> 13) == 0b111, \
        f"C_BNEZ encoding greșit: 0x{c_bnez_w:04x}"

    # C_ADDI16SP: quadrant=01, funct3=011, rd=2
    c_a16sp_w = encode(int(Op.C_ADDI16SP), 0, 0, 0, 0) & 0xFFFF
    assert (c_a16sp_w & 0b11) == 0b01 and (c_a16sp_w >> 13) == 0b011 \
        and ((c_a16sp_w >> 7) & 0x1F) == 2, \
        f"C_ADDI16SP encoding greșit: 0x{c_a16sp_w:04x}"

    # C_LWSP: quadrant=10, funct3=010
    c_lwsp_w = encode(int(Op.C_LWSP), 5, 0, 0, 0) & 0xFFFF
    assert (c_lwsp_w & 0b11) == 0b10 and (c_lwsp_w >> 13) == 0b010, \
        f"C_LWSP encoding greșit: 0x{c_lwsp_w:04x}"

    # C_SWSP: quadrant=10, funct3=110
    c_swsp_w = encode(int(Op.C_SWSP), 0, 0, 5, 0) & 0xFFFF
    assert (c_swsp_w & 0b11) == 0b10 and (c_swsp_w >> 13) == 0b110, \
        f"C_SWSP encoding greșit: 0x{c_swsp_w:04x}"

    # C_JR: quadrant=10, bit[12]=0, rs2=0
    c_jr_w = encode(int(Op.C_JR), 0, 3, 0, 0) & 0xFFFF
    assert (c_jr_w & 0b11) == 0b10 and ((c_jr_w >> 12) & 0xF) == 0b1000, \
        f"C_JR encoding greșit: 0x{c_jr_w:04x}"

    # C_JALR: quadrant=10, bit[12]=1, rs2=0
    c_jalr_w = encode(int(Op.C_JALR), 0, 3, 0, 0) & 0xFFFF
    assert (c_jalr_w & 0b11) == 0b10 and ((c_jalr_w >> 12) & 0xF) == 0b1001, \
        f"C_JALR encoding greșit: 0x{c_jalr_w:04x}"

    print(f"[OK] L9 self-test: {N_OPS} ops × {IMM_BUCKETS} buckets")
    print(f"  MRET:       0x{_MRET:08x}")
    print(f"  WFI:        0x{_WFI:08x}")
    print(f"  FENCE.I:    0x{_FENCE_I:08x}")
    print(f"  C_J(off=4): 0x{encode(int(Op.C_J),0,0,0,0):08x}")
    print(f"  C_JAL(4):   0x{encode(int(Op.C_JAL),0,0,0,0):08x}")
    print(f"  C_BEQZ(4):  0x{encode(int(Op.C_BEQZ),0,2,0,0):08x}")
    print(f"  C_JR(x3):   0x{encode(int(Op.C_JR),0,3,0,0):08x}")
    print(f"  C_JALR(x3): 0x{encode(int(Op.C_JALR),0,3,0,0):08x}")
    print(f"  C_EBREAK:   0x{encode(int(Op.C_EBREAK),0,0,0,0):08x}")


if __name__ == "__main__":
    _self_test()
