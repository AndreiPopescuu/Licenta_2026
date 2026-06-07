"""codec_l9_focused.py — 30 ops focalizate pe ibex_controller + CSR paths.

Schimbări față de codec_l9:
  - encode() primește csr_bucket explicit (index direct în L9_CSRS)
    → agentul controlează direct ce CSR accesează
  - Op set redus la 30 ops: ops esențiale + toate instrucțiunile L9-noi
    → spațiu de acțiuni mai mic, convergență mai rapidă

Op set:
  ADDI, LUI, ADD                        — setup registre
  CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI  — acces direct CSR
  ECALL, EBREAK, MRET, WFI, FENCE_I    — excepții + control flow special
  BEQ, BNE, JAL, JALR, LW, SW          — branch + memorie
  C_JAL, C_J, C_BEQZ, C_BNEZ          — compressed jumps/branches
  C_ADDI16SP, C_LWSP, C_SWSP          — compressed stack
  C_JR, C_JALR, C_EBREAK              — compressed jump-register + break

Total: 30 ops.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L9 = Path(__file__).resolve().parent
_L7 = _L9.parent / "level7_stimulus"
sys.path.insert(0, str(_L9))
sys.path.insert(0, str(_L7))

from codec_l9 import (  # noqa: E402
    L9_CSRS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    _MRET, _WFI, _FENCE_I,
    _C_NOP, _C_EBREAK_16,
    _C_JUMP_OFFSETS, _C_BRANCH_OFFSETS, _C_ADDI16SP_OFFS, _C_LSTK_OFFSETS,
    _pack_rvc,
    _c_jump_imm_bits,
    _encode_c_j, _encode_c_jal,
    _encode_c_branch,
    _encode_c_addi16sp,
    _encode_c_lwsp, _encode_c_swsp,
    _encode_c_jr, _encode_c_jalr,
)

IMM_BUCKETS   = IMM_BUCKETS   # re-export
N_CSR_BUCKETS = len(L9_CSRS)  # 33 — agent alege direct indexul în pool

_ADDI_IMMS = [0, 1, 4, 8, -1]
_LUI_IMMS  = [0x00000, 0x00001, 0x80000, 0xFFFFF, 0x12345]
_JAL_OFFS  = [4, 8, 12, 16, 20]

_CSR_F3  = {3: 0b001, 4: 0b010, 5: 0b011}   # CSRRW, CSRRS, CSRRC
_CSRI_F3 = {6: 0b101, 7: 0b110, 8: 0b111}   # CSRRWI, CSRRSI, CSRRCI

_ECALL  = 0x00000073
_EBREAK = 0x00100073


class Op(IntEnum):
    ADDI       = 0
    LUI        = 1
    ADD        = 2
    CSRRW      = 3
    CSRRS      = 4
    CSRRC      = 5
    CSRRWI     = 6
    CSRRSI     = 7
    CSRRCI     = 8
    ECALL      = 9
    EBREAK     = 10
    MRET       = 11
    WFI        = 12
    FENCE_I    = 13
    BEQ        = 14
    BNE        = 15
    JAL        = 16
    JALR       = 17
    LW         = 18
    SW         = 19
    C_JAL      = 20
    C_J        = 21
    C_BEQZ     = 22
    C_BNEZ     = 23
    C_ADDI16SP = 24
    C_LWSP     = 25
    C_SWSP     = 26
    C_JR       = 27
    C_JALR     = 28
    C_EBREAK   = 29


N_OPS = 30


def _encode_csr(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    csr = L9_CSRS[csr_bucket % len(L9_CSRS)]
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_csri(funct3: int, rd: int, uimm: int, csr_bucket: int) -> int:
    csr = L9_CSRS[csr_bucket % len(L9_CSRS)]
    return (csr << 20) | ((uimm & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_addi(rd: int, rs1: int, imm_bucket: int) -> int:
    imm = _ADDI_IMMS[imm_bucket % len(_ADDI_IMMS)] & 0xFFF
    return (imm << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b0010011


def _encode_lui(rd: int, imm_bucket: int) -> int:
    imm20 = _LUI_IMMS[imm_bucket % len(_LUI_IMMS)] & 0xFFFFF
    return (imm20 << 12) | ((rd & 0x1F) << 7) | 0b0110111


def _encode_add(rd: int, rs1: int, rs2: int) -> int:
    return ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b0110011


def _encode_branch(funct3: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    off = _JAL_OFFS[imm_bucket % len(_JAL_OFFS)] & 0x1FFE
    b12 = (off >> 12) & 1; b11 = (off >> 11) & 1
    b10_5 = (off >> 5) & 0x3F; b4_1 = (off >> 1) & 0xF
    return (b12 << 31) | (b10_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (funct3 << 12) | (b4_1 << 8) | (b11 << 7) | 0b1100011


def _encode_jal(rd: int, imm_bucket: int) -> int:
    off = _JAL_OFFS[imm_bucket % len(_JAL_OFFS)] & 0x1FFFFE
    b20    = (off >> 20) & 1; b10_1 = (off >> 1) & 0x3FF
    b11    = (off >> 11) & 1; b19_12 = (off >> 12) & 0xFF
    return (b20 << 31) | (b10_1 << 21) | (b11 << 20) | (b19_12 << 12) \
         | ((rd & 0x1F) << 7) | 0b1101111


def _encode_jalr(rd: int, rs1: int, imm_bucket: int) -> int:
    imm = (_JAL_OFFS[imm_bucket % len(_JAL_OFFS)]) & 0xFFF
    return (imm << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b1100111


def _encode_lw(rd: int, rs1: int, imm_bucket: int) -> int:
    imm = (_JAL_OFFS[imm_bucket % len(_JAL_OFFS)] * 4) & 0xFFF
    return (imm << 20) | ((rs1 & 0x1F) << 15) | (0b010 << 12) | ((rd & 0x1F) << 7) | 0b0000011


def _encode_sw(rs1: int, rs2: int, imm_bucket: int) -> int:
    imm = (_JAL_OFFS[imm_bucket % len(_JAL_OFFS)] * 4) & 0xFFF
    imm11_5 = (imm >> 5) & 0x7F; imm4_0 = imm & 0x1F
    return (imm11_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (0b010 << 12) | (imm4_0 << 7) | 0b0100011


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int, csr_bucket: int = 0) -> int:
    op = Op(op_i)
    if op == Op.ADDI:       return _encode_addi(rd, rs1, imm_bucket)
    if op == Op.LUI:        return _encode_lui(rd, imm_bucket)
    if op == Op.ADD:        return _encode_add(rd, rs1, rs2)
    if op == Op.CSRRW:      return _encode_csr(_CSR_F3[3],  rd, rs1, csr_bucket)
    if op == Op.CSRRS:      return _encode_csr(_CSR_F3[4],  rd, rs1, csr_bucket)
    if op == Op.CSRRC:      return _encode_csr(_CSR_F3[5],  rd, rs1, csr_bucket)
    if op == Op.CSRRWI:     return _encode_csri(_CSRI_F3[6], rd, rs1, csr_bucket)
    if op == Op.CSRRSI:     return _encode_csri(_CSRI_F3[7], rd, rs1, csr_bucket)
    if op == Op.CSRRCI:     return _encode_csri(_CSRI_F3[8], rd, rs1, csr_bucket)
    if op == Op.ECALL:      return _ECALL
    if op == Op.EBREAK:     return _EBREAK
    if op == Op.MRET:       return _MRET
    if op == Op.WFI:        return _WFI
    if op == Op.FENCE_I:    return _FENCE_I
    if op == Op.BEQ:        return _encode_branch(0b000, rs1, rs2, imm_bucket)
    if op == Op.BNE:        return _encode_branch(0b001, rs1, rs2, imm_bucket)
    if op == Op.JAL:        return _encode_jal(rd, imm_bucket)
    if op == Op.JALR:       return _encode_jalr(rd, rs1, imm_bucket)
    if op == Op.LW:         return _encode_lw(rd, rs1, imm_bucket)
    if op == Op.SW:         return _encode_sw(rs1, rs2, imm_bucket)
    if op == Op.C_JAL:      return _pack_rvc(_encode_c_jal(imm_bucket))
    if op == Op.C_J:        return _pack_rvc(_encode_c_j(imm_bucket))
    if op == Op.C_BEQZ:     return _pack_rvc(_encode_c_branch(0b110, rs1, imm_bucket))
    if op == Op.C_BNEZ:     return _pack_rvc(_encode_c_branch(0b111, rs1, imm_bucket))
    if op == Op.C_ADDI16SP: return _pack_rvc(_encode_c_addi16sp(imm_bucket))
    if op == Op.C_LWSP:     return _pack_rvc(_encode_c_lwsp(rd, imm_bucket))
    if op == Op.C_SWSP:     return _pack_rvc(_encode_c_swsp(rs2, imm_bucket))
    if op == Op.C_JR:       return _pack_rvc(_encode_c_jr(rs1))
    if op == Op.C_JALR:     return _pack_rvc(_encode_c_jalr(rs1))
    if op == Op.C_EBREAK:   return _pack_rvc(_C_EBREAK_16)
    raise ValueError(f"Op necunoscut: {op_i}")


def emit_program(actions):
    nop = _encode_addi(0, 0, 0)
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            for cb in range(N_CSR_BUCKETS):
                w = encode(op_i, 5, 6, 7, ib, cb)
                assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} cb={cb} → 0x{w:08x}"

    assert encode(int(Op.ECALL),   0, 0, 0, 0, 0) == _ECALL
    assert encode(int(Op.EBREAK),  0, 0, 0, 0, 0) == _EBREAK
    assert encode(int(Op.MRET),    0, 0, 0, 0, 0) == _MRET
    assert encode(int(Op.WFI),     0, 0, 0, 0, 0) == _WFI
    assert encode(int(Op.FENCE_I), 0, 0, 0, 0, 0) == _FENCE_I

    # CSR direct: csr_bucket=0 → L9_CSRS[0]
    csrrw = encode(int(Op.CSRRW), 1, 2, 0, 0, 0)
    assert (csrrw & 0x7F) == 0b1110011
    assert ((csrrw >> 20) & 0xFFF) == L9_CSRS[0]

    # CSR direct: csr_bucket pentru mcycle (0xB00)
    mcycle_idx = L9_CSRS.index(0xB00)
    csrrw_mc = encode(int(Op.CSRRW), 0, 1, 0, 0, mcycle_idx)
    assert ((csrrw_mc >> 20) & 0xFFF) == 0xB00

    # C_J: quadrant=01, funct3=101
    c_j_w = encode(int(Op.C_J), 0, 0, 0, 0, 0) & 0xFFFF
    assert (c_j_w & 0b11) == 0b01 and (c_j_w >> 13) == 0b101

    # C_EBREAK
    assert encode(int(Op.C_EBREAK), 0, 0, 0, 0, 0) == (_C_NOP << 16) | _C_EBREAK_16

    print(f"[OK] L9-focused self-test: {N_OPS} ops × {IMM_BUCKETS} imm × {N_CSR_BUCKETS} csr")
    print(f"  CSR pool: {len(L9_CSRS)} registre (agent alege direct)")
    print(f"  mcycle la csr_bucket={mcycle_idx}")


if __name__ == "__main__":
    _self_test()
