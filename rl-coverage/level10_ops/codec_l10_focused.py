"""codec_l10_focused.py — 22 ops focalizate pe ibex_cs_registers.

Schimbări față de v1:
  - encode() primește csr_bucket explicit (index direct în L10_CSRS, nu hash)
    → agentul controlează direct ce CSR accesează
  - LW_MISALIGN, SW_MISALIGN, LH_MISALIGN folosesc rs1 ca bază (nu x0 fix)
    → adresa = rs1_value + 1 → agent poate togglui mai mulți biți din mtval
"""

import sys
from pathlib import Path
from enum import IntEnum

_L10 = Path(__file__).resolve().parent
_L9  = _L10.parent / "level9_ops"
_L7  = _L10.parent / "level7_stimulus"
sys.path.insert(0, str(_L9))
sys.path.insert(0, str(_L7))

from codec_l10 import L10_CSRS, IMM_BUCKETS, IMM_BUCKET_VALUES  # noqa: E402

IMM_BUCKETS   = IMM_BUCKETS   # re-export
N_CSR_BUCKETS = len(L10_CSRS) # 33 — agent alege direct indexul în pool

_CSR_F3  = {3: 0b001, 4: 0b010, 5: 0b011}   # CSRRW=3, CSRRS=4, CSRRC=5
_CSRI_F3 = {6: 0b101, 7: 0b110, 8: 0b111}   # CSRRWI=6, CSRRSI=7, CSRRCI=8

_ADDI_IMMS = [0, 1, 4, 8, -1]
_LUI_IMMS  = [0x00000, 0x00001, 0x80000, 0xFFFFF, 0x12345]
_JAL_OFFS  = [4, 8, 12, 16, 20]

_ECALL    = 0x00000073
_EBREAK   = 0x00100073
_MRET     = 0x30200073
_WFI      = 0x10500073
_ILLEGAL  = 0x0000000B   # CUSTOM-0 → mcause=2


class Op(IntEnum):
    ADDI         = 0
    LUI          = 1
    ADD          = 2
    CSRRW        = 3
    CSRRS        = 4
    CSRRC        = 5
    CSRRWI       = 6
    CSRRSI       = 7
    CSRRCI       = 8
    ECALL        = 9
    EBREAK       = 10
    MRET         = 11
    WFI          = 12
    BEQ          = 13
    BNE          = 14
    JAL          = 15
    LW           = 16
    SW           = 17
    ILLEGAL_INSN = 18
    LW_MISALIGN  = 19
    SW_MISALIGN  = 20
    LH_MISALIGN  = 21


N_OPS = 22


def _encode_csr(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    csr = L10_CSRS[csr_bucket % len(L10_CSRS)]
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_csri(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    uimm = rs1 & 0x1F  # câmpul rs1 e refolosit ca immediate pe 5 biți
    csr  = L10_CSRS[csr_bucket % len(L10_CSRS)]
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


def _encode_lw(rd: int, rs1: int, imm_bucket: int) -> int:
    imm = (_JAL_OFFS[imm_bucket % len(_JAL_OFFS)] * 4) & 0xFFF
    return (imm << 20) | ((rs1 & 0x1F) << 15) | (0b010 << 12) | ((rd & 0x1F) << 7) | 0b0000011


def _encode_sw(rs1: int, rs2: int, imm_bucket: int) -> int:
    imm = (_JAL_OFFS[imm_bucket % len(_JAL_OFFS)] * 4) & 0xFFF
    imm11_5 = (imm >> 5) & 0x7F; imm4_0 = imm & 0x1F
    return (imm11_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (0b010 << 12) | (imm4_0 << 7) | 0b0100011


def _encode_lw_misalign(rs1: int) -> int:
    # LW x0, 1(rs1) — adresa = rs1_value+1, misaliniat pt word
    return (1 << 20) | ((rs1 & 0x1F) << 15) | (0b010 << 12) | 0b0000011


def _encode_sw_misalign(rs1: int) -> int:
    # SW x0, 1(rs1) — adresa = rs1_value+1, mcause=6
    # S-type: imm[11:5]=0, rs2=x0, rs1=rs1, funct3=010, imm[4:0]=1
    return ((rs1 & 0x1F) << 15) | (0b010 << 12) | (1 << 7) | 0b0100011


def _encode_lh_misalign(rs1: int) -> int:
    # LH x0, 1(rs1) — adresa = rs1_value+1, misaliniat pt halfword
    return (1 << 20) | ((rs1 & 0x1F) << 15) | (0b001 << 12) | 0b0000011


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int, csr_bucket: int = 0) -> int:
    op = Op(op_i)
    if op == Op.ADDI:         return _encode_addi(rd, rs1, imm_bucket)
    if op == Op.LUI:          return _encode_lui(rd, imm_bucket)
    if op == Op.ADD:          return _encode_add(rd, rs1, rs2)
    if op == Op.CSRRW:        return _encode_csr(_CSR_F3[3],  rd, rs1, csr_bucket)
    if op == Op.CSRRS:        return _encode_csr(_CSR_F3[4],  rd, rs1, csr_bucket)
    if op == Op.CSRRC:        return _encode_csr(_CSR_F3[5],  rd, rs1, csr_bucket)
    if op == Op.CSRRWI:       return _encode_csri(_CSRI_F3[6], rd, rs1, csr_bucket)
    if op == Op.CSRRSI:       return _encode_csri(_CSRI_F3[7], rd, rs1, csr_bucket)
    if op == Op.CSRRCI:       return _encode_csri(_CSRI_F3[8], rd, rs1, csr_bucket)
    if op == Op.ECALL:        return _ECALL
    if op == Op.EBREAK:       return _EBREAK
    if op == Op.MRET:         return _MRET
    if op == Op.WFI:          return _WFI
    if op == Op.BEQ:          return _encode_branch(0b000, rs1, rs2, imm_bucket)
    if op == Op.BNE:          return _encode_branch(0b001, rs1, rs2, imm_bucket)
    if op == Op.JAL:          return _encode_jal(rd, imm_bucket)
    if op == Op.LW:           return _encode_lw(rd, rs1, imm_bucket)
    if op == Op.SW:           return _encode_sw(rs1, rs2, imm_bucket)
    if op == Op.ILLEGAL_INSN: return _ILLEGAL
    if op == Op.LW_MISALIGN:  return _encode_lw_misalign(rs1)
    if op == Op.SW_MISALIGN:  return _encode_sw_misalign(rs1)
    if op == Op.LH_MISALIGN:  return _encode_lh_misalign(rs1)
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

    assert encode(int(Op.ECALL),        0, 0, 0, 0, 0) == _ECALL
    assert encode(int(Op.EBREAK),       0, 0, 0, 0, 0) == _EBREAK
    assert encode(int(Op.MRET),         0, 0, 0, 0, 0) == _MRET
    assert encode(int(Op.WFI),          0, 0, 0, 0, 0) == _WFI
    assert encode(int(Op.ILLEGAL_INSN), 0, 0, 0, 0, 0) == _ILLEGAL

    # Misalign cu rs1=0: adresa=1 (la fel ca înainte)
    lw_mis = encode(int(Op.LW_MISALIGN), 0, 0, 0, 0, 0)
    assert (lw_mis & 0x7F) == 0b0000011
    assert ((lw_mis >> 12) & 0x7) == 0b010
    assert ((lw_mis >> 20) & 0xFFF) == 1

    # Misalign cu rs1=8: adresa=x8+1 (biți noi în mtval)
    lw_mis_r8 = encode(int(Op.LW_MISALIGN), 0, 8, 0, 0, 0)
    assert ((lw_mis_r8 >> 15) & 0x1F) == 8

    # CSR direct: csr_bucket=0 → L10_CSRS[0]
    csrrw = encode(int(Op.CSRRW), 1, 2, 0, 0, 0)
    assert (csrrw & 0x7F) == 0b1110011
    expected_csr = L10_CSRS[0]
    assert ((csrrw >> 20) & 0xFFF) == expected_csr

    # CSR direct: csr_bucket pentru mcycle (0xB00)
    mcycle_idx = L10_CSRS.index(0xB00)
    csrrw_mc = encode(int(Op.CSRRW), 0, 1, 0, 0, mcycle_idx)
    assert ((csrrw_mc >> 20) & 0xFFF) == 0xB00

    print(f"[OK] L10-focused v2 self-test: {N_OPS} ops × {IMM_BUCKETS} imm × {N_CSR_BUCKETS} csr")
    print(f"  CSR pool: {len(L10_CSRS)} registre (agent alege direct)")
    print(f"  mcycle la csr_bucket={mcycle_idx}")
    print(f"  Misalign ops: folosesc rs1 ca bază (nu x0 fix)")


if __name__ == "__main__":
    _self_test()
