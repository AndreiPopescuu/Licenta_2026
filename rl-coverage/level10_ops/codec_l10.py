"""Level 10 codec — L9 (83 ops) + 4 new ops = 87 ops.

Țintă principală: ibex_cs_registers (50.4% la L9).

Problema L9: deși CSR-urile sunt bine acoperite, modulul ibex_cs_registers
are coverage groups separate pentru fiecare tip de excepție (mcause value).
La L9 avem doar:
  - mcause=3  (EBREAK / C_EBREAK)
  - mcause=11 (ECALL — environment call from M-mode)
  - mcause=3  (WFI nu generează excepție — e handled de controller)

Lipsesc complet:
  - mcause=2  → illegal instruction
  - mcause=4  → load address misaligned
  - mcause=6  → store/AMO address misaligned

Fiecare dintre acestea are bins separate în:
  - cp_exc_cause (ibex_cs_registers)
  - cp_mtval_written (mtval prinde PC pentru illegal, adresa pentru misalign)
  - ibex_controller FSM paths (HANDLE_MEM_EX branch)

Adăugăm și mip (0x344) în lista CSR — singurul registru de trap-control
complet absent din L9_CSRS.

Encodinguri fixe (nu depind de rd/rs1/rs2/imm):
  ILLEGAL_INSN  = 0x0000000B  (CUSTOM-0, bit[1:0]=11, opcode=0b0001011)
                  → Ibex nu implementează custom-0 → illegal instruction
  LW_MISALIGN   = 0x00102003  (LW x0, 1(x0) — adresa 1, misaliniat pt word)
  SW_MISALIGN   = 0x000020A3  (SW x0, 1(x0) — adresa 1, misaliniat pt word)
  LH_MISALIGN   = 0x00101003  (LH x0, 1(x0) — adresa 1, misaliniat pt half)

Trap handler-ul existent (instalat de cocotb) face mepc += 4 + MRET pentru
orice excepție, deci instrucțiunile de mai sus sunt sigure în simulare.

Total: 87 ops.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L9 = Path(__file__).resolve().parent.parent / "level9_ops"
_L7 = Path(__file__).resolve().parent.parent / "level7_stimulus"
sys.path.insert(0, str(_L9))
sys.path.insert(0, str(_L7))

from codec_l9 import (  # noqa: E402
    Op as L9Op,
    N_OPS as L9_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    L9_CSRS,
    encode as l9_encode,
)

# L10: adaugă mip la pool-ul de CSR-uri.
# mip (0x344) era singurul registru de trap-control absent din L9_CSRS.
# E read-mostly (majoritatea biților reflectă linii hardware de întrerupere),
# dar citirea lui exercită decode path-ul CSR și umple cp_csr_read bins.
L10_CSRS = L9_CSRS + [
    0x344,  # mip — machine interrupt pending
]

N_CSR_BUCKETS = len(L10_CSRS)  # agent alege direct indexul în pool

_CSR_F3  = {27: 0b001, 28: 0b010, 29: 0b011}   # CSRRW, CSRRS, CSRRC
_CSRI_F3 = {66: 0b101, 67: 0b110, 68: 0b111}   # CSRRWI, CSRRSI, CSRRCI


def _encode_csr_l10(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    csr = L10_CSRS[csr_bucket % len(L10_CSRS)]
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_csri_l10(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    uimm = rs1 & 0x1F
    csr  = L10_CSRS[csr_bucket % len(L10_CSRS)]
    return (csr << 20) | ((uimm & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


class Op(IntEnum):
    # L9 ops identice (0-82)
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
    # L10 additions — exception diversity
    ILLEGAL_INSN = 83  # mcause=2: illegal instruction
    LW_MISALIGN  = 84  # mcause=4: load address misaligned
    SW_MISALIGN  = 85  # mcause=6: store/AMO address misaligned
    LH_MISALIGN  = 86  # mcause=4: load misalign (halfword path, alt bins)


N_OPS = 87

# Encodinguri fixe pentru ops noi
# CUSTOM-0: opcode=0b0001011, toate celelalte câmpuri 0.
# Ibex nu implementează custom-0 → IllegalInstruction exception, mcause=2.
# mtval primește PC-ul instrucțiunii ilegale.
_ILLEGAL_INSN = 0x0000000B

# LW x0, 1(x0): adresa = 0+1 = 1, misaliniat pentru word (4B align).
# mcause=4 (Load Address Misaligned). mtval = adresa 1.
# Verificarea misalignment are loc ÎNAINTE de accesul la memorie (RISC-V spec),
# deci nu contează că adresa 1 nu e mapped — excepția e mcause=4, nu mcause=5.
_LW_MISALIGN = 0x00102003

# SW x0, 1(x0): adresa = 0+1 = 1, misaliniat pentru word.
# mcause=6 (Store/AMO Address Misaligned). mtval = adresa 1.
_SW_MISALIGN = 0x000020A3

# LH x0, 1(x0): adresa = 0+1 = 1, misaliniat pentru halfword (2B align).
# Acoperă același mcause=4 dar prin funct3=001 (LH) în loc de 010 (LW),
# exercitând un branch diferit în ibex_load_store_unit.
_LH_MISALIGN = 0x00101003


def _encode_lw_misalign(rs1: int) -> int:
    return (1 << 20) | ((rs1 & 0x1F) << 15) | (0b010 << 12) | 0b0000011

def _encode_sw_misalign(rs1: int) -> int:
    return ((rs1 & 0x1F) << 15) | (0b010 << 12) | (1 << 7) | 0b0100011

def _encode_lh_misalign(rs1: int) -> int:
    return (1 << 20) | ((rs1 & 0x1F) << 15) | (0b001 << 12) | 0b0000011


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int, csr_bucket: int = 0) -> int:
    # CSR ops — agent alege direct via csr_bucket
    if op_i in _CSR_F3:
        return _encode_csr_l10(_CSR_F3[op_i], rd, rs1, csr_bucket)
    if op_i in _CSRI_F3:
        return _encode_csri_l10(_CSRI_F3[op_i], rd, rs1, csr_bucket)
    # Ops noi L10 — misalign cu rs1 configurat
    if op_i == int(Op.ILLEGAL_INSN):
        return _ILLEGAL_INSN
    if op_i == int(Op.LW_MISALIGN):
        return _encode_lw_misalign(rs1)
    if op_i == int(Op.SW_MISALIGN):
        return _encode_sw_misalign(rs1)
    if op_i == int(Op.LH_MISALIGN):
        return _encode_lh_misalign(rs1)
    # Tot ce e < 83 e op L9
    return l9_encode(op_i, rd, rs1, rs2, imm_bucket)


def emit_program(actions):
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)  # ADDI x0, x0, 0 — NOP
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} → 0x{w:08x}"

    # Verificări encoding
    assert encode(int(Op.ILLEGAL_INSN), 0, 0, 0, 0) == _ILLEGAL_INSN
    assert encode(int(Op.LW_MISALIGN),  0, 0, 0, 0) == _LW_MISALIGN
    assert encode(int(Op.SW_MISALIGN),  0, 0, 0, 0) == _SW_MISALIGN
    assert encode(int(Op.LH_MISALIGN),  0, 0, 0, 0) == _LH_MISALIGN

    # ILLEGAL_INSN: opcode = 0b0001011 (CUSTOM-0), bits[1:0]=11
    assert (_ILLEGAL_INSN & 0x7F) == 0b0001011

    # LW_MISALIGN: opcode=LOAD (0b0000011), funct3=010 (LW), imm=1
    assert (_LW_MISALIGN & 0x7F) == 0b0000011          # LOAD opcode
    assert ((_LW_MISALIGN >> 12) & 0x7) == 0b010       # funct3=LW
    assert ((_LW_MISALIGN >> 20) & 0xFFF) == 1         # imm=1 → misaliniat

    # SW_MISALIGN: opcode=STORE (0b0100011), funct3=010 (SW)
    assert (_SW_MISALIGN & 0x7F) == 0b0100011          # STORE opcode
    assert ((_SW_MISALIGN >> 12) & 0x7) == 0b010       # funct3=SW

    # LH_MISALIGN: opcode=LOAD (0b0000011), funct3=001 (LH), imm=1
    assert (_LH_MISALIGN & 0x7F) == 0b0000011          # LOAD opcode
    assert ((_LH_MISALIGN >> 12) & 0x7) == 0b001       # funct3=LH
    assert ((_LH_MISALIGN >> 20) & 0xFFF) == 1         # imm=1 → misaliniat

    # mip (0x344) accesibil prin CSR ops
    n = len(L10_CSRS)
    mip_in_pool = 0x344 in L10_CSRS
    assert mip_in_pool, "mip (0x344) lipsește din L10_CSRS!"

    print(f"[OK] L10 self-test: {N_OPS} ops × {IMM_BUCKETS} buckets")
    print(f"  CSR pool: {n} registre (L9={len(L9_CSRS)}, +1 mip)")
    print(f"  ILLEGAL_INSN: 0x{_ILLEGAL_INSN:08x}  → mcause=2")
    print(f"  LW_MISALIGN:  0x{_LW_MISALIGN:08x}  → mcause=4 (LW x0,1(x0))")
    print(f"  SW_MISALIGN:  0x{_SW_MISALIGN:08x}  → mcause=6 (SW x0,1(x0))")
    print(f"  LH_MISALIGN:  0x{_LH_MISALIGN:08x}  → mcause=4 (LH x0,1(x0))")


if __name__ == "__main__":
    _self_test()
