"""Level 7 codec — L6 RVC codec + AUIPC + ECALL + EBREAK.

Extends codec_rvc (61 ops) with three ops that the analyze_unreachable
report flagged as the biggest reachable-but-unemittable gap:

  + AUIPC (U-type)    — unlocks imm_u_type_o paths in the decoder / ALU
  + ECALL (SYSTEM)    — unlocks the whole exception path
                        (mepc, mtval, mcause, mstack_*, controller FSM).
                        The L7 cocotb driver installs a trap handler that
                        advances mepc before MRET, so ECALL doesn't loop.
  + EBREAK (SYSTEM)   — second exception cause code, same path.

Total: 64 ops.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L6 = (Path(__file__).resolve().parent.parent / "level6_rvc")
sys.path.insert(0, str(_L6))

from codec_rvc import (  # noqa: E402
    Op as L6Op,
    N_OPS as L6_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    SHAMT_BUCKET_VALUES,
    BRANCH_BUCKET_OFFSETS,
    JAL_BUCKET_OFFSETS,
    SAFE_CSRS,
    RVC_ALL,
    C_NOP,
    encode_base as l6_encode_base,
    encode_rvc as l6_encode_rvc,
    emit_program as _unused_emit,
)


class Op(IntEnum):
    # Keep the first 61 ops identical to L6 (so a policy trained on L6
    # transfers trivially here).
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
    # --- Level 7 additions ---
    AUIPC  = 61    # U-type, diverse upper-immediate buckets
    ECALL  = 62    # trap via mtvec, handler advances mepc, MRET returns
    EBREAK = 63    # same trap path, different mcause


N_OPS = 64

# AUIPC: 20-bit upper-immediate.  Buckets chosen to toggle high bits.
AUIPC_IMM_BUCKETS = [0x00000, 0x00001, 0x12345, 0xABCDE, 0xFFFFF]

# Override L6's 5-CSR safe list with a much wider set.  All of these are
# safe to CSRRS/CSRRC/CSRRW because either:
#   - they're read-only (writes are ignored or become hints), OR
#   - they store data that doesn't affect control flow (mscratch, mepc,
#     mcause, mtval), OR
#   - they're counter CSRs that can be clobbered with no ill effect.
# Writing to {mtvec, mie, mip, mstatus} is avoided because those change
# trap / IRQ behaviour.
L7_SAFE_CSRS = [
    # Machine trap handling (safe to read; safe to write mscratch/mepc/mcause/mtval
    # because we don't rely on their values outside the trap handler)
    0x340,  # mscratch  (R/W scratchpad)
    0x341,  # mepc      (R/W, clobbered on next trap anyway)
    0x342,  # mcause    (R/W)
    0x343,  # mtval     (R/W)
    # Machine information (read-only, writes become no-ops)
    0xF11,  # mvendorid
    0xF12,  # marchid
    0xF13,  # mimpid
    0xF14,  # mhartid
    0xF15,  # mconfigptr
    # Machine counters (R/W, read even when unused — always live in Ibex)
    0xB00,  # mcycle
    0xB02,  # minstret
    0xB80,  # mcycleh
    0xB82,  # minstreth
    # Count inhibit (R/W, gating the counters)
    0x320,  # mcountinhibit
    # Performance counter handles (R/O zero when MHPMCounterNum=0, but the
    # decode path still fires — exercises the CSR address decoder)
    0xB03, 0xB04, 0xB05, 0xB06, 0xB07, 0xB08,  # mhpmcounter3..8
    0x323, 0x324, 0x325, 0x326, 0x327, 0x328,  # mhpmevent3..8
    # User-level read-only counters
    0xC00,  # cycle
    0xC01,  # time
    0xC02,  # instret
]


def encode_auipc(rd: int, imm_bucket: int) -> int:
    """AUIPC rd, imm  —  rd = PC + (imm << 12).  U-type, opcode = 0b0010111."""
    imm20 = AUIPC_IMM_BUCKETS[imm_bucket % len(AUIPC_IMM_BUCKETS)] & 0xFFFFF
    return (imm20 << 12) | ((rd & 0x1F) << 7) | 0b0010111


def encode_ecall() -> int:
    """ECALL  —  I-type with imm=0, rs1=0, rd=0, funct3=0, opcode=0b1110011."""
    return 0x00000073


def encode_ebreak() -> int:
    """EBREAK —  same layout, imm=1."""
    return 0x00100073


# CSR opcode and funct3 encodings (copied from codec_rvc).
_CSR_F3 = {27: 0b001, 28: 0b010, 29: 0b011}   # CSRRW, CSRRS, CSRRC


def _encode_csr_l7(op_i: int, rd: int, rs1: int, imm_bucket: int) -> int:
    """CSR encoder that uses (imm_bucket ^ rs1) to select from L7_SAFE_CSRS.

    Uses all 5 bits of rs1 plus the imm bucket so a random agent cycles
    through all ~30 CSRs rather than just the 5 the L6 codec reached.
    """
    csr_idx = (imm_bucket * 7 + rs1) % len(L7_SAFE_CSRS)
    csr = L7_SAFE_CSRS[csr_idx]
    f3 = _CSR_F3[op_i]
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    if op_i == int(Op.AUIPC):
        return encode_auipc(rd, imm_bucket)
    if op_i == int(Op.ECALL):
        return encode_ecall()
    if op_i == int(Op.EBREAK):
        return encode_ebreak()
    # CSR ops: override L6's 5-CSR selector to use L7_SAFE_CSRS (30 CSRs).
    if op_i in (27, 28, 29):
        return _encode_csr_l7(op_i, rd, rs1, imm_bucket)
    # Anything else < 61 is an L6 op — delegate.
    l6_op = L6Op(op_i)
    if l6_op in RVC_ALL:
        rvc_word = l6_encode_rvc(l6_op, rd, rs1, rs2, imm_bucket)
        return (C_NOP << 16) | (rvc_word & 0xFFFF)
    return l6_encode_base(l6_op, rd, rs1, rs2, imm_bucket)


def emit_program(actions):
    """Same contract as L6: one 32-bit word per action, 16 NOP tail."""
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    # Encode every op across every imm bucket, confirm opcode bits for new ops.
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF
    # AUIPC opcode check
    for ib in range(IMM_BUCKETS):
        w = encode(int(Op.AUIPC), 3, 0, 0, ib)
        assert (w & 0x7F) == 0b0010111, f"AUIPC opcode wrong: 0x{w:08x}"
    # ECALL / EBREAK
    assert encode(int(Op.ECALL), 0, 0, 0, 0) == 0x00000073
    assert encode(int(Op.EBREAK), 0, 0, 0, 0) == 0x00100073
    print(f"[OK] L7 self-test: {N_OPS} ops × {IMM_BUCKETS} buckets.")
    print(f"  AUIPC samples:  " + ", ".join(
        f"0x{encode(int(Op.AUIPC),3,0,0,ib):08x}" for ib in range(IMM_BUCKETS)))
    print(f"  ECALL:  0x{encode(int(Op.ECALL),0,0,0,0):08x}")
    print(f"  EBREAK: 0x{encode(int(Op.EBREAK),0,0,0,0):08x}")


if __name__ == "__main__":
    _self_test()
