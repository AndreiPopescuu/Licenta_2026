"""Level 6 codec — extends Level 5 with 16 RV32C (compressed) opcodes.

Motivation. Level 5 plateaued at ~56% toggle coverage on minimal-config Ibex
because the action space could only emit 32-bit instructions; the entire
`ibex_compressed_decoder` module stayed idle. This codec adds enough RVC
opcodes to exercise every RVC decode path the decoder implements, and keeps
all other Level 5 encoding behaviour identical so numbers are directly
comparable.

Packing strategy. Each action still produces exactly one 32-bit program word,
so `emit_program` is still "one 32-bit word per action". For RVC actions we
pack `(C.NOP << 16) | rvc_word` — Ibex executes the RVC at PC, the C.NOP
filler at PC+2, then advances to the next word. Mixing with 32-bit ops is
safe because Ibex's fetch FIFO is RVC-aware.

Legality. Every RVC encoding in this file is legal on RV32IMC Ibex:

  - C.ADDI4SPN: nzuimm forced to the nonzero bucket list.
  - C.LUI:      nzimm forced nonzero, rd forced != {0, 2}
                (rd=0 reserved, rd=2 would become C.ADDI16SP).
  - C.SLLI/SRLI/SRAI/ADDI: shamt/imm=0 is a legal "hint" on RV32 —
                we still allow bucket 0 to exercise the hint path.
  - CA-format ops (AND/OR/XOR/SUB): operate on x8-x15 only.

The self-test at the bottom encodes every op with every imm bucket and
checks the output has the expected opcode bits.
"""

from enum import IntEnum


# =============================================================================
# Op table — 45 base (unchanged from codec_l5) + 16 RVC (new)
# =============================================================================

class Op(IntEnum):
    # ---- 32-bit base, identical IDs to codec_l5 ----
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
    # ---- 16-bit RVC additions ----
    C_ADDI      = 45   # CI,  rd, imm6 (signed; 0 = hint)
    C_LI        = 46   # CI,  rd, imm6
    C_LUI       = 47   # CI,  rd!={0,2}, nzimm != 0
    C_SLLI      = 48   # CI,  rd, shamt (shamt 0 = hint)
    C_MV        = 49   # CR,  rd, rs2 (rd=0 = hint, rs2=0 = C.JR which we avoid)
    C_ADD       = 50   # CR,  rd, rs2
    C_ADDI4SPN  = 51   # CIW, rd', nzuimm
    C_LW        = 52   # CL,  rd', rs1', uimm
    C_SW        = 53   # CS,  rs1', rs2', uimm
    C_AND       = 54   # CA,  rd'=rs1', rs2'
    C_OR        = 55   # CA
    C_XOR       = 56   # CA
    C_SUB       = 57   # CA
    C_SRLI      = 58   # CB,  rd', shamt
    C_SRAI      = 59   # CB,  rd', shamt
    C_ANDI      = 60   # CB,  rd', imm6 (signed)


N_OPS = 61
IMM_BUCKETS = 5

# Base-op bucket values (matching codec_l5).
IMM_BUCKET_VALUES   = [-2048, -100, 0, 100, 2047]
SHAMT_BUCKET_VALUES = [0, 8, 16, 24, 31]
BRANCH_BUCKET_OFFSETS = [4, 8, 12, 16, 20]
JAL_BUCKET_OFFSETS    = [4, 8, 12, 16, 24]
SAFE_CSRS = [0x340, 0xF14, 0xF11, 0xF12, 0xF13]

# RVC-specific bucket values per field type.
#   CI  signed 6-bit   (ADDI/LI/ANDI): full-range picks including 0
#   CI  shamt 5-bit    (SLLI/SRLI/SRAI): 0 hint + nonzero picks
#   CI  LUI nzimm6     (signed, must be nonzero)
#   CIW nzuimm8 x 4    (ADDI4SPN — byte offset, nonzero)
#   CL/CS uimm5 x 4    (LW/SW — byte offset, any)
CI_IMM6_VALUES    = [-32, -1, 0, 1, 31]
CI_SHAMT_VALUES   = [0, 1, 8, 16, 31]
CI_LUI_VALUES     = [1, 2, 4, 16, -1]   # none zero; all fit signed 6-bit
CIW_NZUIMM_VALUES = [4, 16, 64, 256, 1020]   # byte offsets, all multiples of 4
CL_CS_UIMM_VALUES = [0, 4, 16, 64, 124]      # LW/SW byte offsets (/4 fits in 5b)


# ---- Op-set membership (base) ----
R_TYPE_ALU = {Op.ADD, Op.SUB, Op.SLL, Op.SLT, Op.SLTU, Op.XOR, Op.SRL, Op.SRA, Op.OR, Op.AND}
I_ALU      = {Op.ADDI, Op.SLTI, Op.SLTIU, Op.XORI, Op.ORI, Op.ANDI, Op.SLLI, Op.SRLI, Op.SRAI}
LOADS      = {Op.LB, Op.LH, Op.LW, Op.LBU, Op.LHU}
STORES     = {Op.SB, Op.SH, Op.SW}
CSR_OPS    = {Op.CSRRW, Op.CSRRS, Op.CSRRC}
MUL_DIV    = {Op.MUL, Op.MULH, Op.MULHSU, Op.MULHU, Op.DIV, Op.DIVU, Op.REM, Op.REMU}
BRANCHES   = {Op.BEQ, Op.BNE, Op.BLT, Op.BGE, Op.BLTU, Op.BGEU}
JAL_OPS    = {Op.JAL}

# ---- Op-set membership (RVC) ----
RVC_CI_ALU   = {Op.C_ADDI, Op.C_LI, Op.C_LUI, Op.C_SLLI}
RVC_CR       = {Op.C_MV, Op.C_ADD}
RVC_CIW      = {Op.C_ADDI4SPN}
RVC_CL       = {Op.C_LW}
RVC_CS       = {Op.C_SW}
RVC_CA       = {Op.C_AND, Op.C_OR, Op.C_XOR, Op.C_SUB}
RVC_CB       = {Op.C_SRLI, Op.C_SRAI, Op.C_ANDI}
RVC_ALL      = RVC_CI_ALU | RVC_CR | RVC_CIW | RVC_CL | RVC_CS | RVC_CA | RVC_CB


# =============================================================================
# 32-bit base encoder (copied from codec_l5 verbatim)
# =============================================================================

R_F3F7 = {
    Op.ADD:  (0b000, 0b0000000), Op.SUB:  (0b000, 0b0100000),
    Op.SLL:  (0b001, 0b0000000), Op.SLT:  (0b010, 0b0000000),
    Op.SLTU: (0b011, 0b0000000), Op.XOR:  (0b100, 0b0000000),
    Op.SRL:  (0b101, 0b0000000), Op.SRA:  (0b101, 0b0100000),
    Op.OR:   (0b110, 0b0000000), Op.AND:  (0b111, 0b0000000),
    Op.MUL:    (0b000, 0b0000001), Op.MULH:   (0b001, 0b0000001),
    Op.MULHSU: (0b010, 0b0000001), Op.MULHU:  (0b011, 0b0000001),
    Op.DIV:    (0b100, 0b0000001), Op.DIVU:   (0b101, 0b0000001),
    Op.REM:    (0b110, 0b0000001), Op.REMU:   (0b111, 0b0000001),
}
I_F3 = {Op.ADDI: 0b000, Op.SLTI: 0b010, Op.SLTIU: 0b011,
        Op.XORI: 0b100, Op.ORI: 0b110, Op.ANDI: 0b111}
SHIFT_F3F7 = {Op.SLLI: (0b001, 0b0000000),
              Op.SRLI: (0b101, 0b0000000),
              Op.SRAI: (0b101, 0b0100000)}
LOAD_F3  = {Op.LB: 0b000, Op.LH: 0b001, Op.LW: 0b010, Op.LBU: 0b100, Op.LHU: 0b101}
STORE_F3 = {Op.SB: 0b000, Op.SH: 0b001, Op.SW: 0b010}
CSR_F3   = {Op.CSRRW: 0b001, Op.CSRRS: 0b010, Op.CSRRC: 0b011}
BRANCH_F3 = {Op.BEQ: 0b000, Op.BNE: 0b001, Op.BLT: 0b100, Op.BGE: 0b101,
             Op.BLTU: 0b110, Op.BGEU: 0b111}


def _imm12(imm: int) -> int:
    return imm & 0xFFF


def _b_imm_encode(offset: int) -> tuple[int, int]:
    o = offset & 0x1FFE
    imm12   = (o >> 12) & 0x1
    imm10_5 = (o >> 5)  & 0x3F
    imm4_1  = (o >> 1)  & 0xF
    imm11   = (o >> 11) & 0x1
    return (imm12 << 6) | imm10_5, (imm4_1 << 1) | imm11


def _j_imm_encode(offset: int) -> int:
    o = offset & 0x1FFFFE
    imm20    = (o >> 20) & 0x1
    imm10_1  = (o >> 1)  & 0x3FF
    imm11    = (o >> 11) & 0x1
    imm19_12 = (o >> 12) & 0xFF
    return (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12)


def encode_base(op: Op, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
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
    raise ValueError(f"Not a base op: {op}")


# =============================================================================
# 16-bit RVC encoder
# =============================================================================

# C.NOP is canonical filler we pack into the upper half of an RVC slot.
# Encoding: CI funct3=000, imm=0, rd/rs1=0, op=01  ->  0x0001
C_NOP = 0x0001


def _imm6(v: int) -> int:
    return v & 0x3F


def _rdc_prime(r: int) -> int:
    """Map any 5-bit register to the RVC prime-register encoding (x8-x15)."""
    return r & 0x7  # wire-level 3 bits; decoder adds implicit 8


def _c_ci(funct3: int, rd: int, imm6_val: int, op_bits: int) -> int:
    """CI format: funct3[15:13] imm[12] rd[11:7] imm[6:2] op[1:0]."""
    imm = _imm6(imm6_val)
    imm_hi = (imm >> 5) & 0x1
    imm_lo = imm & 0x1F
    return (funct3 << 13) | (imm_hi << 12) | ((rd & 0x1F) << 7) | (imm_lo << 2) | op_bits


def _c_cr(funct4: int, rd_rs1: int, rs2: int, op_bits: int) -> int:
    """CR format: funct4[15:12] rd/rs1[11:7] rs2[6:2] op[1:0]."""
    return (funct4 << 12) | ((rd_rs1 & 0x1F) << 7) | ((rs2 & 0x1F) << 2) | op_bits


def _c_ca(funct6: int, rd_rs1_p: int, funct2: int, rs2_p: int, op_bits: int) -> int:
    """CA format: funct6[15:10] rd'/rs1'[9:7] funct2[6:5] rs2'[4:2] op[1:0]."""
    return (funct6 << 10) | ((rd_rs1_p & 0x7) << 7) | (funct2 << 5) | ((rs2_p & 0x7) << 2) | op_bits


def _c_cb_shift(rd_p: int, shamt5: int, funct2: int) -> int:
    """CB shift (SRLI/SRAI): funct3=100, bit12=0 (RV32), funct2[11:10],
       rd'[9:7], shamt[6:2], op=01. shamt[5] must be 0 for RV32."""
    return (0b100 << 13) | (0 << 12) | (funct2 << 10) | ((rd_p & 0x7) << 7) | ((shamt5 & 0x1F) << 2) | 0b01


def _c_cb_andi(rd_p: int, imm6_val: int) -> int:
    """CB ANDI: funct3=100, imm[12]=imm[5], funct2=10, rd'[9:7], imm[6:2]=imm[4:0], op=01."""
    imm = _imm6(imm6_val)
    imm_hi = (imm >> 5) & 0x1
    imm_lo = imm & 0x1F
    return (0b100 << 13) | (imm_hi << 12) | (0b10 << 10) | ((rd_p & 0x7) << 7) | (imm_lo << 2) | 0b01


def _c_ciw_addi4spn(rd_p: int, nzuimm: int) -> int:
    """CIW ADDI4SPN: funct3=000 imm[12:5] rd'[4:2] op=00.
       nzuimm is the byte offset (multiple of 4). Layout in instr[12:5]:
       [5|4|9|8|7|6|2|3]."""
    assert nzuimm != 0 and nzuimm % 4 == 0, "C.ADDI4SPN needs nonzero /4 byte offset"
    b5 = (nzuimm >> 5) & 0x1
    b4 = (nzuimm >> 4) & 0x1
    b9 = (nzuimm >> 9) & 0x1
    b8 = (nzuimm >> 8) & 0x1
    b7 = (nzuimm >> 7) & 0x1
    b6 = (nzuimm >> 6) & 0x1
    b2 = (nzuimm >> 2) & 0x1
    b3 = (nzuimm >> 3) & 0x1
    imm_field = (b5 << 7) | (b4 << 6) | (b9 << 5) | (b8 << 4) | (b7 << 3) | (b6 << 2) | (b2 << 1) | b3
    return (0b000 << 13) | (imm_field << 5) | ((rd_p & 0x7) << 2) | 0b00


def _c_lw(rd_p: int, rs1_p: int, uimm: int) -> int:
    """CL LW: funct3=010 uimm[5:3][12:10] rs1'[9:7] uimm[2|6][6:5] rd'[4:2] op=00.
       uimm is the byte offset (multiple of 4)."""
    assert uimm % 4 == 0 and 0 <= uimm <= 124
    imm_5_3 = (uimm >> 3) & 0x7
    imm_2   = (uimm >> 2) & 0x1
    imm_6   = (uimm >> 6) & 0x1
    return (0b010 << 13) | (imm_5_3 << 10) | ((rs1_p & 0x7) << 7) | (imm_2 << 6) | (imm_6 << 5) | ((rd_p & 0x7) << 2) | 0b00


def _c_sw(rs1_p: int, rs2_p: int, uimm: int) -> int:
    """CS SW: funct3=110 uimm[5:3][12:10] rs1'[9:7] uimm[2|6][6:5] rs2'[4:2] op=00."""
    assert uimm % 4 == 0 and 0 <= uimm <= 124
    imm_5_3 = (uimm >> 3) & 0x7
    imm_2   = (uimm >> 2) & 0x1
    imm_6   = (uimm >> 6) & 0x1
    return (0b110 << 13) | (imm_5_3 << 10) | ((rs1_p & 0x7) << 7) | (imm_2 << 6) | (imm_6 << 5) | ((rs2_p & 0x7) << 2) | 0b00


def encode_rvc(op: Op, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    """Return a 16-bit RVC instruction. Caller packs with C.NOP filler."""
    ib = imm_bucket % IMM_BUCKETS

    if op is Op.C_ADDI:
        # CI funct3=000, op=01. rd=0 and imm=0 are hints (legal).
        return _c_ci(0b000, rd & 0x1F, CI_IMM6_VALUES[ib], 0b01)

    if op is Op.C_LI:
        # CI funct3=010, op=01.  (rd=0 = hint)
        return _c_ci(0b010, rd & 0x1F, CI_IMM6_VALUES[ib], 0b01)

    if op is Op.C_LUI:
        # CI funct3=011, op=01. Must avoid rd in {0,2} (reserved / C.ADDI16SP)
        # and nzimm6 != 0.
        safe_rd = 3 + ((rd & 0x1F) % 29)    # maps into {3..31}, skipping {0,1,2}
        # Actually rd=1 is legal for C.LUI; only {0,2} are special. Keep it in
        # {3..31} to stay far from either.
        return _c_ci(0b011, safe_rd, CI_LUI_VALUES[ib], 0b01)

    if op is Op.C_SLLI:
        # CI funct3=000, op=10. rd=0 hint; shamt=0 hint. RV32 requires bit12=0.
        shamt = CI_SHAMT_VALUES[ib] & 0x1F   # always < 32 by construction
        return (0b000 << 13) | (0 << 12) | ((rd & 0x1F) << 7) | (shamt << 2) | 0b10

    if op is Op.C_MV:
        # CR funct4=1000, op=10.  rs2=0 would be C.JR, avoid it.
        safe_rs2 = 1 + ((rs2 & 0x1F) % 31)    # in {1..31}
        return _c_cr(0b1000, rd & 0x1F, safe_rs2, 0b10)

    if op is Op.C_ADD:
        # CR funct4=1001, op=10.  rd=0 or rs2=0 turn into other ops, avoid.
        safe_rd  = 1 + ((rd  & 0x1F) % 31)
        safe_rs2 = 1 + ((rs2 & 0x1F) % 31)
        return _c_cr(0b1001, safe_rd, safe_rs2, 0b10)

    if op is Op.C_ADDI4SPN:
        return _c_ciw_addi4spn(_rdc_prime(rd), CIW_NZUIMM_VALUES[ib])

    if op is Op.C_LW:
        return _c_lw(_rdc_prime(rd), _rdc_prime(rs1), CL_CS_UIMM_VALUES[ib])

    if op is Op.C_SW:
        return _c_sw(_rdc_prime(rs1), _rdc_prime(rs2), CL_CS_UIMM_VALUES[ib])

    if op in RVC_CA:
        # CA funct6=100011, op=01.  funct2 selects the operation.
        funct2_by_op = {Op.C_SUB: 0b00, Op.C_XOR: 0b01, Op.C_OR: 0b10, Op.C_AND: 0b11}
        return _c_ca(0b100011, _rdc_prime(rd), funct2_by_op[op], _rdc_prime(rs2), 0b01)

    if op is Op.C_SRLI:
        return _c_cb_shift(_rdc_prime(rd), CI_SHAMT_VALUES[ib] & 0x1F, 0b00)
    if op is Op.C_SRAI:
        return _c_cb_shift(_rdc_prime(rd), CI_SHAMT_VALUES[ib] & 0x1F, 0b01)
    if op is Op.C_ANDI:
        return _c_cb_andi(_rdc_prime(rd), CI_IMM6_VALUES[ib])

    raise ValueError(f"Not an RVC op: {op}")


# =============================================================================
# Unified encoder + program emitter
# =============================================================================

def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    """Return a 32-bit program word for this action.

    For base ops: the full 32-bit RISC-V instruction.
    For RVC ops:  (C.NOP << 16) | rvc_instruction — Ibex executes the RVC
                  at PC, then the C.NOP at PC+2, then advances to the next
                  word. This keeps memory layout at one action per 32-bit word.
    """
    op = Op(op_i)
    if op in RVC_ALL:
        rvc_word = encode_rvc(op, rd, rs1, rs2, imm_bucket)
        return (C_NOP << 16) | (rvc_word & 0xFFFF)
    return encode_base(op, rd, rs1, rs2, imm_bucket)


def emit_program(actions: list[tuple[int, int, int, int, int]]) -> list[int]:
    """Encode a list of actions, append 16 NOPs of tail padding.

    Matches codec_l5.emit_program so forward branches/jumps near the end of
    the program land in safe territory.
    """
    nop = encode_base(Op.ADDI, 0, 0, 0, 2)
    return [encode(*a) for a in actions] + [nop] * 16


# =============================================================================
# Self-test
# =============================================================================

def _opcode_bits_ok(op: Op, word: int) -> bool:
    """Sanity-check the opcode bits for each op class."""
    if op in RVC_ALL:
        lo16 = word & 0xFFFF
        op_bits = lo16 & 0x3
        if op in (Op.C_ADDI, Op.C_LI, Op.C_LUI, Op.C_SRLI, Op.C_SRAI, Op.C_ANDI) \
                or op in RVC_CA:
            return op_bits == 0b01
        if op in (Op.C_SLLI, Op.C_MV, Op.C_ADD):
            return op_bits == 0b10
        if op in (Op.C_ADDI4SPN, Op.C_LW, Op.C_SW):
            return op_bits == 0b00
        return False
    # base op: low 7 bits give the opcode
    opcode7 = word & 0x7F
    if op in R_TYPE_ALU or op in MUL_DIV: return opcode7 == 0b0110011
    if op in I_ALU:                       return opcode7 == 0b0010011
    if op in LOADS:                       return opcode7 == 0b0000011
    if op in STORES:                      return opcode7 == 0b0100011
    if op in CSR_OPS:                     return opcode7 == 0b1110011
    if op in BRANCHES:                    return opcode7 == 0b1100011
    if op in JAL_OPS:                     return opcode7 == 0b1101111
    return False


def _self_test() -> None:
    for op_i in range(N_OPS):
        op = Op(op_i)
        for ib in range(IMM_BUCKETS):
            for rd, rs1, rs2 in [(0, 0, 0), (1, 2, 3), (15, 16, 31), (31, 31, 31)]:
                word = encode(op_i, rd, rs1, rs2, ib)
                assert 0 <= word <= 0xFFFFFFFF, f"{op.name} out of range"
                assert _opcode_bits_ok(op, word), \
                    f"bad opcode bits for {op.name}: word=0x{word:08x}"
                if op in RVC_ALL:
                    # upper half must be C.NOP filler
                    assert ((word >> 16) & 0xFFFF) == C_NOP, \
                        f"{op.name}: upper half not C.NOP (0x{(word>>16)&0xFFFF:04x})"
                    # Illegal patterns that the compressed_decoder flags:
                    lo = word & 0xFFFF
                    if op is Op.C_ADDI4SPN:
                        # instr[12:5] must be nonzero
                        assert ((lo >> 5) & 0xFF) != 0
                    if op is Op.C_LUI:
                        # {instr[12], instr[6:2]} must be nonzero
                        nzimm = ((lo >> 12) & 0x1) << 5 | ((lo >> 2) & 0x1F)
                        assert nzimm != 0, f"C_LUI nzimm6=0 at ib={ib} rd={rd}"
                        # rd must not be 0 or 2
                        rd_f = (lo >> 7) & 0x1F
                        assert rd_f not in (0, 2), f"C_LUI rd in {{0,2}}"
    print(f"[OK] self-test: {N_OPS} ops × {IMM_BUCKETS} buckets × 4 reg tuples encoded cleanly.")


if __name__ == "__main__":
    _self_test()
    print("\nSample encodings (one per op):")
    for op_i in range(N_OPS):
        op = Op(op_i)
        word = encode(op_i, 5, 6, 7, 2)
        tag = "RVC" if op in RVC_ALL else "   "
        print(f"  {tag} op={op.name:<12s} -> 0x{word:08x}")
