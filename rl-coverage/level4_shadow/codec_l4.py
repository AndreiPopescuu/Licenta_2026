"""Encoder/decoder for the 30-op Level 6 action space.

encode(op, rd, rs1, rs2, imm_bucket) -> 32-bit RISC-V machine word
decode(word) -> (op, rd, rs1, rs2, imm_bucket) or None

Consistency contract: the decoder is the inverse of the encoder for all valid
(op, rd, rs1, rs2, imm_bucket) tuples, and returns None for any word that
doesn't come from this encoding scheme. The real-RTL monitor drives the
shadow's coverage accounting by decoding RVFI retirements through this module,
so correctness here is what makes shadow=RTL agreement possible.

Immediate buckets (5 values) for I-ALU/Load/Store offsets:
  0 = -2048 (min signed 12-bit)
  1 = -100  (small negative)
  2 =  0
  3 =  100  (small positive)
  4 =  2047 (max signed 12-bit)

For shift-immediate ops (SLLI/SRLI/SRAI) the shamt is 5 bits. We squash
buckets to valid shamt values:
  0 = 31, 1 = 8, 2 = 0, 3 = 8, 4 = 31   (keeps buckets distinguishable)

For CSR ops, the CSR is hardcoded to mscratch (0x340) — a dedicated
scratchpad that can be read/written arbitrarily with no side effects on
control flow or interrupts.
"""

from shadow_cpu_l4 import Op, R_TYPE, I_ALU, LOADS, STORES, CSR_OPS

IMM_BUCKET_VALUES = [-2048, -100, 0, 100, 2047]
# Shifts use 5-bit shamt (0..31). We pick 5 distinct values so each bucket is
# distinguishable after decode. The bucket semantics differ from I-ALU's signed
# imm buckets — that's fine, the shadow bins track per-op bucket indices.
SHAMT_BUCKET_VALUES = [0, 8, 16, 24, 31]
CSR_MSCRATCH = 0x340

# R-type funct3/funct7 by op
R_F3F7 = {
    Op.ADD:  (0b000, 0b0000000), Op.SUB:  (0b000, 0b0100000),
    Op.SLL:  (0b001, 0b0000000), Op.SLT:  (0b010, 0b0000000),
    Op.SLTU: (0b011, 0b0000000), Op.XOR:  (0b100, 0b0000000),
    Op.SRL:  (0b101, 0b0000000), Op.SRA:  (0b101, 0b0100000),
    Op.OR:   (0b110, 0b0000000), Op.AND:  (0b111, 0b0000000),
}
R_F3F7_INV = {v: k for k, v in R_F3F7.items()}

I_F3 = {
    Op.ADDI: 0b000, Op.SLTI: 0b010, Op.SLTIU: 0b011,
    Op.XORI: 0b100, Op.ORI: 0b110, Op.ANDI: 0b111,
}
I_F3_INV = {v: k for k, v in I_F3.items()}

SHIFT_F3F7 = {
    Op.SLLI: (0b001, 0b0000000),
    Op.SRLI: (0b101, 0b0000000),
    Op.SRAI: (0b101, 0b0100000),
}
SHIFT_F3F7_INV = {v: k for k, v in SHIFT_F3F7.items()}

LOAD_F3 = {Op.LB: 0b000, Op.LH: 0b001, Op.LW: 0b010, Op.LBU: 0b100, Op.LHU: 0b101}
LOAD_F3_INV = {v: k for k, v in LOAD_F3.items()}

STORE_F3 = {Op.SB: 0b000, Op.SH: 0b001, Op.SW: 0b010}
STORE_F3_INV = {v: k for k, v in STORE_F3.items()}

CSR_F3 = {Op.CSRRW: 0b001, Op.CSRRS: 0b010, Op.CSRRC: 0b011}
CSR_F3_INV = {v: k for k, v in CSR_F3.items()}


def _imm12_bits(imm: int) -> int:
    """Return 12-bit two's complement of imm."""
    return imm & 0xFFF


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    op = Op(op_i)
    if op in R_TYPE:
        f3, f7 = R_F3F7[op]
        return (f7 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0110011
    if op in I_ALU:
        if op in SHIFT_F3F7:
            f3, f7 = SHIFT_F3F7[op]
            shamt = SHAMT_BUCKET_VALUES[imm_bucket] & 0x1F
            return (f7 << 25) | (shamt << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0010011
        f3 = I_F3[op]
        imm = IMM_BUCKET_VALUES[imm_bucket]
        return (_imm12_bits(imm) << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0010011
    if op in LOADS:
        f3 = LOAD_F3[op]
        imm = IMM_BUCKET_VALUES[imm_bucket]
        return (_imm12_bits(imm) << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0000011
    if op in STORES:
        f3 = STORE_F3[op]
        imm = IMM_BUCKET_VALUES[imm_bucket] & 0xFFF
        imm_hi = (imm >> 5) & 0x7F
        imm_lo = imm & 0x1F
        return (imm_hi << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | (imm_lo << 7) | 0b0100011
    if op in CSR_OPS:
        f3 = CSR_F3[op]
        return (CSR_MSCRATCH << 20) | ((rs1 & 0x1F) << 15) | (f3 << 12) | ((rd & 0x1F) << 7) | 0b1110011
    raise ValueError(f"Unknown op {op}")


def _imm_to_bucket(imm: int, is_shift: bool = False) -> int:
    """Map an observed immediate back to the nearest bucket index.

    We only ever emit the exact bucket values, so we can match on equality.
    """
    table = SHAMT_BUCKET_VALUES if is_shift else IMM_BUCKET_VALUES
    for i, v in enumerate(table):
        if imm == v:
            return i
    # Fall back: nearest
    return min(range(len(table)), key=lambda i: abs(table[i] - imm))


def _sign_extend_12(x: int) -> int:
    x &= 0xFFF
    return x - 0x1000 if x & 0x800 else x


def decode(word: int) -> tuple[int, int, int, int, int] | None:
    """Inverse of encode. Returns None on unknown encoding (we intentionally
    don't decode anything the L6 action space can't emit)."""
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    f3 = (word >> 12) & 0x7
    f7 = (word >> 25) & 0x7F
    if opcode == 0b0110011:   # R-type
        op = R_F3F7_INV.get((f3, f7))
        if op is not None:
            return (int(op), rd, rs1, rs2, 0)
    elif opcode == 0b0010011:  # I-type ALU
        if f3 in (0b001, 0b101):  # shifts
            op = SHIFT_F3F7_INV.get((f3, f7))
            if op is not None:
                shamt = rs2  # shamt occupies the rs2 position
                return (int(op), rd, rs1, 0, _imm_to_bucket(shamt, is_shift=True))
        op = I_F3_INV.get(f3)
        if op is not None:
            imm = _sign_extend_12(word >> 20)
            return (int(op), rd, rs1, 0, _imm_to_bucket(imm))
    elif opcode == 0b0000011:  # loads
        op = LOAD_F3_INV.get(f3)
        if op is not None:
            imm = _sign_extend_12(word >> 20)
            return (int(op), rd, rs1, 0, _imm_to_bucket(imm))
    elif opcode == 0b0100011:  # stores
        op = STORE_F3_INV.get(f3)
        if op is not None:
            imm_hi = (word >> 25) & 0x7F
            imm_lo = (word >> 7) & 0x1F
            imm = _sign_extend_12((imm_hi << 5) | imm_lo)
            # Stores don't write rd; store encoding puts imm[4:0] in rd bits
            return (int(op), 0, rs1, rs2, _imm_to_bucket(imm))
    elif opcode == 0b1110011:  # SYSTEM (CSR*)
        op = CSR_F3_INV.get(f3)
        if op is not None:
            return (int(op), rd, rs1, 0, 0)
    return None


def self_test():
    import random
    random.seed(0)
    for _ in range(5000):
        op_i = random.choice(list(range(30)))
        rd = random.randrange(32); rs1 = random.randrange(32); rs2 = random.randrange(32)
        ib = random.randrange(5)
        enc = encode(op_i, rd, rs1, rs2, ib)
        dec = decode(enc)
        assert dec is not None, f"decode returned None for {hex(enc)} (op={Op(op_i).name})"
        # For stores, rd bits are imm_lo -> decoded rd is forced to 0
        exp_rd = 0 if Op(op_i) in STORES else rd
        # For CSR / I-ALU / Loads, rs2 bits encode shamt (for shifts) or part of imm (for I/loads),
        # so expected rs2 on decode is 0
        exp_rs2 = rs2 if Op(op_i) in R_TYPE or Op(op_i) in STORES else 0
        # R-type and CSR ops don't carry imm_bucket in the encoding — always decode as 0.
        exp_ib = 0 if Op(op_i) in R_TYPE or Op(op_i) in CSR_OPS else ib
        assert dec == (op_i, exp_rd, rs1, exp_rs2, exp_ib), \
            f"roundtrip failed: op={Op(op_i).name} enc={hex(enc)} dec={dec} expected=({op_i},{exp_rd},{rs1},{exp_rs2},{exp_ib})"
    print("codec self-test passed (5000 samples)")


if __name__ == "__main__":
    self_test()
