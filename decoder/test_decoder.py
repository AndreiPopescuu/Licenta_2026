"""
Ibex RISC-V Decoder Demo
========================
This script drives the Ibex instruction decoder with test stimuli and
tracks which coverage bins get hit. It demonstrates the full verification
loop: generate stimulus -> simulate -> measure coverage.

The decoder is a purely combinational circuit (no clock needed). We feed
it a 32-bit instruction and instantly read what operation it decoded.

Technology stack:
  - ibex_decoder.sv   = the chip's blueprint (SystemVerilog, we don't touch it)
  - Verilator          = compiles the .sv into a software simulation
  - cocotb             = lets us drive the simulation from Python
  - this file          = our test logic (pure Python)
"""

import cocotb
from cocotb.triggers import Timer
import random

# ---------------------------------------------------------------------------
# ALU operation codes (from ibex_pkg.sv)
# These are the integer values the decoder outputs to tell the ALU what to do
# ---------------------------------------------------------------------------
ALU_OPS = {
    0: "ADD",   1: "SUB",   2: "XOR",   3: "OR",    4: "AND",
    8: "SRA",   9: "SRL",  10: "SLL",  25: "LT",   26: "LTU",
   43: "SLT",  44: "SLTU",
}

# Operand source selectors
OP_A_REG_A = 0
OP_B_REG_B = 0
OP_B_IMM = 1
RF_WD_EX = 0


# ---------------------------------------------------------------------------
# RISC-V instruction encoders
# These functions build valid 32-bit instructions from human-readable parts
# ---------------------------------------------------------------------------

def encode_r_type(funct7, rs2, rs1, funct3, rd):
    """R-type: register-register operations (ADD, SUB, AND, OR, XOR, shifts)"""
    opcode = 0b0110011
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

def encode_i_type(imm12, rs1, funct3, rd):
    """I-type: register-immediate operations (ADDI, ANDI, ORI, etc.)"""
    opcode = 0b0010011
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

def encode_load(imm12, rs1, funct3, rd):
    """Load: read from memory (LW, LH, LB)"""
    opcode = 0b0000011
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

def encode_store(imm12, rs2, rs1, funct3):
    """Store: write to memory (SW, SH, SB)"""
    opcode = 0b0100011
    imm_hi = (imm12 >> 5) & 0x7F
    imm_lo = imm12 & 0x1F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode


# ---------------------------------------------------------------------------
# Coverage database
# Tracks which (operation, register) combinations have been exercised
# ---------------------------------------------------------------------------

class CoverageDB:
    def __init__(self):
        # Type 1: which operations have we seen?
        self.ops_seen = set()
        # Type 2: which register ports have been used?
        self.read_a_regs = set()    # 32 possible
        self.read_b_regs = set()    # 32 possible
        self.write_regs = set()     # 32 possible
        # Type 3: cross-coverage (operation x register)
        self.cross = set()          # (op, port, reg) tuples
        # Stats
        self.total_stimuli = 0
        self.illegal_count = 0

    def record(self, op_name, rd, rs1, rs2, has_rs2):
        """Record a single decoded instruction's coverage."""
        self.ops_seen.add(op_name)
        if rs1 is not None:
            self.read_a_regs.add(rs1)
            self.cross.add((op_name, 'rA', rs1))
        if has_rs2 and rs2 is not None:
            self.read_b_regs.add(rs2)
            self.cross.add((op_name, 'rB', rs2))
        if rd is not None:
            self.write_regs.add(rd)
            self.cross.add((op_name, 'wr', rd))

    @property
    def total_bins(self):
        # 26 ops + 96 reg ports + ~1985 cross = ~2107
        # We'll compute our actual maximums dynamically
        return len(self.ops_seen) + len(self.read_a_regs) + len(self.read_b_regs) + len(self.write_regs) + len(self.cross)

    def report(self):
        print("\n" + "=" * 60)
        print("COVERAGE REPORT")
        print("=" * 60)
        print(f"Total stimuli applied:  {self.total_stimuli}")
        print(f"Illegal instructions:   {self.illegal_count}")
        print(f"Valid instructions:      {self.total_stimuli - self.illegal_count}")
        print()
        print(f"Operations covered:     {len(self.ops_seen)}")
        print(f"  {sorted(self.ops_seen)}")
        print(f"Read-A registers used:  {len(self.read_a_regs)} / 32")
        print(f"Read-B registers used:  {len(self.read_b_regs)} / 32")
        print(f"Write registers used:   {len(self.write_regs)} / 32")
        print(f"Cross-coverage bins:    {len(self.cross)}")
        print(f"Total bins covered:     {self.total_bins}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------

@cocotb.test()
async def run_demo(dut):
    """
    Demonstrate three approaches to testing the Ibex decoder:
    1. Hand-crafted targeted stimuli (a human wrote these)
    2. Pure random stimuli (the baseline)
    3. Smart random (understands the encoding, targets gaps)
    """
    coverage = CoverageDB()

    async def apply_and_sample(instruction, label=""):
        """Feed one instruction to the decoder, read what comes out."""
        dut.insn_i.value = instruction
        await Timer(1, units='ns')

        coverage.total_stimuli += 1

        # Read decoder outputs
        illegal = int(dut.u_decoder.illegal_insn_o.value)
        if illegal:
            coverage.illegal_count += 1
            if label:
                print(f"  {label}: 0x{instruction:08X} -> ILLEGAL")
            return

        alu_op_val = int(dut.u_decoder.alu_operator_o.value)
        rf_we = int(dut.u_decoder.rf_we_o.value)
        rf_ren_a = int(dut.u_decoder.rf_ren_a_o.value)
        rf_ren_b = int(dut.u_decoder.rf_ren_b_o.value)
        rf_raddr_a = int(dut.u_decoder.rf_raddr_a_o.value)
        rf_raddr_b = int(dut.u_decoder.rf_raddr_b_o.value)
        rf_waddr = int(dut.u_decoder.rf_waddr_o.value)
        data_req = int(dut.u_decoder.data_req_o.value)
        data_we = int(dut.u_decoder.data_we_o.value)
        data_type = int(dut.u_decoder.data_type_o.value)
        op_b_mux = int(dut.u_decoder.alu_op_b_mux_sel_o.value)
        op_a_mux = int(dut.u_decoder.alu_op_a_mux_sel_o.value)
        rf_wdata_sel = int(dut.u_decoder.rf_wdata_sel_o.value)
        mult_sel = int(dut.u_decoder.mult_sel_o.value)
        div_sel = int(dut.u_decoder.div_sel_o.value)

        # Determine what kind of operation this is
        op_name = None
        has_rs2 = False
        rd = rf_waddr if rf_we else None
        rs1 = rf_raddr_a if rf_ren_a else None
        rs2 = rf_raddr_b if rf_ren_b else None

        if data_req:
            # Memory operation
            size_names = {0: "LW", 1: "LH", 2: "LB"} if not data_we else {0: "SW", 1: "SH", 2: "SB"}
            op_name = size_names.get(data_type, f"MEM_{data_type}")
            has_rs2 = bool(data_we)  # stores use rs2
            if not data_we:
                # loads write to rd
                rd = rf_waddr
        elif rf_we and not mult_sel and not div_sel and rf_ren_a and op_a_mux == OP_A_REG_A and rf_wdata_sel == RF_WD_EX:
            # ALU operation
            alu_name = ALU_OPS.get(alu_op_val, f"ALU_{alu_op_val}")
            if op_b_mux == OP_B_IMM:
                op_name = f"{alu_name}I"  # immediate variant
            else:
                op_name = alu_name
                has_rs2 = True

        if op_name:
            coverage.record(op_name, rd, rs1, rs2, has_rs2)

        if label:
            reg_info = f"rs1=x{rs1}" if rs1 is not None else ""
            if has_rs2 and rs2 is not None:
                reg_info += f", rs2=x{rs2}"
            if rd is not None:
                reg_info += f", rd=x{rd}"
            print(f"  {label}: 0x{instruction:08X} -> {op_name or '???'} ({reg_info})")

    # ===================================================================
    # PHASE 1: Hand-crafted stimuli (what a human engineer would write)
    # ===================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Hand-crafted targeted stimuli")
    print("=" * 60)
    print("These are instructions a human deliberately constructed")
    print("to cover specific operations and registers.\n")

    # ADD x1, x2, x3  (add reg2 and reg3, store in reg1)
    await apply_and_sample(encode_r_type(0b0000000, 3, 2, 0b000, 1), "ADD x1,x2,x3")

    # SUB x4, x5, x6
    await apply_and_sample(encode_r_type(0b0100000, 6, 5, 0b000, 4), "SUB x4,x5,x6")

    # AND x7, x8, x9
    await apply_and_sample(encode_r_type(0b0000000, 9, 8, 0b111, 7), "AND x7,x8,x9")

    # OR x10, x11, x12
    await apply_and_sample(encode_r_type(0b0000000, 12, 11, 0b110, 10), "OR x10,x11,x12")

    # XOR x13, x14, x15
    await apply_and_sample(encode_r_type(0b0000000, 15, 14, 0b100, 13), "XOR x13,x14,x15")

    # SLL x16, x17, x18  (shift left logical)
    await apply_and_sample(encode_r_type(0b0000000, 18, 17, 0b001, 16), "SLL x16,x17,x18")

    # SRL x19, x20, x21  (shift right logical)
    await apply_and_sample(encode_r_type(0b0000000, 21, 20, 0b101, 19), "SRL x19,x20,x21")

    # SRA x22, x23, x24  (shift right arithmetic)
    await apply_and_sample(encode_r_type(0b0100000, 24, 23, 0b101, 22), "SRA x22,x23,x24")

    # SLT x25, x26, x27  (set less than)
    await apply_and_sample(encode_r_type(0b0000000, 27, 26, 0b010, 25), "SLT x25,x26,x27")

    # SLTU x28, x29, x30  (set less than unsigned)
    await apply_and_sample(encode_r_type(0b0000000, 30, 29, 0b011, 28), "SLTU x28,x29,x30")

    # ADDI x31, x0, 42  (load immediate 42 into x31)
    await apply_and_sample(encode_i_type(42, 0, 0b000, 31), "ADDI x31,x0,42")

    # LW x1, 0(x2)  (load word from memory)
    await apply_and_sample(encode_load(0, 2, 0b010, 1), "LW x1,0(x2)")

    # SW x3, 0(x4)  (store word to memory)
    await apply_and_sample(encode_store(0, 3, 4, 0b010), "SW x3,0(x4)")

    # LH x5, 0(x6)  (load halfword)
    await apply_and_sample(encode_load(0, 6, 0b001, 5), "LH x5,0(x6)")

    # SB x7, 0(x8)  (store byte)
    await apply_and_sample(encode_store(0, 7, 8, 0b000), "SB x7,0(x8)")

    coverage.report()

    # ===================================================================
    # PHASE 2: Pure random stimuli (the baseline AI must beat)
    # ===================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Pure random 32-bit values (100 stimuli)")
    print("=" * 60)
    print("Most random 32-bit values are NOT valid RISC-V instructions.")
    print("This demonstrates why random testing is inefficient.\n")

    bins_before = coverage.total_bins
    random.seed(42)
    for i in range(100):
        await apply_and_sample(random.randint(0, 0xFFFFFFFF))

    bins_after = coverage.total_bins
    print(f"\n  100 random stimuli added {bins_after - bins_before} new coverage bins")
    coverage.report()

    # ===================================================================
    # PHASE 3: Smart random (understands encoding, targets gaps)
    # ===================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Smart random (encoding-aware, 100 stimuli)")
    print("=" * 60)
    print("Generates valid R-type and I-type instructions with")
    print("systematically varied registers and operations.\n")

    bins_before = coverage.total_bins

    # R-type operations: (funct7, funct3) pairs
    r_type_ops = [
        (0b0000000, 0b000, "ADD"),
        (0b0100000, 0b000, "SUB"),
        (0b0000000, 0b111, "AND"),
        (0b0000000, 0b110, "OR"),
        (0b0000000, 0b100, "XOR"),
        (0b0000000, 0b001, "SLL"),
        (0b0000000, 0b101, "SRL"),
        (0b0100000, 0b101, "SRA"),
        (0b0000000, 0b010, "SLT"),
        (0b0000000, 0b011, "SLTU"),
    ]

    # I-type operations: funct3 values
    i_type_ops = [
        (0b000, "ADDI"),
        (0b111, "ANDI"),
        (0b110, "ORI"),
        (0b100, "XORI"),
        (0b010, "SLTI"),
        (0b011, "SLTIU"),
    ]

    count = 0
    # Sweep R-type: every operation with varied registers
    for funct7, funct3, name in r_type_ops:
        for _ in range(5):
            rd = random.randint(0, 31)
            rs1 = random.randint(0, 31)
            rs2 = random.randint(0, 31)
            insn = encode_r_type(funct7, rs2, rs1, funct3, rd)
            await apply_and_sample(insn)
            count += 1
            if count >= 100:
                break
        if count >= 100:
            break

    # Fill remaining budget with I-type
    while count < 100:
        funct3, name = random.choice(i_type_ops)
        rd = random.randint(0, 31)
        rs1 = random.randint(0, 31)
        imm = random.randint(-2048, 2047)
        insn = encode_i_type(imm, rs1, funct3, rd)
        await apply_and_sample(insn)
        count += 1

    bins_after = coverage.total_bins
    print(f"  100 smart-random stimuli added {bins_after - bins_before} new coverage bins")
    coverage.report()

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total stimuli:          {coverage.total_stimuli}")
    print(f"Total illegal:          {coverage.illegal_count}")
    print(f"Total coverage bins:    {coverage.total_bins}")
    print()
    print("This is what the AI would replace: instead of hand-crafted")
    print("or random stimuli, an LLM or RL agent proposes instructions")
    print("that maximize coverage per stimulus. The simulator and")
    print("coverage tracking stay exactly the same.")
    print("=" * 60)
