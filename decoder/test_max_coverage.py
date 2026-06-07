"""
Maximum Coverage Test for the Ibex RISC-V Decoder
==================================================
This script systematically generates the minimum set of instructions needed
to hit every reachable coverage bin in the Ibex decoder.

Strategy:
  For each operation, sweep all 32 registers across every port that
  operation uses. One instruction per register value, with the same
  register used for all ports (rd=rs1=rs2=i), covering 3 cross-bins
  per instruction for R-type, or 2 for I-type/load/store.

  Total: 10 R-type ops x 32 + 9 I-type ops x 32 + 3 loads x 32
         + 3 stores x 32 = 800 instructions to cover ~2041 bins.
"""

import cocotb
from cocotb.triggers import Timer

# ── ALU operation codes from ibex_pkg.sv ──
ALU_OPS = {
    0: "ADD",   1: "SUB",   2: "XOR",   3: "OR",    4: "AND",
    8: "SRA",   9: "SRL",  10: "SLL",  25: "LT",   26: "LTU",
   43: "SLT",  44: "SLTU",
}
OP_A_REG_A = 0
OP_B_REG_B = 0
OP_B_IMM = 1
RF_WD_EX = 0


# ── Instruction encoders ──

def r_type(funct7, rs2, rs1, funct3, rd):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0110011

def i_type(imm12, rs1, funct3, rd):
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0010011

def load(imm12, rs1, funct3, rd):
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0000011

def store(imm12, rs2, rs1, funct3):
    return ((imm12 >> 5) & 0x7F) << 25 | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm12 & 0x1F) << 7) | 0b0100011


# ── Coverage database (matches LLM4DV structure) ──

class Coverage:
    def __init__(self):
        self.alu_ops = {op: 0 for op in ["add","sub","or","xor","and","sll","srl","sra","slt","sltu"]}
        self.alu_imm_ops = {op: 0 for op in ["add","sub","or","xor","and","sll","srl","sra","slt","sltu"]}
        self.load_ops = {s: 0 for s in ["word","half-word","byte"]}
        self.store_ops = {s: 0 for s in ["word","half-word","byte"]}

        self.read_reg_a = [0] * 32
        self.read_reg_b = [0] * 32
        self.write_reg = [0] * 32

        # Cross-coverage: dict of op_name -> [0]*32 for each port
        ops = ["add","sub","or","xor","and","sll","srl","sra","slt","sltu"]
        mems = ["word","half-word","byte"]

        self.alu_x_ra = {op: [0]*32 for op in ops}
        self.alu_x_rb = {op: [0]*32 for op in ops}
        self.alu_x_wr = {op: [0]*32 for op in ops}

        self.imm_x_ra = {op: [0]*32 for op in ops}
        self.imm_x_wr = {op: [0]*32 for op in ops}

        self.load_x_ra = {s: [0]*32 for s in mems}
        self.load_x_wr = {s: [0]*32 for s in mems}

        self.store_x_ra = {s: [0]*32 for s in mems}
        self.store_x_rb = {s: [0]*32 for s in mems}

        self.illegal_count = 0
        self.total = 0

    def count_bins(self):
        """Count total covered bins, matching LLM4DV's methodology."""
        bins = 0

        # Type 1: operation bins (26 total)
        bins += sum(1 for v in self.alu_ops.values() if v > 0)
        bins += sum(1 for v in self.alu_imm_ops.values() if v > 0)
        bins += sum(1 for v in self.load_ops.values() if v > 0)
        bins += sum(1 for v in self.store_ops.values() if v > 0)

        # Type 2: register port bins (96 total)
        bins += sum(1 for v in self.read_reg_a if v > 0)
        bins += sum(1 for v in self.read_reg_b if v > 0)
        bins += sum(1 for v in self.write_reg if v > 0)

        # Type 3: cross-coverage
        for d in [self.alu_x_ra, self.alu_x_rb, self.alu_x_wr,
                  self.imm_x_ra, self.imm_x_wr,
                  self.load_x_ra, self.load_x_wr,
                  self.store_x_ra, self.store_x_rb]:
            for lst in d.values():
                bins += sum(1 for v in lst if v > 0)

        return bins

    def count_total_defined(self):
        """Total bins defined in the coverage model (including unreachable)."""
        type1 = 26
        type2 = 96
        # R-type: 10 ops x 3 ports x 32 regs = 960
        # I-type: 10 ops x 2 ports x 32 regs = 640
        # Load:   3 ops x 2 ports x 32 regs  = 192
        # Store:  3 ops x 2 ports x 32 regs  = 192
        type3 = 960 + 640 + 192 + 192
        return type1 + type2 + type3


@cocotb.test()
async def max_coverage_test(dut):
    """Generate the optimal stimulus set for maximum decoder coverage."""

    cov = Coverage()

    async def apply(instruction):
        dut.insn_i.value = instruction
        await Timer(1, units='ns')
        cov.total += 1

        illegal = int(dut.u_decoder.illegal_insn_o.value)
        if illegal:
            cov.illegal_count += 1
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

        # Register ports
        rs1 = rf_raddr_a if rf_ren_a else None
        rs2 = rf_raddr_b if rf_ren_b else None
        rd = rf_waddr if rf_we else None

        if rs1 is not None:
            cov.read_reg_a[rs1] += 1
        if rs2 is not None:
            cov.read_reg_b[rs2] += 1

        # Memory operations
        if data_req:
            size_map = {0: "word", 1: "half-word", 2: "byte"}
            size = size_map.get(data_type)
            if size:
                if data_we:
                    # Store: uses read_A (base addr) and read_B (data)
                    cov.store_ops[size] += 1
                    if rd is None:
                        # stores write to memory not register, but rf_waddr
                        # still appears in instruction encoding
                        pass
                    if rs1 is not None:
                        cov.store_x_ra[size][rs1] += 1
                    if rs2 is not None:
                        cov.store_x_rb[size][rs2] += 1
                else:
                    # Load: uses read_A (address) and write (destination)
                    rd = rf_waddr  # loads always write
                    cov.write_reg[rd] += 1
                    cov.load_ops[size] += 1
                    if rs1 is not None:
                        cov.load_x_ra[size][rs1] += 1
                    cov.load_x_wr[size][rd] += 1
            return

        if rd is not None:
            cov.write_reg[rd] += 1

        # ALU operations
        if rf_we and not mult_sel and not div_sel and rf_ren_a and op_a_mux == OP_A_REG_A and rf_wdata_sel == RF_WD_EX:
            alu_name_map = {0:"add",1:"sub",2:"xor",3:"or",4:"and",
                           8:"sra",9:"srl",10:"sll",43:"slt",44:"sltu"}
            alu_name = alu_name_map.get(alu_op_val)
            if alu_name:
                if op_b_mux == OP_B_IMM:
                    cov.alu_imm_ops[alu_name] += 1
                    if rs1 is not None:
                        cov.imm_x_ra[alu_name][rs1] += 1
                    if rd is not None:
                        cov.imm_x_wr[alu_name][rd] += 1
                else:
                    cov.alu_ops[alu_name] += 1
                    if rs1 is not None:
                        cov.alu_x_ra[alu_name][rs1] += 1
                    if rs2 is not None:
                        cov.alu_x_rb[alu_name][rs2] += 1
                    if rd is not None:
                        cov.alu_x_wr[alu_name][rd] += 1

    # ==================================================================
    # GENERATE OPTIMAL STIMULUS SET
    # ==================================================================

    # ── R-type operations: (funct7, funct3) ──
    # Each uses read_A(rs1), read_B(rs2), write(rd)
    r_ops = [
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

    print("=" * 70)
    print("GENERATING OPTIMAL STIMULUS SET")
    print("=" * 70)

    # For each R-type op, sweep reg 0..31 with rd=rs1=rs2=i
    # This covers all 96 cross-bins (32 x 3 ports) per op in 32 instructions
    for funct7, funct3, name in r_ops:
        for i in range(32):
            await apply(r_type(funct7, i, i, funct3, i))
        print(f"  {name:5s}  32 instructions -> bins so far: {cov.count_bins()}")

    # ── I-type operations: funct3 ──
    # Each uses read_A(rs1), write(rd). No read_B.
    # Note: SUBI does not exist in RISC-V! SUB has no immediate variant.
    i_ops = [
        (0b000, "ADDI"),
        (0b111, "ANDI"),
        (0b110, "ORI"),
        (0b100, "XORI"),
        (0b010, "SLTI"),
        (0b011, "SLTIU"),
    ]

    for funct3, name in i_ops:
        for i in range(32):
            await apply(i_type(42, i, funct3, i))
        print(f"  {name:5s}  32 instructions -> bins so far: {cov.count_bins()}")

    # Shift immediates need special encoding (imm[11:5] encodes the variant)
    shift_i_ops = [
        (0b0000000, 0b001, "SLLI"),   # funct7=0000000, funct3=001
        (0b0000000, 0b101, "SRLI"),   # funct7=0000000, funct3=101
        (0b0100000, 0b101, "SRAI"),   # funct7=0100000, funct3=101
    ]

    for funct7, funct3, name in shift_i_ops:
        for i in range(32):
            # Shift immediate: imm[11:5]=funct7, imm[4:0]=shift_amount
            imm = (funct7 << 5) | 1  # shift by 1
            await apply(i_type(imm, i, funct3, i))
        print(f"  {name:5s}  32 instructions -> bins so far: {cov.count_bins()}")

    # ── Load operations ──
    # Each uses read_A(rs1) for address, write(rd) for destination
    load_ops = [
        (0b010, "LW"),    # word
        (0b001, "LH"),    # half-word
        (0b000, "LB"),    # byte
    ]

    for funct3, name in load_ops:
        for i in range(32):
            await apply(load(0, i, funct3, i))
        print(f"  {name:5s}  32 instructions -> bins so far: {cov.count_bins()}")

    # ── Store operations ──
    # Each uses read_A(rs1) for base address, read_B(rs2) for data
    store_ops = [
        (0b010, "SW"),    # word
        (0b001, "SH"),    # half-word
        (0b000, "SB"),    # byte
    ]

    for funct3, name in store_ops:
        for i in range(32):
            await apply(store(0, i, i, funct3))
        print(f"  {name:5s}  32 instructions -> bins so far: {cov.count_bins()}")

    # ==================================================================
    # RESULTS
    # ==================================================================
    covered = cov.count_bins()
    total_defined = cov.count_total_defined()

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total instructions sent:   {cov.total}")
    print(f"Illegal instructions:      {cov.illegal_count}")
    print(f"Valid instructions:         {cov.total - cov.illegal_count}")
    print()
    print(f"Coverage bins hit:         {covered}")
    print(f"Coverage bins defined:     {total_defined}")
    print(f"Coverage rate:             {covered/total_defined*100:.2f}%")
    print()

    # Detail: which operations were NOT covered?
    print("Operation coverage detail:")
    print(f"  R-type ALU ops:  {sum(1 for v in cov.alu_ops.values() if v > 0)}/10")
    for op, cnt in cov.alu_ops.items():
        status = "HIT" if cnt > 0 else "MISS"
        print(f"    {op:6s}: {status} ({cnt})")

    print(f"  I-type ALU ops:  {sum(1 for v in cov.alu_imm_ops.values() if v > 0)}/10")
    for op, cnt in cov.alu_imm_ops.items():
        status = "HIT" if cnt > 0 else "MISS"
        print(f"    {op:6s}i: {status} ({cnt})")

    print(f"  Load ops:        {sum(1 for v in cov.load_ops.values() if v > 0)}/3")
    print(f"  Store ops:       {sum(1 for v in cov.store_ops.values() if v > 0)}/3")
    print()

    # Register coverage
    ra_hit = sum(1 for v in cov.read_reg_a if v > 0)
    rb_hit = sum(1 for v in cov.read_reg_b if v > 0)
    wr_hit = sum(1 for v in cov.write_reg if v > 0)
    print(f"Register port coverage:")
    print(f"  Read A:   {ra_hit}/32")
    print(f"  Read B:   {rb_hit}/32")
    print(f"  Write:    {wr_hit}/32")
    print()

    # Cross-coverage detail
    cross_hit = 0
    cross_total = 0
    for label, dicts in [
        ("R-type x read_A", cov.alu_x_ra),
        ("R-type x read_B", cov.alu_x_rb),
        ("R-type x write",  cov.alu_x_wr),
        ("I-type x read_A", cov.imm_x_ra),
        ("I-type x write",  cov.imm_x_wr),
        ("Load x read_A",   cov.load_x_ra),
        ("Load x write",    cov.load_x_wr),
        ("Store x read_A",  cov.store_x_ra),
        ("Store x read_B",  cov.store_x_rb),
    ]:
        hit = sum(sum(1 for v in lst if v > 0) for lst in dicts.values())
        tot = sum(len(lst) for lst in dicts.values())
        cross_hit += hit
        cross_total += tot
        print(f"  {label:20s}: {hit:4d} / {tot:4d}")

    print(f"  {'TOTAL':20s}: {cross_hit:4d} / {cross_total:4d}")
    print()

    # The unreachable bins
    unreachable_ops = []
    if cov.alu_imm_ops.get("sub", 0) == 0:
        unreachable_ops.append("SUBI (no subtract-immediate in RISC-V ISA)")
    print("Unreachable bins (by ISA design):")
    for op in unreachable_ops:
        print(f"  - {op}")
    unreachable_count = total_defined - covered
    if cov.alu_imm_ops.get("sub", 0) == 0:
        unreachable_cross = 64  # sub x read_A(32) + sub x write(32)
        unreachable_type1 = 1   # sub immediate
        print(f"  = {unreachable_type1 + unreachable_cross} bins are unreachable")
        reachable_total = total_defined - unreachable_type1 - unreachable_cross
        print(f"\nAdjusted coverage (reachable bins only):")
        print(f"  {covered} / {reachable_total} = {covered/reachable_total*100:.2f}%")
    print("=" * 70)
