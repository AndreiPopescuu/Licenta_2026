"""
Full Ibex CPU Coverage Test
============================
Load programs into the CPU's instruction memory and let it execute them.
Track coverage using the RVFI (RISC-V Formal Interface) to see which
instructions actually retire.

196 bins across 6 types:
  Type 1 (SEEN):        14 bins - execute each operation once
  Type 2 (ZERO_DST):    11 bins - R-type/JAL with rd=x0
  Type 3 (ZERO_SRC):    13 bins - R-type/S-type with x0 as source
  Type 4 (SAME_SRC):    13 bins - R-type/S-type with rs1==rs2
  Type 5 (BR):           2 bins - JAL forward + backward
  Type 6 (RAW_HAZARD): 143 bins - each R/S-type after R/JAL-type with
                                   data dependency
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, ReadWrite

from instruction_monitor import InstructionMonitor
from instructions import Instr, Cov


# ── Instruction encoders ──

def r_type(funct7, rs2, rs1, funct3, rd):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0110011

def s_type(imm12, rs2, rs1, funct3):
    imm_hi = (imm12 >> 5) & 0x7F
    imm_lo = imm12 & 0x1F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | 0b0100011

def jal(rd, offset):
    """Encode JAL instruction. offset is in bytes, must be even."""
    o = offset & 0x1FFFFF  # 21-bit signed
    imm20    = (o >> 20) & 0x1
    imm10_1  = (o >> 1)  & 0x3FF
    imm11    = (o >> 11) & 0x1
    imm19_12 = (o >> 12) & 0xFF
    return (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | 0b1101111

def addi(rd, rs1, imm):
    """ADDI - used to set up register values."""
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b0010011

def nop():
    """NOP = ADDI x0, x0, 0"""
    return addi(0, 0, 0)


# R-type operation encoders
R_OPS = {
    'ADD':  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b000, rd),
    'SUB':  lambda rd, rs1, rs2: r_type(0b0100000, rs2, rs1, 0b000, rd),
    'AND':  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b111, rd),
    'OR':   lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b110, rd),
    'XOR':  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b100, rd),
    'SLL':  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b001, rd),
    'SRL':  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b101, rd),
    'SRA':  lambda rd, rs1, rs2: r_type(0b0100000, rs2, rs1, 0b101, rd),
    'SLT':  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b010, rd),
    'SLTU': lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b011, rd),
}

S_OPS = {
    'SW': lambda rs2, rs1, imm=0: s_type(imm, rs2, rs1, 0b010),
    'SH': lambda rs2, rs1, imm=0: s_type(imm, rs2, rs1, 0b001),
    'SB': lambda rs2, rs1, imm=0: s_type(imm, rs2, rs1, 0b000),
}

R_OP_NAMES = list(R_OPS.keys())
S_OP_NAMES = list(S_OPS.keys())


class MemAgent:
    """Simple memory agent: responds to CPU memory requests."""
    def __init__(self, dut, mem_name, handle_writes=True):
        self.mem_dict = {}
        self.clk = dut.clk_i
        self.gnt = getattr(dut, mem_name + "_gnt_i")
        self.req = getattr(dut, mem_name + "_req_o")
        self.addr = getattr(dut, mem_name + "_addr_o")
        self.rvalid = getattr(dut, mem_name + "_rvalid_i")
        self.rdata = getattr(dut, mem_name + "_rdata_i")
        self.handle_writes = handle_writes
        if handle_writes:
            self.we = getattr(dut, mem_name + "_we_o")
            self.wdata = getattr(dut, mem_name + "_wdata_o")

    def write_mem(self, addr, word):
        self.mem_dict[addr] = word

    def load_program(self, instructions, start_addr):
        for i, insn in enumerate(instructions):
            self.mem_dict[start_addr + i * 4] = insn

    async def run_mem(self):
        self.gnt.value = 0
        self.rvalid.value = 0
        while True:
            await ClockCycles(self.clk, 1)
            await ReadWrite()
            self.rvalid.value = 0
            if self.req.value:
                self.gnt.value = 1
                access_addr = self.addr.value
                write_data = None
                if self.handle_writes and self.we.value:
                    write_data = self.wdata.value
                await ClockCycles(self.clk, 1)
                await ReadWrite()
                self.gnt.value = 0
                self.rvalid.value = 1
                if self.handle_writes and write_data:
                    self.rdata.value = 0
                    self.mem_dict[int(access_addr)] = int(write_data)
                else:
                    # Default: WFI encoding (causes CPU to sleep, not crash)
                    self.rdata.value = self.mem_dict.get(int(access_addr), 0x10500073)


def build_program():
    """Build a program that hits all 196 coverage bins."""
    prog = []

    # Setup: load useful values into registers
    # x1=1, x2=2, x3=3, ... using ADDI from x0
    for i in range(1, 16):
        prog.append(addi(i, 0, i))

    # ── Type 1 (SEEN) + Type 2 (ZERO_DST) + Type 3 (ZERO_SRC) + Type 4 (SAME_SRC) ──
    # For each R-type operation, emit:
    #   1. Normal:    OP x16, x1, x2        -> SEEN
    #   2. Zero dst:  OP x0,  x1, x2        -> ZERO_DST
    #   3. Zero src:  OP x17, x0, x2        -> ZERO_SRC
    #   4. Same src:  OP x18, x3, x3        -> SAME_SRC

    for name, enc in R_OPS.items():
        prog.append(enc(16, 1, 2))   # SEEN
        prog.append(enc(0, 1, 2))    # ZERO_DST
        prog.append(enc(17, 0, 2))   # ZERO_SRC
        prog.append(enc(18, 3, 3))   # SAME_SRC

    # For each S-type operation:
    #   1. Normal:    SX x1, 0(x2)           -> SEEN
    #   2. Zero src:  SX x0, 0(x2)           -> ZERO_SRC (rs2=x0)
    #   3. Same src:  SX x3, 0(x3)           -> SAME_SRC

    for name, enc in S_OPS.items():
        prog.append(enc(1, 2, 0))    # SEEN: store x1 to addr in x2
        prog.append(enc(0, 2, 0))    # ZERO_SRC: store x0
        prog.append(enc(3, 3, 0))    # SAME_SRC: rs1==rs2

    # ── Type 5 (JAL forward + backward) + JAL SEEN + ZERO_DST ──
    # JAL forward: jump ahead by 8 bytes (skip 1 instruction)
    prog.append(jal(1, 8))           # JAL x1, +8 -> SEEN + BR_FORWARDS
    prog.append(nop())               # skipped
    # JAL with rd=x0 (ZERO_DST for JAL)
    prog.append(jal(0, 8))           # JAL x0, +8 -> ZERO_DST
    prog.append(nop())               # skipped
    # JAL backward: jump back
    # We need to be careful: jumping backward means re-executing code.
    # Use a register as a flag: first time through, set flag and jump back;
    # second time, skip the jump.
    # Simple approach: JAL backward to a NOP, then continue.
    # addr of next insn = current + 4. We want to jump to current - 4.
    # But that would create an infinite loop. Instead:
    # Place a forward JAL that goes +12 to skip the backward JAL on re-entry
    prog.append(addi(20, 0, 0))      # x20 = 0 (flag)
    prog.append(jal(1, -4))          # JAL x1, -4 -> BR_BACKWARDS (jumps back to addi)
    # After backward JAL executes, it returns to addi(x20,0,0) which then falls
    # through to jal(1,-4) again -> infinite loop!
    # Fix: use a different strategy. Pre-place the backward JAL so it's only
    # executed once by jumping INTO it from a forward JAL.

    # Actually, let's simplify. The backward JAL will re-execute the previous
    # instruction. To avoid infinite loops, we can overwrite the backward JAL
    # after first execution. But we can't self-modify easily.
    # Simplest: accept a tiny loop. The JAL backward will jump to itself-4,
    # execute the instruction there, then hit the JAL backward again, then
    # re-execute. We need the NEXT instruction after the loop to break out.
    # Use: the addi above sets x20. Then JAL goes back to it. Then addi runs
    # again (x20=0 again), then JAL runs again. Infinite.
    # Instead, let's just do: the backward JAL target is a forward JAL that
    # skips over the backward JAL.

    # Let me restart the JAL section cleanly:
    # Remove the problematic backward JAL we just added
    prog.pop()  # remove jal(1, -4)
    prog.pop()  # remove addi(20, 0, 0)

    # Clean JAL section:
    #   addr+0:  JAL x1, +12        (forward, skips 2 insns)
    #   addr+4:  NOP                (skipped on first pass)
    #   addr+8:  JAL x2, +12        (skipped on first pass; target of backward JAL)
    #   addr+12: NOP                (landed here from first JAL)
    #   addr+16: JAL x3, -8         (backward jump to addr+8)
    #   addr+20: NOP                (landed here from JAL at addr+8)
    # Execution: addr+0 -> addr+12 -> addr+16(backward) -> addr+8 -> addr+20
    # This gives us: forward JAL at addr+0, backward JAL at addr+16,
    # and addr+8 is a forward JAL that escapes the loop.

    # But we already emitted the forward JALs above. Let me just redo this part.
    # Remove the two JALs and NOP we added:
    prog.pop()  # nop (after jal(0,8))
    prog.pop()  # jal(0, 8)
    prog.pop()  # nop (after jal(1,8))
    prog.pop()  # jal(1, 8)

    # Now add clean JAL sequence:
    prog.append(jal(1, 16))           # [A+0]  JAL x1, +16 -> forward, go to A+16
    prog.append(jal(0, 12))           # [A+4]  JAL x0, +12 -> forward + ZERO_DST, go to A+16
    prog.append(nop())                # [A+8]  NOP (landing pad for backward JAL below)
    prog.append(jal(4, 12))           # [A+12] escape: JAL x4, +12 -> go to A+24
    prog.append(jal(1, -8))           # [A+16] JAL x1, -8 -> backward, go to A+8
    prog.append(nop())                # [A+20] (not reached directly)
    # Execution: A+0(fwd)->A+16(bwd)->A+8(nop)->A+12(escape)->A+24
    # Then A+4 is never reached... we need it to be reached for ZERO_DST.
    # Let me restructure:

    prog.pop()  # A+20
    prog.pop()  # A+16
    prog.pop()  # A+12
    prog.pop()  # A+8
    prog.pop()  # A+4
    prog.pop()  # A+0

    # Final clean JAL section:
    prog.append(jal(1, 8))            # [A] JAL x1,+8 -> SEEN+BR_FORWARDS, skip to A+8
    prog.append(nop())                # [A+4] skipped
    prog.append(jal(0, 8))            # [A+8] JAL x0,+8 -> ZERO_DST+BR_FORWARDS, skip to A+16
    prog.append(nop())                # [A+12] skipped
    # For backward JAL: jump to A+12 (the NOP), then fall through to...
    # we need an escape after the NOP at A+12.
    # Rewrite: put escape JAL at A+12 instead of NOP
    prog.pop()                        # remove A+12 nop
    prog.append(jal(5, 12))           # [A+12] escape JAL -> goes to A+24
    prog.append(jal(2, -4))           # [A+16] JAL x2,-4 -> BR_BACKWARDS, goes to A+12
    prog.append(nop())                # [A+20] not reached
    prog.append(nop())                # [A+24] continuation after escape
    # Execution: A(fwd)->A+8(fwd)->A+16(bwd)->A+12(escape)->A+24. All 3 JALs execute.

    # ── Type 6 (RAW_HAZARD) ──
    # The monitor checks: did the PREVIOUS retired instruction write to a
    # register that the CURRENT retired instruction reads?
    # Key: producer and consumer must be TRULY consecutive in retirement order.
    # No other instruction (like a setup ADDI) can be between them.
    #
    # For R-type producer: OP x19, x1, x2     (writes x19)
    # For R-type consumer: OP x20, x19, x1    (reads x19 -> RAW hazard)
    # For JAL producer:    JAL x19, +8         (writes x19, skips 1 insn)
    #                      NOP                 (skipped by JAL)
    # For S-type consumer: SX x19, 0(x1)      (reads x19 via rs2)

    all_consumers = list(R_OP_NAMES) + list(S_OP_NAMES)  # 13
    producer_ops_list = list(R_OPS.keys()) + ['JAL']     # 11

    for prod_name in producer_ops_list:
        for cons_name in all_consumers:
            # Producer writes to x19
            if prod_name == 'JAL':
                prog.append(jal(19, 8))     # JAL x19, +8
                prog.append(nop())          # skipped by JAL
            else:
                prog.append(R_OPS[prod_name](19, 1, 2))

            # Consumer reads x19 IMMEDIATELY after producer retires
            if cons_name in R_OPS:
                prog.append(R_OPS[cons_name](20, 19, 1))
            else:
                # S-type: rs2 is the data register, rs1 is address
                prog.append(S_OPS[cons_name](19, 1, 0))

    # End with WFI (wait for interrupt - halts CPU cleanly)
    prog.append(0x10500073)  # WFI

    return prog


@cocotb.test()
async def full_cpu_test(dut):
    """Load and run a program targeting all 196 coverage bins."""

    # Setup memory agents
    dut.data_gnt_i.value = 0
    dut.data_rvalid_i.value = 0

    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = MemAgent(dut, "data", handle_writes=True)
    monitor = InstructionMonitor(dut)

    # Build and load program
    program = build_program()
    print(f"\nProgram size: {len(program)} instructions ({len(program)*4} bytes)")
    imem.load_program(program, 0x100080)

    # Start clock and reset
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    # Reset sequence
    dut.rst_ni.value = 1
    await Timer(15, units="ns")
    dut.rst_ni.value = 0
    await ClockCycles(dut.clk_i, 3)
    await Timer(5, units="ns")
    dut.rst_ni.value = 1

    # Start memory agents
    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())

    # Run simulation - let CPU execute the program
    # Each instruction takes ~2-4 clock cycles in the 2-stage pipeline
    max_cycles = len(program) * 10 + 200
    print(f"Running for up to {max_cycles} clock cycles...")

    for cycle in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()
        monitor.sample_insn_coverage()

    # ── Report results ──
    print("\n" + "=" * 70)
    print("FULL IBEX CPU COVERAGE RESULTS")
    print("=" * 70)

    cov_dict = monitor.coverage_db.get_coverage_dict()
    total_bins = len(cov_dict)
    covered_bins = sum(1 for v in cov_dict.values() if v > 0)

    print(f"\nTotal bins:    {total_bins}")
    print(f"Covered bins:  {covered_bins}")
    print(f"Coverage:      {covered_bins}/{total_bins} = {covered_bins/total_bins*100:.1f}%")

    # Detail by type
    print(f"\n{'Bin':40s} {'Count':>6s}  Status")
    print("-" * 55)

    seen_bins = []
    zero_dst_bins = []
    zero_src_bins = []
    same_src_bins = []
    branch_bins = []
    raw_bins = []

    for name, count in sorted(cov_dict.items()):
        status = "HIT" if count > 0 else "MISS"
        if "_seen" in name and "->" not in name:
            seen_bins.append((name, count, status))
        elif "_zero_dst" in name:
            zero_dst_bins.append((name, count, status))
        elif "_zero_src" in name:
            zero_src_bins.append((name, count, status))
        elif "_same_src" in name:
            same_src_bins.append((name, count, status))
        elif "br_" in name:
            branch_bins.append((name, count, status))
        elif "raw_hazard" in name:
            raw_bins.append((name, count, status))

    for label, bins in [
        ("TYPE 1 - SEEN", seen_bins),
        ("TYPE 2 - ZERO_DST", zero_dst_bins),
        ("TYPE 3 - ZERO_SRC", zero_src_bins),
        ("TYPE 4 - SAME_SRC", same_src_bins),
        ("TYPE 5 - BRANCH", branch_bins),
        ("TYPE 6 - RAW HAZARD", raw_bins),
    ]:
        hit = sum(1 for _, c, _ in bins if c > 0)
        print(f"\n  {label} ({hit}/{len(bins)}):")
        for name, count, status in bins:
            marker = "  " if count > 0 else ">>"
            print(f"  {marker} {name:40s} {count:6d}  {status}")

    print("\n" + "=" * 70)
    print(f"FINAL: {covered_bins}/{total_bins} bins = {covered_bins/total_bins*100:.1f}%")
    print("=" * 70)
