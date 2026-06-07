"""
Validation: run an RL-generated instruction sequence through real Ibex RTL
and compare the resulting coverage against the Python shadow's prediction.

Reads /tmp/rl_program.json (written by an RL-generator script in rl-coverage/).
"""

import os, sys, json
directory = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, os.path.dirname(directory))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, ReadWrite

from instruction_monitor import InstructionMonitor
from test_cpu_coverage import MemAgent, r_type, s_type, addi

# Map our shadow Op enum values (0..13) to real instruction encoders.
# Only the 13 non-JAL ops are supported in this validation.
R_ENC = {
    0:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b000, rd),  # ADD
    1:  lambda rd, rs1, rs2: r_type(0b0100000, rs2, rs1, 0b000, rd),  # SUB
    2:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b001, rd),  # SLL
    3:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b010, rd),  # SLT
    4:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b011, rd),  # SLTU
    5:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b100, rd),  # XOR
    6:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b101, rd),  # SRL
    7:  lambda rd, rs1, rs2: r_type(0b0100000, rs2, rs1, 0b101, rd),  # SRA
    8:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b110, rd),  # OR
    9:  lambda rd, rs1, rs2: r_type(0b0000000, rs2, rs1, 0b111, rd),  # AND
    10: lambda rd, rs1, rs2: s_type(0, rs2, rs1, 0b000),              # SB  (imm=0)
    11: lambda rd, rs1, rs2: s_type(0, rs2, rs1, 0b001),              # SH
    12: lambda rd, rs1, rs2: s_type(0, rs2, rs1, 0b010),              # SW
}

WFI = 0x10500073
PROGRAM_PATH = os.environ.get("RL_PROGRAM_JSON", "/tmp/rl_program.json")

# BEQ x0, x0, +8  (opcode 0b1100011, not in our 14 ops → fires no coverage bins)
# Used as trampoline scaffolding for backward JAL.
BEQ_SKIP = 0x00000463


def jal_enc(rd, offset):
    o = offset & 0x1FFFFF
    imm20    = (o >> 20) & 0x1
    imm10_1  = (o >> 1)  & 0x3FF
    imm11    = (o >> 11) & 0x1
    imm19_12 = (o >> 12) & 0xFF
    return (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | 0b1101111


def build_from_sequence(seq):
    prog = []
    for entry in seq:
        op  = int(entry[0])
        rd  = int(entry[1])
        rs1 = int(entry[2])
        rs2 = int(entry[3])
        imm_sign = int(entry[4]) if len(entry) > 4 else 0

        if op == 13:  # JAL
            if imm_sign < 0:
                # Backward JAL trampoline — no infinite loop, no extra coverage bins:
                #   [A]:   BEQ x0,x0,+8  → skip to [A+8] (the backward JAL)
                #   [A+4]: BEQ x0,x0,+8  → trampoline: escape to [A+12] (next instr)
                #   [A+8]: jal(rd, -4)   → backward JAL → jumps to [A+4] (trampoline)
                prog.append(BEQ_SKIP)           # skip to backward JAL
                prog.append(BEQ_SKIP)           # trampoline: escape past backward JAL
                prog.append(jal_enc(rd, -4))    # backward JAL
            else:
                # Forward JAL: offset +4 = jump directly to next instruction.
                # No NOP needed, RAW hazard chain preserved.
                prog.append(jal_enc(rd, +4))
        elif op in R_ENC:
            prog.append(R_ENC[op](rd, rs1, rs2))
    prog.append(WFI)
    return prog


@cocotb.test()
async def rl_validation(dut):
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    seq = payload["sequence"]
    shadow_hit = set(payload["shadow_hit_bins"])
    print(f"\nLoaded {len(seq)} instructions, shadow predicts {len(shadow_hit)} bins hit.")

    prog = build_from_sequence(seq)
    print(f"Assembled program: {len(prog)} machine words (last = WFI).")

    # Standard CPU boot sequence (copied from test_cpu_coverage.full_cpu_test)
    dut.data_gnt_i.value = 0
    dut.data_rvalid_i.value = 0
    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = MemAgent(dut, "data", handle_writes=True)
    monitor = InstructionMonitor(dut)
    imem.load_program(prog, 0x100080)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.rst_ni.value = 1
    await Timer(15, units="ns")
    dut.rst_ni.value = 0
    await ClockCycles(dut.clk_i, 3)
    await Timer(5, units="ns")
    dut.rst_ni.value = 1
    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())

    # ~4 cycles per retire on 2-stage pipeline + stalls for stores + slack
    max_cycles = len(prog) * 8 + 200
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()
        monitor.sample_insn_coverage()

    cov = monitor.coverage_db.get_coverage_dict()
    real_hit = {name for name, cnt in cov.items() if cnt > 0}

    print("\n" + "=" * 70)
    print(" SHADOW vs REAL-RTL COVERAGE COMPARISON")
    print("=" * 70)
    print(f"  Shadow predicted:   {len(shadow_hit):3d} bins")
    print(f"  Real RTL observed:  {len(real_hit):3d} bins")
    print(f"  Agreement:          {len(shadow_hit & real_hit):3d} bins (intersection)")

    only_shadow = shadow_hit - real_hit
    only_real = real_hit - shadow_hit
    print(f"\n  Bins only in shadow (shadow overpredicts): {len(only_shadow)}")
    for b in sorted(only_shadow)[:20]:
        print(f"    - {b}")
    if len(only_shadow) > 20:
        print(f"    ...and {len(only_shadow)-20} more")

    print(f"\n  Bins only in real (shadow underpredicts): {len(only_real)}")
    for b in sorted(only_real)[:20]:
        print(f"    + {b}")
    if len(only_real) > 20:
        print(f"    ...and {len(only_real)-20} more")

    print("\n" + "=" * 70)
    if shadow_hit == real_hit:
        print(" VALIDATION PASSED: shadow matches real RTL exactly.")
    else:
        pct = 100 * len(shadow_hit & real_hit) / max(len(shadow_hit | real_hit), 1)
        print(f" Overlap = {pct:.1f}% (of union). Shadow is {'close' if pct > 90 else 'diverging'}.")
    print("=" * 70)
