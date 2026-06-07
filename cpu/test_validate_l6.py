"""Real-RTL validation for the 5615-bin Level 6 benchmark.

Loads a program produced by rl-coverage/level4_shadow/rl_emit_l6.py, executes it on
the real Verilator-compiled Ibex CPU, and runs a shadow-backed monitor on the
RVFI retirement stream. Compares the observed bin set against the shadow's
predicted bin set.
"""

import os, sys, json
# Add rl-coverage/level4_shadow to sys.path so shadow_cpu_l6 / codec_l6 resolve.
# __file__ isn't always set when cocotb imports us as a module.
_here = os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else "."))
_l4 = os.path.abspath(os.path.join(_here, "..", "rl-coverage", "level4_shadow"))
if _l4 not in sys.path:
    sys.path.insert(0, _l4)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, ReadWrite

# Reuse the memory agent from the existing test
from test_cpu_coverage import MemAgent
from shadow_cpu_l6 import (
    L6History, bins_for_step, advance_history, BIN_NAMES, N_BINS
)
from codec_l6 import decode


WFI = 0x10500073
PROGRAM_PATH = os.environ.get("RL_L6_JSON", "/tmp/rl_l6_program.json")


class ShadowMonitor:
    """Samples RVFI, decodes each retired instruction via codec_l6, and drives
    the shadow's bin-firing logic. Tracks which bins ever fired on real RTL."""

    def __init__(self, dut):
        self.valid = dut.u_top.rvfi_valid
        self.insn = dut.u_top.rvfi_insn
        self.hist = L6History()
        self.hit: set[int] = set()
        self.retired_count = 0
        self.unknown_count = 0

    def sample(self):
        if int(self.valid.value) == 0:
            return
        word = int(self.insn.value)
        self.retired_count += 1
        dec = decode(word)
        if dec is None:
            self.unknown_count += 1
            return
        op_i, rd, rs1, rs2, ib = dec
        for b in bins_for_step(op_i, rd, rs1, rs2, ib, self.hist):
            self.hit.add(b)
        advance_history(op_i, rd, rs1, rs2, self.hist)


@cocotb.test()
async def validate_l6(dut):
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    shadow_hit = set(payload["shadow_hit_bins"])
    machine = list(payload["machine_code"])
    print(f"\nProgram: {len(machine)} instructions, agent={payload['agent']}, seed={payload['seed']}")
    print(f"Shadow predicts {len(shadow_hit)} bins hit.")

    # Boot + memory setup (copied pattern from test_cpu_coverage.full_cpu_test)
    dut.data_gnt_i.value = 0
    dut.data_rvalid_i.value = 0
    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = MemAgent(dut, "data", handle_writes=True)

    # Load program + WFI sentinel
    prog = machine + [WFI]
    imem.load_program(prog, 0x100080)

    monitor = ShadowMonitor(dut)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.rst_ni.value = 1
    await Timer(15, units="ns")
    dut.rst_ni.value = 0
    await ClockCycles(dut.clk_i, 3)
    await Timer(5, units="ns")
    dut.rst_ni.value = 1
    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())

    # Let the CPU retire everything. Loads/CSRs can stall, so budget generously.
    max_cycles = len(prog) * 10 + 500
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()
        monitor.sample()

    real_hit = {BIN_NAMES[b] for b in monitor.hit}

    print("\n" + "=" * 70)
    print(" LEVEL 6 SHADOW vs REAL-RTL COMPARISON")
    print("=" * 70)
    print(f"  Instructions retired: {monitor.retired_count}  "
          f"(undecodable: {monitor.unknown_count})")
    print(f"  Shadow predicted:     {len(shadow_hit):>4}")
    print(f"  Real RTL observed:    {len(real_hit):>4}")
    print(f"  Intersection:         {len(shadow_hit & real_hit):>4}")
    only_s = shadow_hit - real_hit
    only_r = real_hit - shadow_hit
    print(f"  Shadow-only (missed): {len(only_s):>4}")
    for b in sorted(only_s)[:10]:
        print(f"    - {b}")
    if len(only_s) > 10:
        print(f"    ...{len(only_s)-10} more")
    print(f"  Real-only (missed):   {len(only_r):>4}")
    for b in sorted(only_r)[:10]:
        print(f"    + {b}")
    if len(only_r) > 10:
        print(f"    ...{len(only_r)-10} more")

    print()
    if shadow_hit == real_hit:
        print(" VALIDATION PASSED: shadow matches real Ibex exactly on all 5615 bins.")
    else:
        overlap_pct = 100 * len(shadow_hit & real_hit) / max(len(shadow_hit | real_hit), 1)
        print(f" Overlap = {overlap_pct:.2f}% of union (shadow may need a tweak).")
    print("=" * 70)
