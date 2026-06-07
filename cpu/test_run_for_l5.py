"""Minimal cocotb test for Level 5: load a program from JSON, run it, exit.

Intentionally no shadow validation, no noisy output. Verilator's --coverage
flag (already in the Makefile) writes coverage.dat as a side effect of the
run. The Level 5 trainer reads that file after Vtop exits.
"""

import os, sys, json
_here = os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else "."))
_ml4dv = os.path.abspath(os.path.join(_here, ".."))
if _ml4dv not in sys.path:
    sys.path.insert(0, _ml4dv)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, ReadWrite

from test_cpu_coverage import MemAgent

WFI = 0x10500073
PROGRAM_PATH = os.environ.get("RL_L5_JSON", "/tmp/rl_l5_program.json")


@cocotb.test()
async def run_program(dut):
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)
    machine = list(payload["machine_code"]) + [WFI]

    dut.data_gnt_i.value = 0
    dut.data_rvalid_i.value = 0
    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = MemAgent(dut, "data", handle_writes=True)
    imem.load_program(machine, 0x100080)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.rst_ni.value = 1
    await Timer(15, units="ns")
    dut.rst_ni.value = 0
    await ClockCycles(dut.clk_i, 3)
    await Timer(5, units="ns")
    dut.rst_ni.value = 1
    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())

    max_cycles = len(machine) * 8 + 200
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()

    print(f"L5_RUN_COMPLETE: ran {len(machine)} instructions")
