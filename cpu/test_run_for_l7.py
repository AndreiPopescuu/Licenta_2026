"""Level 7 cocotb driver with stimulus enhancements:

  1. Instruction memory preloaded with a minimal mtvec-setup prologue so the
     agent can safely emit ECALL / EBREAK — they trap to a stub that just
     does MRET.
  2. Data memory returns a *diverse address-dependent pattern* for unwritten
     addresses (rather than the fixed 0x10500073 WFI).  This gives loads
     real bit diversity on the data bus and downstream LSU / register-file
     read paths without requiring the agent to store-then-load.

Program layout in memory:
    0x00100080  <prologue: LUI + CSRRW to set mtvec = 0x00200000>
    0x00100088  <agent's program, appended by the caller>
    0x00100088+4*N  <WFI tail — CPU sleeps>
    0x00200000  <trap handler: read mepc, +4, write mepc, MRET>

The trap handler advances mepc by 4 so ECALL/EBREAK/illegal-instr don't
infinite-loop on their own PC.
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


WFI  = 0x10500073
MRET = 0x30200073
PROGRAM_PATH = os.environ.get("RL_L7_JSON", "/tmp/rl_l7_program.json")


# Prologue (pre-pended by this driver, not counted in the agent's action budget):
# LUI  x10, 0x00200   -> x10 = 0x00200000
# CSRRW x0, 0x305, x10 -> mtvec = x10
PROLOGUE = [
    0x00200537,   # LUI  x10, 0x200
    0x30551073,   # CSRRW x0, 0x305, x10   (writes mtvec)
]

TRAP_HANDLER_ADDR = 0x00200000

# Trap handler: read mepc into x10, add 4, write back, MRET.
# This ensures ECALL/EBREAK/illegal-instr return to the instruction *after*
# the trap, instead of re-entering it and infinite-looping.
TRAP_HANDLER = [
    0x34102573,   # CSRRS x10, 0x341, x0   (x10 = mepc)
    0x00450513,   # ADDI  x10, x10, 4
    0x34151073,   # CSRRW x0, 0x341, x10   (mepc = x10)
    MRET,         # MRET
]


class DiverseMemAgent(MemAgent):
    """Data-memory agent that returns address-dependent patterns on read-miss.

    Uses `addr XOR 0xDEADBEEF` as the default to toggle high bits of the data
    bus, rather than the constant 0x10500073 WFI.  Writes are recorded
    normally (so store-then-load still works with the written value).
    """
    async def run_mem(self):
        self.gnt.value = 0
        self.rvalid.value = 0
        while True:
            await ClockCycles(self.clk, 1)
            await ReadWrite()
            self.rvalid.value = 0
            if self.req.value:
                self.gnt.value = 1
                access_addr = int(self.addr.value)
                write_data = None
                if self.handle_writes and self.we.value:
                    write_data = int(self.wdata.value)
                await ClockCycles(self.clk, 1)
                await ReadWrite()
                self.gnt.value = 0
                self.rvalid.value = 1
                if self.handle_writes and write_data is not None:
                    self.rdata.value = 0
                    self.mem_dict[access_addr] = write_data
                elif access_addr in self.mem_dict:
                    self.rdata.value = self.mem_dict[access_addr]
                else:
                    pattern = (access_addr ^ 0xDEADBEEF) & 0xFFFFFFFF
                    self.rdata.value = pattern


@cocotb.test()
async def run_program(dut):
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    agent_machine = list(payload["machine_code"])
    full_program = PROLOGUE + agent_machine + [WFI]

    dut.data_gnt_i.value = 0
    dut.data_rvalid_i.value = 0
    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = DiverseMemAgent(dut, "data", handle_writes=True)

    imem.load_program(full_program, 0x100080)
    # Trap handler at mtvec's destination (4 instructions, 16 bytes)
    for i, word in enumerate(TRAP_HANDLER):
        imem.mem_dict[TRAP_HANDLER_ADDR + 4 * i] = word

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.rst_ni.value = 1
    await Timer(15, units="ns")
    dut.rst_ni.value = 0
    await ClockCycles(dut.clk_i, 3)
    await Timer(5, units="ns")
    dut.rst_ni.value = 1
    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())

    # Slightly more cycles than L5 since a trap round-trip is a few extra cycles
    max_cycles = len(full_program) * 10 + 400
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()

    print(f"L7_RUN_COMPLETE: ran {len(full_program)} instructions "
          f"(prologue={len(PROLOGUE)}, agent={len(agent_machine)})")
