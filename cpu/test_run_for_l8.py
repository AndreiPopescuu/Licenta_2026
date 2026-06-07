"""Level 8 cocotb driver — enhanced for LINE coverage.

Extends the L7 driver with:
  1. Longer prologue that pre-initialises all 32 registers with diverse values
     (ensures branches compare non-trivial operands from the first instruction).
  2. Trap handler advanced: reads both mepc (0x341) and mcause (0x342) before
     MRET, toggling more CSR-register paths.
  3. Data memory returns addr XOR 0xDEADBEEF for read-misses (same as L7).
  4. Accepts programs from RL_L8_JSON (falls back to RL_L7_JSON for compat).

Program layout:
    0x00100080  PROLOGUE (LUI/ADDI to initialise registers)
    0x00100080 + 4*len(PROLOGUE)  agent program
    ...
    0x00200000  TRAP HANDLER (reads mepc+mcause, advances mepc, MRET)
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


WFI   = 0x10500073
MRET  = 0x30200073


# ── Register initialisation prologue ─────────────────────────────────────
# Goal: give every register a distinct, non-zero, non-trivial value so that
# branches, ALU ops, and loads receive meaningful operands from instruction 1.
#
# Strategy: use LUI+ADDI pairs for regs that need a 32-bit value;
# plain ADDI x0 for small values.
#
# Register plan:
#   x0  = 0  (hardwired)
#   x1  = 0x00001001
#   x2  = 0x00002002   (SP-like, will be overwritten by agent)
#   x3  = 0x00003003
#   ...
#   x10 = 0x00010010   (function argument — diverse)
#   x11 = 0x0000000B   (small)
#   x15 = 0x0000000F
#   x16 = 0xFFFFFFFF   (-1 for signed-compare tests)
#   x17 = 0x7FFFFFFF   (INT_MAX)
#   x18 = 0x80000000   (INT_MIN, largest unsigned)
#   x19 = 0x55555555   (alternating bits)
#   x20 = 0x00010000   (safe data-memory base address)
#   x21 = 0xAAAAAAAA   (alternating bits, inverted)
#   x22 = 0x12345678
#   x23 = 0xFEDCBA98 (sign-extended upper portion)
#   x24..x31 = small values 24..31

def _addi_w(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b0010011

def _lui_w(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | 0b0110111

def _csrrw_w(rd, rs1, csr):
    return ((csr & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | (0b001 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _build_prologue():
    p = []

    def emit(*ws): p.extend(ws)

    # Set mtvec = 0x00200000 (trap handler location)
    emit(_lui_w(10, 0x00200))          # x10 = 0x00200000
    emit(_csrrw_w(0, 10, 0x305))       # CSR[mtvec] = x10

    # Register diversity setup
    for r in range(1, 16):
        emit(_addi_w(r, 0, r))          # x1..x15 = 1..15

    # x16 = -1
    emit(_addi_w(16, 0, 0xFFF))        # ADDI x16, x0, -1 (sign-extended)

    # x17 = 0x7FFFFFFF (INT_MAX)
    emit(_lui_w(17, 0x7FFFF))
    emit(_addi_w(17, 17, 0xFFF))       # x17 = 0x7FFFF000 + 0xFFF = 0x7FFFFFFF

    # x18 = 0x80000000 (INT_MIN / largest unsigned)
    emit(_lui_w(18, 0x80000))

    # x19 = 0x55555555 (alternating bits)
    emit(_lui_w(19, 0x55555))
    emit(_addi_w(19, 19, 0x555))       # x19 = 0x55555000 + 0x555 = 0x55555555

    # x20 = 0x00010000 (data-mem base)
    emit(_lui_w(20, 0x00010))

    # x21 = 0xAAAAAAAA (alternating bits inverted)
    emit(_lui_w(21, 0xAAAAA))
    emit(_addi_w(21, 21, 0xAAA & 0xFFF))  # sign-extends: 0xAAAA000 - 0x556 = 0xAAA9AAA?
    # Use correct two's complement: 0xAAA = -0x556 as 12-bit signed
    # LUI 0xAAAAB + ADDI -0x556... just use simpler approach:
    emit(_lui_w(21, 0xAAAAB))
    emit(_addi_w(21, 21, -0x556 & 0xFFF))  # 0xAAAAB000 - 0x556 = 0xAAAAAAAA

    # x22 = 0x12345678
    emit(_lui_w(22, 0x12345))
    emit(_addi_w(22, 22, 0x678))

    # x23 = 0xFEDCBA98
    # LUI 0xFEDCC + ADDI -0x368 (because 0xFEDCC000 - 0x368 = 0xFEDCBA98? Let's check:
    # 0xFEDCC000 = 4275765248, 0xBA98 = 47768, 4275765248 + 47768? No.
    # FEDCBA98: LUI = 0xFEDCB, ADDI = 0xA98 (sign-ext = 0xFFFFF-0x568 = -0xA68? no)
    # 0xA98 as 12-bit = 2712, positive. 0xFEDCB000 + 0xA98 = 0xFEDCBA98. ✓
    emit(_lui_w(23, 0xFEDCB))
    emit(_addi_w(23, 23, 0xA98))

    # x24..x31 = 24..31
    for r in range(24, 32):
        emit(_addi_w(r, 0, r))

    return p


PROLOGUE = _build_prologue()

TRAP_HANDLER_ADDR = 0x00200000

# Enhanced trap handler: reads mepc, mcause, mtval; advances mepc by 4; MRET.
# This exercises more CSR-register read paths in ibex_cs_registers.
def _build_trap_handler():
    p = []
    # x10 = mepc
    p.append(0x34102573)   # CSRRS x10, 0x341, x0
    # x11 = mcause
    p.append(0x34202573 | (11 << 7))  # CSRRS x11, 0x342, x0
    # Actually encode properly:
    # CSRRS rd=x11, rs1=x0, csr=0x342
    p[-1] = (0x342 << 20) | (0 << 15) | (0b010 << 12) | (11 << 7) | 0b1110011
    # x12 = mtval
    p.append((0x343 << 20) | (0 << 15) | (0b010 << 12) | (12 << 7) | 0b1110011)
    # x10 = mepc + 4
    p.append(_addi_w(10, 10, 4))
    # mepc = x10
    p.append(0x34151073)   # CSRRW x0, 0x341, x10
    # MRET
    p.append(MRET)
    return p

TRAP_HANDLER = _build_trap_handler()


class DiverseMemAgent(MemAgent):
    """Data-memory agent: returns (addr XOR 0xDEADBEEF) for read-misses."""
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
                    self.rdata.value = (access_addr ^ 0xDEADBEEF) & 0xFFFFFFFF


PROGRAM_PATH = os.environ.get(
    "RL_L8_JSON",
    os.environ.get("RL_L7_JSON", "/tmp/rl_l8_program.json")
)


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

    # Extra cycles for MUL/DIV (multi-cycle ops) + trap round-trips
    max_cycles = len(full_program) * 80 + 5000
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()

    print(f"L8_RUN_COMPLETE: ran {len(full_program)} instr "
          f"(prologue={len(PROLOGUE)}, agent={len(agent_machine)})")
