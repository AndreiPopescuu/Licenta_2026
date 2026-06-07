"""test_run_for_l9.py — cocotb driver cu RVFI monitoring pentru functional coverage.

Diferențe față de L8:
  - Citește RVFI la fiecare ciclu valid și construiește coverage model în Python
  - Scrie coverage_functional.json la final (în loc de coverage.dat Verilator)
  - Prologue identic cu L8 (32-reg init + mtvec setup)
"""

import os, sys, json
_here = os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else "."))
if _here not in sys.path: sys.path.insert(0, _here)
_l9 = os.path.join(_here, "..", "rl-coverage", "level9_functional")
if _l9 not in sys.path: sys.path.insert(0, _l9)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, ReadWrite
from test_cpu_coverage import MemAgent
from coverage_model import CoverageModel

WFI  = 0x10500073
MRET = 0x30200073
PROGRAM_PATH = os.environ.get("RL_L9_JSON", "/tmp/rl_l9_program.json")
COV_OUT      = os.environ.get("RL_L9_COV",  "/tmp/rl_l9_coverage.json")

def _addi(rd,rs1,imm): return ((imm&0xFFF)<<20)|((rs1&0x1F)<<15)|((rd&0x1F)<<7)|0x13
def _lui(rd,i20):      return ((i20&0xFFFFF)<<12)|((rd&0x1F)<<7)|0x37
def _csrrw(rd,rs1,csr):return ((csr&0xFFF)<<20)|((rs1&0x1F)<<15)|(1<<12)|((rd&0x1F)<<7)|0x73

def _build_prologue():
    p = []
    p += [_lui(10, 0x00200), _csrrw(0, 10, 0x305)]  # mtvec = 0x00200000
    for r in range(1, 16): p.append(_addi(r, 0, r))
    p.append(_addi(16, 0, 0xFFF))                    # x16 = -1
    p += [_lui(17, 0x7FFFF), _addi(17, 17, 0xFFF)]  # x17 = INT_MAX
    p.append(_lui(18, 0x80000))                       # x18 = INT_MIN
    p.append(_lui(20, 0x00010))                       # x20 = 0x10000
    p += [_lui(21, 0x12345), _addi(21, 21, 0x678)]   # x21 = 0x12345678
    for r in range(22, 32): p.append(_addi(r, 0, r))
    return p

PROLOGUE = _build_prologue()
TRAP_HANDLER_ADDR = 0x00200000
TRAP_HANDLER = [
    (0x341<<20)|(0<<15)|(2<<12)|(10<<7)|0x73,  # CSRRS x10, mepc, x0
    _addi(10, 10, 4),                            # ADDI x10, x10, 4
    (0x341<<20)|(10<<15)|(1<<12)|(0<<7)|0x73,   # CSRRW x0, mepc, x10
    MRET,
]


class DiverseMemAgent(MemAgent):
    async def run_mem(self):
        self.gnt.value = 0; self.rvalid.value = 0
        while True:
            await ClockCycles(self.clk, 1); await ReadWrite()
            self.rvalid.value = 0
            if self.req.value:
                self.gnt.value = 1
                addr = int(self.addr.value)
                wr = int(self.wdata.value) if (self.handle_writes and self.we.value) else None
                await ClockCycles(self.clk, 1); await ReadWrite()
                self.gnt.value = 0; self.rvalid.value = 1
                if wr is not None:
                    self.rdata.value = 0; self.mem_dict[addr] = wr
                elif addr in self.mem_dict:
                    self.rdata.value = self.mem_dict[addr]
                else:
                    self.rdata.value = (addr ^ 0xDEADBEEF) & 0xFFFFFFFF


@cocotb.test()
async def run_program(dut):
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    agent_machine = list(payload["machine_code"])
    full_program  = PROLOGUE + agent_machine + [WFI]

    dut.data_gnt_i.value = 0; dut.data_rvalid_i.value = 0
    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = DiverseMemAgent(dut, "data", handle_writes=True)
    imem.load_program(full_program, 0x100080)
    for i, w in enumerate(TRAP_HANDLER):
        imem.mem_dict[TRAP_HANDLER_ADDR + 4*i] = w

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.rst_ni.value = 1; await Timer(15, units="ns")
    dut.rst_ni.value = 0; await ClockCycles(dut.clk_i, 3); await Timer(5, units="ns")
    dut.rst_ni.value = 1
    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())

    # ── RVFI monitoring ───────────────────────────────────────────────────
    cov_model = CoverageModel()
    rvfi = dut.u_top   # hierarchical reference

    max_cycles = len(full_program) * 15 + 1000
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1); await ReadWrite()
        try:
            if int(rvfi.rvfi_valid.value) == 1:
                cov_model.sample({
                    "insn":      int(rvfi.rvfi_insn.value),
                    "rs1_addr":  int(rvfi.rvfi_rs1_addr.value),
                    "rs2_addr":  int(rvfi.rvfi_rs2_addr.value),
                    "rd_addr":   int(rvfi.rvfi_rd_addr.value),
                    "rs1_rdata": int(rvfi.rvfi_rs1_rdata.value),
                    "rs2_rdata": int(rvfi.rvfi_rs2_rdata.value),
                    "rd_wdata":  int(rvfi.rvfi_rd_wdata.value),
                    "pc_rdata":  int(rvfi.rvfi_pc_rdata.value),
                    "pc_wdata":  int(rvfi.rvfi_pc_wdata.value),
                    "mem_addr":  int(rvfi.rvfi_mem_addr.value),
                    "trap":      int(rvfi.rvfi_trap.value) == 1,
                })
        except Exception:
            pass

    # ── Scrie rezultatele ─────────────────────────────────────────────────
    covered, total = cov_model.count_bins()
    hit_set = list(cov_model.hit_set())
    result = {
        "covered": covered,
        "total":   total,
        "pct":     100.0 * covered / max(total, 1),
        "hits":    hit_set,
    }
    with open(COV_OUT, "w") as f:
        json.dump(result, f)
    print(f"L9_DONE: {covered}/{total} = {result['pct']:.2f}% functional coverage")
