"""Validation: run an RL-generated chain-benchmark program through real Ibex RTL
and compare 1739-bin coverage against the Python shadow's prediction.

Reads /tmp/rl_chains_program.json (written by train_and_emit_chains.py).

Coverage is computed by decoding each retired instruction from rvfi_insn and
running the same bins_for_step / advance_history logic as the shadow.
"""

import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "rl-coverage", "level3_chains"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadWrite, Timer

from test_cpu_coverage import MemAgent, r_type, s_type
from shadow_cpu_chains import (
    Op, bins_for_step, advance_history, ChainHistory, BIN_NAMES, N_BINS
)

PROGRAM_PATH = os.environ.get("RL_CHAINS_JSON", "/tmp/rl_chains_program.json")
WFI = 0x10500073

# ── Instruction encoders (op_idx → 32-bit word) ──────────────────────────────

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
    10: lambda rd, rs1, rs2: s_type(0, rs2, rs1, 0b000),              # SB
    11: lambda rd, rs1, rs2: s_type(0, rs2, rs1, 0b001),              # SH
    12: lambda rd, rs1, rs2: s_type(0, rs2, rs1, 0b010),              # SW
}


def build_program(seq):
    prog = [R_ENC[int(e[0])](int(e[1]), int(e[2]), int(e[3])) for e in seq]
    prog.append(WFI)
    return prog


# ── RVFI instruction decoder ─────────────────────────────────────────────────

_R_DECODE = {
    (0b000, 0): 0,  (0b000, 1): 1,  # ADD, SUB
    (0b001, 0): 2,  (0b010, 0): 3,  (0b011, 0): 4,  # SLL, SLT, SLTU
    (0b100, 0): 5,  (0b101, 0): 6,  (0b101, 1): 7,  # XOR, SRL, SRA
    (0b110, 0): 8,  (0b111, 0): 9,                   # OR, AND
}
_S_DECODE = {0b000: 10, 0b001: 11, 0b010: 12}        # SB, SH, SW


def decode_word(word):
    """Return (op_idx, rd, rs1, rs2) or None for unknown/WFI."""
    opcode = word & 0x7F
    rd     = (word >> 7)  & 0x1F
    funct3 = (word >> 12) & 0x7
    rs1    = (word >> 15) & 0x1F
    rs2    = (word >> 20) & 0x1F
    funct7_b30 = (word >> 30) & 0x1
    if opcode == 0b0110011:
        op = _R_DECODE.get((funct3, funct7_b30))
        if op is not None:
            return (op, rd, rs1, rs2)
    elif opcode == 0b0100011:
        op = _S_DECODE.get(funct3)
        if op is not None:
            return (op, 0, rs1, rs2)
    return None


# ── cocotb test ───────────────────────────────────────────────────────────────

@cocotb.test()
async def rl_validation_chains(dut):
    t_start = time.time()
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    seq        = payload["sequence"]
    shadow_hit = set(payload["shadow_hit_bins"])
    print(f"\nLoaded {len(seq)} instructions | shadow predicts {len(shadow_hit)}/{N_BINS} bins")

    prog = build_program(seq)
    print(f"Assembled {len(prog)} machine words (last = WFI)")

    dut.data_gnt_i.value    = 0
    dut.data_rvalid_i.value = 0
    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = MemAgent(dut, "data",  handle_writes=True)
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

    # Track 1739-bin coverage from RVFI stream
    covered = set()
    hist    = ChainHistory()

    max_cycles = len(prog) * 8 + 200
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()
        if int(dut.u_top.rvfi_valid.value) == 0:
            continue
        word = int(dut.u_top.rvfi_insn.value)
        decoded = decode_word(word)
        if decoded is None:
            continue
        op_idx, rd, rs1, rs2 = decoded
        for b in bins_for_step(op_idx, rd, rs1, rs2, hist):
            covered.add(BIN_NAMES[b])
        advance_history(op_idx, rd, rs1, rs2, hist)

    real_hit    = covered
    only_shadow = shadow_hit - real_hit
    only_real   = real_hit   - shadow_hit

    print("\n" + "=" * 70)
    print(f" SHADOW vs REAL-RTL  [1739-bin chains benchmark]")
    print("=" * 70)
    print(f"  Shadow predicted:   {len(shadow_hit):4d} / {N_BINS}")
    print(f"  Real RTL observed:  {len(real_hit):4d} / {N_BINS}")
    print(f"  Agreement:          {len(shadow_hit & real_hit):4d} bins")
    print(f"\n  Only in shadow (overpredicts): {len(only_shadow)}")
    for b in sorted(only_shadow)[:15]: print(f"    - {b}")
    if len(only_shadow) > 15: print(f"    ...and {len(only_shadow)-15} more")
    print(f"\n  Only in real (underpredicts):  {len(only_real)}")
    for b in sorted(only_real)[:15]: print(f"    + {b}")
    if len(only_real) > 15: print(f"    ...and {len(only_real)-15} more")
    print("\n" + "=" * 70)
    if shadow_hit == real_hit:
        print(f" VALIDATION PASSED: {len(real_hit)}/{N_BINS} bins — shadow matches RTL exactly.")
    else:
        pct = 100 * len(shadow_hit & real_hit) / max(len(shadow_hit | real_hit), 1)
        print(f" Overlap = {pct:.1f}% ({'close' if pct > 90 else 'diverging'}).")
    print(f" RTL validation time: {time.time()-t_start:.1f}s")
    print("=" * 70)
