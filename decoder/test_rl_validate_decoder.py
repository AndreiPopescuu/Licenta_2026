"""Validation: run an RL-generated instruction sequence through the real Ibex
decoder RTL and compare coverage against the Python shadow's prediction.

Accepts two JSON formats:
  - mode="structured": sequence of [op_idx, rd, rs1, rs2] tuples
  - mode="raw":        program of raw 32-bit words

Usage:
  make MODULE=test_rl_validate_decoder                          (structured, default)
  RL_DECODER_JSON=/tmp/rl_decoder_raw_program.json make MODULE=test_rl_validate_decoder
"""

import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "rl-coverage", "level1_decoder"))

import cocotb
from cocotb.triggers import Timer

from shadow_decoder import BIN_NAMES, N_BINS, OP_TYPES
from test_max_coverage import Coverage

PROGRAM_PATH = os.environ.get("RL_DECODER_JSON", "/tmp/rl_decoder_program.json")

_OP_KEY = {(k, n): i for i, (k, n) in enumerate(OP_TYPES)}


# ── Instruction encoders (structured → 32-bit word) ──────────────────────────

def _r(funct7, funct3, rd, rs1, rs2):
    return (funct7 << 25)|(rs2 << 20)|(rs1 << 15)|(funct3 << 12)|(rd << 7)|0b0110011

def _i(funct3, rd, rs1, imm=0):
    return ((imm&0xFFF)<<20)|(rs1<<15)|(funct3<<12)|(rd<<7)|0b0010011

def _load(funct3, rd, rs1):
    return (rs1<<15)|(funct3<<12)|(rd<<7)|0b0000011

def _store(funct3, rs1, rs2):
    return (rs2<<20)|(rs1<<15)|(funct3<<12)|0b0100011

_R = {
    "add":  lambda rd,rs1,rs2: _r(0b0000000,0b000,rd,rs1,rs2),
    "sub":  lambda rd,rs1,rs2: _r(0b0100000,0b000,rd,rs1,rs2),
    "xor":  lambda rd,rs1,rs2: _r(0b0000000,0b100,rd,rs1,rs2),
    "or":   lambda rd,rs1,rs2: _r(0b0000000,0b110,rd,rs1,rs2),
    "and":  lambda rd,rs1,rs2: _r(0b0000000,0b111,rd,rs1,rs2),
    "sll":  lambda rd,rs1,rs2: _r(0b0000000,0b001,rd,rs1,rs2),
    "srl":  lambda rd,rs1,rs2: _r(0b0000000,0b101,rd,rs1,rs2),
    "sra":  lambda rd,rs1,rs2: _r(0b0100000,0b101,rd,rs1,rs2),
    "slt":  lambda rd,rs1,rs2: _r(0b0000000,0b010,rd,rs1,rs2),
    "sltu": lambda rd,rs1,rs2: _r(0b0000000,0b011,rd,rs1,rs2),
}
_I = {
    "add":  lambda rd,rs1: _i(0b000,rd,rs1),
    "xor":  lambda rd,rs1: _i(0b100,rd,rs1),
    "or":   lambda rd,rs1: _i(0b110,rd,rs1),
    "and":  lambda rd,rs1: _i(0b111,rd,rs1),
    "sll":  lambda rd,rs1: _i(0b001,rd,rs1),
    "srl":  lambda rd,rs1: _i(0b101,rd,rs1),
    "sra":  lambda rd,rs1: _i(0b101,rd,rs1,imm=0x400),
    "slt":  lambda rd,rs1: _i(0b010,rd,rs1),
    "sltu": lambda rd,rs1: _i(0b011,rd,rs1),
}
_LOAD  = {"word": lambda rd,rs1: _load(0b010,rd,rs1),
           "half-word": lambda rd,rs1: _load(0b001,rd,rs1),
           "byte": lambda rd,rs1: _load(0b000,rd,rs1)}
_STORE = {"word": lambda rs1,rs2: _store(0b010,rs1,rs2),
           "half-word": lambda rs1,rs2: _store(0b001,rs1,rs2),
           "byte": lambda rs1,rs2: _store(0b000,rs1,rs2)}

_ALU_IMM_SUB = _OP_KEY[("alu_imm", "sub")]
_ILLEGAL     = 0x00000000


def structured_to_word(entry):
    op_idx = int(entry[0]); rd = int(entry[1]); rs1 = int(entry[2]); rs2 = int(entry[3])
    if op_idx == _ALU_IMM_SUB:
        return _ILLEGAL
    kind, name = OP_TYPES[op_idx]
    if kind == "alu":        return _R[name](rd, rs1, rs2)
    elif kind == "alu_imm":  return _I[name](rd, rs1)
    elif kind == "load":     return _LOAD[name](rd, rs1)
    elif kind == "store":    return _STORE[name](rs1, rs2)
    return _ILLEGAL


def cov_to_bin_names(cov: Coverage) -> set:
    """Convert Coverage object to set of shadow_decoder bin name strings."""
    bins = set()
    sz_letter = {"word": "W", "half-word": "H", "byte": "B"}

    for name, cnt in cov.alu_ops.items():
        if cnt > 0: bins.add(f"ALU_{name.upper()}")
    for name, cnt in cov.alu_imm_ops.items():
        if cnt > 0 and name != "sub": bins.add(f"ALUI_{name.upper()}I")
    if cov.illegal_count > 0: bins.add("illegal_instruction")
    for sz, cnt in cov.load_ops.items():
        if cnt > 0: bins.add(f"L{sz_letter[sz]}")
    for sz, cnt in cov.store_ops.items():
        if cnt > 0: bins.add(f"S{sz_letter[sz]}")

    for i, cnt in enumerate(cov.read_reg_a):
        if cnt > 0: bins.add(f"read_A_reg_{i}")
    for i, cnt in enumerate(cov.read_reg_b):
        if cnt > 0: bins.add(f"read_B_reg_{i}")
    for i, cnt in enumerate(cov.write_reg):
        if cnt > 0: bins.add(f"write_reg_{i}")

    for op, regs in cov.alu_x_ra.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"{op.upper()}_x_read_A_reg_{i}")
    for op, regs in cov.alu_x_rb.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"{op.upper()}_x_read_B_reg_{i}")
    for op, regs in cov.alu_x_wr.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"{op.upper()}_x_write_reg_{i}")
    for op, regs in cov.imm_x_ra.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"{op.upper()}I_x_read_A_reg_{i}")
    for op, regs in cov.imm_x_wr.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"{op.upper()}I_x_write_reg_{i}")
    for sz, regs in cov.load_x_ra.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"L{sz_letter[sz]}_x_read_A_reg_{i}")
    for sz, regs in cov.load_x_wr.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"L{sz_letter[sz]}_x_write_reg_{i}")
    for sz, regs in cov.store_x_ra.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"S{sz_letter[sz]}_x_read_A_reg_{i}")
    for sz, regs in cov.store_x_rb.items():
        for i, cnt in enumerate(regs):
            if cnt > 0: bins.add(f"S{sz_letter[sz]}_x_read_B_reg_{i}")
    return bins


@cocotb.test()
async def rl_validation_decoder(dut):
    t_start = time.time()
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    mode       = payload.get("mode", "structured")
    shadow_hit = set(payload["shadow_hit_bins"])

    if mode == "raw":
        program = [int(w) for w in payload["program"]]
    else:
        program = [structured_to_word(e) for e in payload["sequence"]]

    print(f"\nMode: {mode} | Instructions: {len(program)} | Shadow predicts: {len(shadow_hit)} bins")

    cov = Coverage()

    async def apply(word):
        dut.insn_i.value = word
        await Timer(1, units='ns')

        illegal = int(dut.u_decoder.illegal_insn_o.value)
        if illegal:
            cov.illegal_count += 1
            return

        rf_we      = int(dut.u_decoder.rf_we_o.value)
        rf_ren_a   = int(dut.u_decoder.rf_ren_a_o.value)
        rf_ren_b   = int(dut.u_decoder.rf_ren_b_o.value)
        rf_raddr_a = int(dut.u_decoder.rf_raddr_a_o.value)
        rf_raddr_b = int(dut.u_decoder.rf_raddr_b_o.value)
        rf_waddr   = int(dut.u_decoder.rf_waddr_o.value)
        data_req   = int(dut.u_decoder.data_req_o.value)
        data_we    = int(dut.u_decoder.data_we_o.value)
        data_type  = int(dut.u_decoder.data_type_o.value)
        op_b_mux   = int(dut.u_decoder.alu_op_b_mux_sel_o.value)
        op_a_mux   = int(dut.u_decoder.alu_op_a_mux_sel_o.value)
        rf_wdata_sel = int(dut.u_decoder.rf_wdata_sel_o.value)
        mult_sel   = int(dut.u_decoder.mult_sel_o.value)
        div_sel    = int(dut.u_decoder.div_sel_o.value)
        alu_op_val = int(dut.u_decoder.alu_operator_o.value)

        rs1 = rf_raddr_a if rf_ren_a else None
        rs2 = rf_raddr_b if rf_ren_b else None
        rd  = rf_waddr   if rf_we    else None

        if rs1 is not None: cov.read_reg_a[rs1] += 1
        if rs2 is not None: cov.read_reg_b[rs2] += 1

        size_map = {0: "word", 1: "half-word", 2: "byte"}
        alu_name_map = {0:"add",1:"sub",2:"xor",3:"or",4:"and",
                        8:"sra",9:"srl",10:"sll",43:"slt",44:"sltu"}

        if data_req:
            sz = size_map.get(data_type)
            if sz:
                if data_we:
                    cov.store_ops[sz] += 1
                    if rs1 is not None: cov.store_x_ra[sz][rs1] += 1
                    if rs2 is not None: cov.store_x_rb[sz][rs2] += 1
                else:
                    rd = rf_waddr
                    cov.write_reg[rd] += 1
                    cov.load_ops[sz] += 1
                    if rs1 is not None: cov.load_x_ra[sz][rs1] += 1
                    cov.load_x_wr[sz][rd] += 1
            return

        if rd is not None: cov.write_reg[rd] += 1

        alu_name = alu_name_map.get(alu_op_val)
        if alu_name and rf_we and not mult_sel and not div_sel and rf_ren_a \
                and op_a_mux == 0 and rf_wdata_sel == 0:
            if op_b_mux == 1:  # immediate
                cov.alu_imm_ops[alu_name] += 1
                if rs1 is not None: cov.imm_x_ra[alu_name][rs1] += 1
                if rd  is not None: cov.imm_x_wr[alu_name][rd]  += 1
            else:
                cov.alu_ops[alu_name] += 1
                if rs1 is not None: cov.alu_x_ra[alu_name][rs1] += 1
                if rs2 is not None: cov.alu_x_rb[alu_name][rs2] += 1
                if rd  is not None: cov.alu_x_wr[alu_name][rd]  += 1

    for word in program:
        await apply(word)

    real_hit   = cov_to_bin_names(cov)
    only_shadow = shadow_hit - real_hit
    only_real   = real_hit  - shadow_hit

    print("\n" + "=" * 70)
    print(f" SHADOW vs REAL DECODER RTL  [{mode.upper()}]")
    print("=" * 70)
    print(f"  Shadow predicted:  {len(shadow_hit):4d} / {N_BINS}")
    print(f"  Real RTL observed: {len(real_hit):4d} / {N_BINS}")
    print(f"  Agreement:         {len(shadow_hit & real_hit):4d} bins")
    print(f"\n  Only in shadow (overpredicts): {len(only_shadow)}")
    for b in sorted(only_shadow)[:10]: print(f"    - {b}")
    if len(only_shadow) > 10: print(f"    ...and {len(only_shadow)-10} more")
    print(f"\n  Only in real (underpredicts): {len(only_real)}")
    for b in sorted(only_real)[:10]: print(f"    + {b}")
    if len(only_real) > 10: print(f"    ...and {len(only_real)-10} more")

    print("\n" + "=" * 70)
    if shadow_hit == real_hit:
        print(f" VALIDATION PASSED: {len(real_hit)}/{N_BINS} bins — shadow matches RTL exactly.")
    else:
        pct = 100 * len(shadow_hit & real_hit) / max(len(shadow_hit | real_hit), 1)
        status = "close" if pct > 90 else "diverging"
        print(f" Overlap = {pct:.1f}% (of union). Shadow is {status}.")
    print(f" RTL validation time: {time.time()-t_start:.1f}s")
    print("=" * 70)
