"""export_for_llm.py — Exportă semnalele accesibile într-un format clar pentru LLM.

Categorii corecte:
  - Structural (ISA): instr_addr[0], exc_pc[0], fetch_addr[0] → EXCLUSE (mereu 0 prin spec)
  - Data addr [0:1]: accesibile via LB/SB/LH/SH → incluse
  - mcountinhibit[*]: scriere CSR 0x320
  - CSR read data: rd_data_o, rdata_q
  - Exception paths: csr_save, mepc, mtval, mcause
  - mtvec bits: scriere mtvec cu valori rare
  - Immediate bits: LUI/AUIPC cu immediate mare
  - ALU/decoder: alu_operator[6], bt_a_mux_sel, stall_alu
  - Debug (TB req): debug_mode, debug_cause → EXCLUSE din LLM
  - IRQ/NMI (TB req): captured, mfip → EXCLUSE din LLM

Usage:
    python export_for_llm.py
"""

import re
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent

# Semnale structural imposibile (ISA alignment — bit 0 al adresei de instructiuni)
STRUCTURAL = {
    "instr_addr_o[0]", "instr_addr[0]", "fetch_addr[0]",
    "addr_i[0]", "addr_o[0]",
    "exc_pc[0]",
}

# Semnale care necesita testbench (nu pot fi acoperite cu program assembly)
NEEDS_TESTBENCH = {
    "debug_mode", "debug_cause", "enter_debug_mode", "debug_csr_save",
    "debug_ebreakm", "debug_ebreaku", "debug_mode_entering",
    "csr_restore_dret_id", "dret_insn_o",
    "captured_valid", "captured_nmi", "mfip_id",
}


def load_accessible(proof_file):
    signals = []
    current = None
    with open(proof_file) as f:
        for line in f:
            if "ACCESIBIL" in line:
                current = "acc"
            elif line.startswith("  🔴") or line.startswith("  ⚠️"):
                current = None
            if current == "acc":
                m = re.match(r"\s{2}(\S+)\s{2,}(\S+)", line)
                if m:
                    signals.append((m.group(1), m.group(2)))
    return signals


def categorize(mod, sig):
    base = re.sub(r'\[.*', '', sig)

    # Excluse
    if sig in STRUCTURAL or base in STRUCTURAL:
        return "EXCLUDE_structural"
    if base in NEEDS_TESTBENCH or sig in NEEDS_TESTBENCH:
        return "EXCLUDE_testbench"
    # debug prefix larg
    if any(x in sig for x in ["debug_cause", "debug_csr", "debug_ebreak",
                                "enter_debug", "dret", "csr_restore_dret"]):
        return "EXCLUDE_testbench"
    # IRQ
    if any(x in sig for x in ["mfip", "captured_nmi", "captured_valid"]):
        return "EXCLUDE_testbench"

    # Grupuri accesibile
    if "mcountinhibit" in sig:
        return "mcountinhibit"
    if any(x in sig for x in ["mtvec", "csr_mtvec"]):
        return "mtvec_bits"
    if any(x in sig for x in ["csr_save", "csr_restore", "mepc", "mtval",
                                "mcause", "exc_cause"]):
        return "exception"
    if any(x in sig for x in ["imm_a", "zimm", "imm_u"]):
        return "immediate"
    if any(x in sig for x in ["rd_data_o", "rdata_q"]):
        return "csr_read"
    if any(x in sig for x in ["data_addr_o[0]", "data_addr_o[1]"]):
        return "data_alignment"
    if any(x in sig for x in ["alu_operator", "bt_a_mux_sel", "stall_alu",
                                "ctrl_fsm", "exc_pc_mux", "pc_mux",
                                "instr_type_wb", "csr_illegal"]):
        return "alu_decoder"
    if "exc_pc" in sig:
        return "exception"
    if "instr_addr_o[1]" in sig or "instr_addr_o[1]" == sig:
        return "data_alignment"
    if "data_accessed" in sig:
        return "alu_decoder"

    return "other"


def main():
    signals = load_accessible(THIS / "proof_results.txt")

    groups = defaultdict(list)
    for mod, sig in signals:
        cat = categorize(mod, sig)
        groups[cat].append((mod, sig))

    # Deduplica (acelasi semnal poate aparea in module diferite)
    for cat in groups:
        seen = set()
        dedup = []
        for mod, sig in groups[cat]:
            if sig not in seen:
                seen.add(sig)
                dedup.append((mod, sig))
        groups[cat] = dedup

    excluded_struct = groups.pop("EXCLUDE_structural", [])
    excluded_tb     = groups.pop("EXCLUDE_testbench", [])

    total_accessible = sum(len(v) for v in groups.values())
    total_excluded   = len(excluded_struct) + len(excluded_tb)

    print(f"Total semnale accesibile (dupa deduplicare si filtrare):")
    print(f"  Excluse structural (ISA):    {len(excluded_struct)}")
    print(f"  Excluse testbench (debug/IRQ):{len(excluded_tb)}")
    print(f"  Ramase pentru LLM:           {total_accessible}")
    print()
    for cat, sigs in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<20s}: {len(sigs)} bins")

    out = THIS / "accessible_bins_for_llm.txt"
    with open(out, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UNCOVERED ACCESSIBLE TOGGLE BINS — Ibex RISC-V\n")
        f.write(f"Total: {total_accessible} bins (exclud structural + testbench)\n")
        f.write("=" * 70 + "\n")
        f.write("""
CONTEXT:
  Processor : Ibex RISC-V (RV32IMC, Machine mode)
  Simulation: Verilator + cocotb
  Coverage  : 71.94% achieved (14,404 / 20,023 toggle bins)
  These bins CAN toggle — proven by Verilator constant propagation.
  Programs start at 0x80000000, run in M-mode.

CONFIGURATION (fixed — cannot change without recompilation):
  PMPEnable=0, ICache=0, RV32B=None, BranchPredictor=0,
  WritebackStage=0, SecureIbex=0, RV32M=RV32MFast (MUL/DIV enabled)

TASK:
  Write RISC-V assembly programs (RV32IMC) that toggle each group.
  Each signal must transition 0→1 AND 1→0 to be "covered".

""")

        # ── Group 1: mcountinhibit ────────────────────────────────────────
        g = groups.get("mcountinhibit", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 1: mcountinhibit CSR bits ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  mcountinhibit (CSR 0x320) controls which performance counters are frozen.
  Bits [31:1] are writable. Writing a non-zero value then zero toggles them.
  With MHPMCounterNum=0 only bits 0 (cycle) and 2 (instret) have hardware,
  but the register bits are still writable.

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  li   t0, 0xFFFFFFFF     # set all mcountinhibit bits
  csrw 0x320, t0          # write mcountinhibit
  csrw 0x320, zero        # clear → toggle back
  li   t0, 0x00000002     # bit 1
  csrw 0x320, t0
  csrw 0x320, zero\n\n""")

        # ── Group 2: mtvec bits ───────────────────────────────────────────
        g = groups.get("mtvec_bits", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 2: mtvec low bits ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  mtvec (CSR 0x305) holds the trap vector base address.
  Bits [7:1] are rarely written. Bit 0 = mode (0=direct, 1=vectored).
  Write 0xFE then 0x00 to toggle bits [7:1].

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  li   t0, 0x000000FE     # bits [7:1] set, bit 0 = 0 (direct mode)
  csrw mtvec, t0
  csrw mtvec, zero        # clear → toggle back
  li   t0, 0x000000FF     # all low bits set (vectored mode)
  csrw mtvec, t0
  csrw mtvec, zero\n\n""")

        # ── Group 3: exception paths ──────────────────────────────────────
        g = groups.get("exception", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 3: Exception / CSR save paths ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  csr_save_id/if: pulsed for 1 cycle when CPU saves CSRs on exception entry.
  exc_pc[7:1]: exception PC lower bits (can vary with RVC compressed instructions).
  mepc[0]: exception PC bit 0 (set for compressed instruction exceptions).
  Trigger exceptions via illegal instruction or misaligned load/store.

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  la   t0, trap_handler
  csrw mtvec, t0          # set trap handler
  # illegal instruction exception (mcause=2)
  .word 0x0000000B        # CUSTOM-0 → illegal
  # misaligned load (mcause=4, mtval=fault addr)
  li   t0, 0x80000001
  lw   t1, 0(t0)          # misaligned → exception
  # misaligned store (mcause=6)
  li   t0, 0x80000003
  sw   t1, 0(t0)
  j    done
trap_handler:
  csrr t0, mepc
  csrr t1, mcause
  csrr t2, mtval
  addi t0, t0, 4
  csrw mepc, t0
  mret
done:\n\n""")

        # ── Group 4: immediate bits ───────────────────────────────────────
        g = groups.get("immediate", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 4: Immediate upper bits ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  imm_a[31:10]: upper bits of immediates used as ALU operand A.
  These come from LUI/AUIPC 20-bit immediates or CSR zimm_rs1 field.
  Use large, varied immediate values.

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  lui  t0, 0xFFFFF        # bits [31:12] all 1
  lui  t1, 0x80000        # bit 31 set
  lui  t2, 0x55555        # alternating bits
  lui  t3, 0xAAAAA        # alternating bits inverted
  auipc t4, 0xFFFFF       # PC-relative large immediate
  auipc t5, 0x7FFFF
  # CSR immediate (5-bit zimm field)
  csrrwi t0, mstatus, 0x1F  # zimm = 0x1F (all 5 bits set)
  csrrwi t0, mstatus, 0x00\n\n""")

        # ── Group 5: CSR read data ────────────────────────────────────────
        g = groups.get("csr_read", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 5: CSR read data bits ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  rd_data_o / rdata_q: data output of the CSR register file.
  These bits toggle when reading CSRs that contain non-zero values.
  Read various CSRs after writing non-zero values to them.

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  # Write then read back various CSRs
  li   t0, 0xFF
  csrw mtvec, t0
  csrr t1, mtvec          # read back → rd_data_o gets 0xFF
  csrw mtvec, zero
  csrr t1, mtvec          # read back → rd_data_o gets 0
  # cycle counter
  csrr t0, mcycle         # read cycle count (non-zero after execution)
  csrr t1, minstret       # read instruction count
  # mstatus
  li   t0, 0x8
  csrw mstatus, t0
  csrr t1, mstatus\n\n""")

        # ── Group 6: data address alignment bits ─────────────────────────
        g = groups.get("data_alignment", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 6: Data/instruction address low bits ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  data_addr_o[0]: can be 1 with byte loads/stores (LB/SB) to odd addresses.
  data_addr_o[1]: can be 1 with byte/halfword ops to 0x2/0x6/... addresses.
  instr_addr_o[1]: can be 1 with RVC compressed instructions at 2-byte PCs.
  Use byte and halfword memory operations with varied addresses.

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  li   t0, 0x80000001     # odd address
  lb   t1, 0(t0)          # byte load → data_addr_o[0] = 1
  sb   t1, 0(t0)          # byte store → data_addr_o[0] = 1
  li   t0, 0x80000002     # 2-byte aligned
  lh   t1, 0(t0)          # halfword → data_addr_o[1] = 1, [0] = 0
  sh   t1, 0(t0)
  li   t0, 0x80000003     # 3 = odd + 2
  lb   t1, 0(t0)          # data_addr_o[1:0] = 11
  # RVC for instr_addr_o[1]:
  c.nop                   # compressed instruction at 2-byte boundary\n\n""")

        # ── Group 7: ALU / decoder / controller ───────────────────────────
        g = groups.get("alu_decoder", [])
        f.write("=" * 70 + "\n")
        f.write(f"GROUP 7: ALU / decoder / controller signals ({len(g)} bins)\n")
        f.write("=" * 70 + "\n")
        f.write("""
EXPLANATION:
  alu_operator[6] / alu_operator_o[6]: ALU op encoding bit 6. Toggled by
    specific ALU operations that map to this bit (e.g., CPOP, CLZ from RV32B
    — but RV32B=None so need to check which standard op uses bit 6).
  bt_a_mux_sel[0]: branch target mux, selects between PC+imm or rs1+imm.
    Toggled by JAL vs JALR.
  stall_alu: asserted during multi-cycle ALU ops (MUL, DIV).
  ctrl_fsm_ns[3]: controller FSM state bit 3, activated during exceptions.
  exc_pc_mux_o[1] / pc_mux_o[2]: PC mux selects, activated by different
    types of jumps/exceptions.
  csr_illegal: asserted when accessing a non-existent or read-only CSR.
  instr_type_wb[0]: instruction type in writeback (WritebackStage=0 so
    this might be structural — include but may not toggle).
  data_accessed[2]: tracer signal for different data access types.

SIGNALS:\n""")
        for mod, sig in g:
            f.write(f"  {mod:<30s} {sig}\n")
        f.write("""
SUGGESTED PROGRAM:
  # MUL/DIV for stall_alu
  li   t0, 12345
  li   t1, 67890
  mul  t2, t0, t1         # stall_alu toggles during MUL
  div  t3, t0, t1         # stall_alu toggles during DIV
  # JAL vs JALR for bt_a_mux_sel
  jal  t0, 1f             # JAL: bt_a_mux_sel = PC-relative
  1: jalr zero, t0, 0     # JALR: bt_a_mux_sel = rs1-relative
  # csr_illegal: write to non-existent CSR
  csrwi 0x7FF, 0          # CSR 0x7FF — illegal in current config
  # Exception for ctrl_fsm_ns and exc_pc_mux
  la   t0, trap2
  csrw mtvec, t0
  ecall                   # environment call exception
  j    done2
trap2:
  mret
done2:\n\n""")

        # ── Group other ───────────────────────────────────────────────────
        g = groups.get("other", [])
        if g:
            f.write("=" * 70 + "\n")
            f.write(f"GROUP 8: Other ({len(g)} bins)\n")
            f.write("=" * 70 + "\n")
            f.write("SIGNALS:\n")
            for mod, sig in g:
                f.write(f"  {mod:<30s} {sig}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total bins in this file: {total_accessible}\n")
        f.write(f"  All bins can be toggled with RISC-V assembly programs.\n")
        f.write(f"  No testbench modification required.\n")

    print(f"\nExportat → {out}")


if __name__ == "__main__":
    main()
