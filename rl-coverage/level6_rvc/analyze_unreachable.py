"""Diagnose why we plateau at ~56% toggle coverage.

For each of the biggest coverage-gap modules, print a sample of uncovered
point names (signal-level). We annotate whether each signal is likely:
  * reachable-with-more-stimulus (e.g. a mul-divider operand bit we haven't hit)
  * unreachable-in-this-config   (e.g. PMP/ICache/debug signal tied off)

This lets us judge whether pushing past 56% is an achievable RL goal or whether
the hard ceiling is set by the minimal-config build.
"""

import sys, re
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
L5_DIR = THIS.parent / "level5_real_rtl"
sys.path.insert(0, str(L5_DIR))
import cov_parser


COVDAT = "../../cpu/coverage.dat"


# Heuristic tags for common Ibex signal names.
# Signals that can't toggle given our minimal config.
# Grouped by why: config parameter that disables the logic driving them.
TIED_OFF_SUBSTRINGS = [
    # PMPEnable = 0
    "pmp_", "csr_pmp", "mseccfg",
    # ICache = 0
    "icache", "ic_data", "ic_tag", "ic_scr", "ic_",
    # DbgTriggerEn = 0
    "debug_req", "dbg_", "trigger_match", "tselect", "tdata",
    "depc", "dcsr", "dscratch", "mstack_", "mcontext",
    # irq inputs tied to 0
    "irq_", "mip", "nmi_",
    # SecureIbex = 0
    "scramble", "scramble_key", "dummy_instr",
    # WritebackStage = 0
    "wb_stage", "_wb_", "rf_write_wb", "writeback_",
    # BranchPredictor = 0
    "branch_predict", "nt_branch", "predict_",
    # ECC / SECDED (unused in our config)
    "secded", "_ecc", "ecc_", "intg_",
    # MHPMCounterNum = 0  -> performance counters tied off
    "mhpmcounter", "mhpmevent", "hpm_",
    # RVFI interface — driven internally but many fanout wires are unused
    "rvfi",
    # RV32B = None -> bit-manipulation datapath not exercised
    "butterfly_result", "minmax_result", "shift_result_rev",
    "singlebit_result", "bitcnt_", "bitfield_",
    "imd_val_q_i[1]",  # the second operand of B-ext state (RV32B-only)
    # MHPMCounterNum / counter width for unused counters
    "counter[3", "counter[4", "counter[5", "counter[6",  # bits > 32 are tied 0
    # Alerts/lockstep/fpga and other top-level unused
    "lockstep", "alert_", "fpga", "core_sleep",
    "fetch_enable", "ram_cfg", "scan_rst",
    "boot_addr",  # fixed constant
    "hart_id",    # fixed to 0
]

# Signals that ARE reachable but need specific stimulus we don't currently emit.
REACHABLE_BUT_NEEDS = [
    # Needs ECALL / EBREAK / exception
    (["mepc", "mtval", "mcause", "mstatus", "mtvec_q"],
     "needs ECALL / exception path"),
    # Needs AUIPC
    (["imm_u_type_o"], "needs AUIPC instruction"),
    # Needs wider MUL/DIV operand coverage (the first operand's state bits)
    (["imd_val_q_i[0]"], "needs MUL/DIV operand diversity"),
]

REACHABLE_HINTS = [
    ("csr_", "CSR diversity — could be unlocked with broader CSR writes"),
    ("multdiv", "MUL/DIV operand bits — could be unlocked with wider value coverage"),
    ("decoder", "decoder path — could be unlocked by specific instruction encodings"),
    ("controller", "controller state — specific sequences (exception, wfi)"),
    ("id_stage", "ID-stage data — depends on instruction mix"),
]


def classify(signal_name: str, point_key: str) -> str:
    hay_lc = signal_name.lower()
    full_lc = point_key.lower()
    for sub in TIED_OFF_SUBSTRINGS:
        if sub.lower() in hay_lc or sub.lower() in full_lc:
            return f"TIED-OFF ({sub})"
    for subs, label in REACHABLE_BUT_NEEDS:
        for sub in subs:
            if sub.lower() in hay_lc:
                return f"NEEDS ({label})"
    return "REACHABLE?"


def parse_point(key: str) -> tuple[str, str, str]:
    """Extract (module, signal, extra) from a cov_parser point key."""
    # Key format: \x01page\x02v_toggle/<mod>\x01<kv>\x01<kv>... roughly.
    page_m = re.search(r"page\x02([^\x01]+)", key)
    sig_m  = re.search(r"\x01o\x02([^\x01]+)", key)
    line_m = re.search(r"\x01l\x02([^\x01]+)", key)
    page = page_m.group(1) if page_m else "?"
    sig  = sig_m.group(1)  if sig_m  else "?"
    line = line_m.group(1) if line_m else "?"
    return page, sig, line


def main():
    s = cov_parser.parse(COVDAT)
    print(f"Loaded coverage.dat  ({len(s.points)} points total)")
    print(f"Overall toggle: {s.by_kind['toggle'][0]:>5}/{s.by_kind['toggle'][1]:<5} "
          f"= {s.kind_pct('toggle'):.2f}%\n")

    # Collect uncovered toggle points per module
    by_module = defaultdict(list)
    tag_totals = defaultdict(int)
    for key, count in s.points.items():
        if count > 0:
            continue
        page, sig, line = parse_point(key)
        if not page.startswith("v_toggle/"):
            continue
        module = page[len("v_toggle/"):].split("__")[0]
        tag = classify(sig, key)
        by_module[module].append((tag, sig, line))
        # coarse tag bucket
        bucket = "TIED-OFF" if tag.startswith("TIED") else \
                 "NEEDS"    if tag.startswith("NEEDS") else "REACHABLE?"
        tag_totals[bucket] += 1

    print("=== Per-module uncovered signal breakdown ===")
    rows = []
    for m, pts in by_module.items():
        tied = sum(1 for (t, _, _) in pts if t.startswith("TIED"))
        needs = sum(1 for (t, _, _) in pts if t.startswith("NEEDS"))
        reach = len(pts) - tied - needs
        rows.append((len(pts), tied, needs, reach, m))
    rows.sort(reverse=True)
    print(f"{'module':<30s} {'uncov':>6s} {'tied':>6s} {'needs':>6s} {'?':>6s}")
    print("-" * 60)
    gt = gn = gr = 0
    for uncov, tied, needs, reach, m in rows[:20]:
        print(f"{m:<30s} {uncov:>6d} {tied:>6d} {needs:>6d} {reach:>6d}")
        gt += tied; gn += needs; gr += reach
    remaining = sum(r[0] for r in rows[20:])
    if remaining:
        print(f"{'… other modules':<30s} {remaining:>6d}")

    print(f"\nTotals across all uncovered toggle points:")
    print(f"  TIED-OFF  (unreachable in this config):   {tag_totals['TIED-OFF']}")
    print(f"  NEEDS     (reachable with known stimulus): {tag_totals['NEEDS']}")
    print(f"  REACHABLE? (likely reachable, unclear):   {tag_totals['REACHABLE?']}")
    total_uncov = tag_totals['TIED-OFF'] + tag_totals['NEEDS'] + tag_totals['REACHABLE?']
    print(f"  TOTAL uncovered toggle:                    {total_uncov}")
    overall_total = s.by_kind['toggle'][1]
    reachable_ceiling = 100.0 * (overall_total - tag_totals['TIED-OFF']) / overall_total
    current_pct = 100.0 * s.by_kind['toggle'][0] / overall_total
    print(f"\n  Hard ceiling (tied-off excluded):    {reachable_ceiling:.2f}%")
    print(f"  Current single-run coverage:          {current_pct:.2f}%")
    print(f"  Room for improvement:                 {reachable_ceiling - current_pct:.2f} points")

    # Show 20 random "REACHABLE?" signals as sanity
    print("\n=== Sample of REACHABLE? uncovered signals (what RL could aim for) ===")
    import random
    rng = random.Random(0)
    reachable = []
    for m, pts in by_module.items():
        for tag, sig, line in pts:
            if not tag.startswith("TIED"):
                reachable.append((m, sig, line))
    rng.shuffle(reachable)
    for m, sig, line in reachable[:25]:
        print(f"  {m:<25s}  line {line:<5s}  {sig}")

    print("\n=== Sample of TIED-OFF (likely unreachable) ===")
    tied_off = []
    for m, pts in by_module.items():
        for tag, sig, line in pts:
            if tag.startswith("TIED"):
                tied_off.append((m, sig, tag))
    rng.shuffle(tied_off)
    for m, sig, tag in tied_off[:15]:
        print(f"  {m:<25s}  {sig:<40s}  {tag}")


if __name__ == "__main__":
    main()
