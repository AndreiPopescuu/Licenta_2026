"""targeted_reachable_test.py

Verifică empiric fiecare semnal REACHABLE? rămas neacoperit:
  - generează programe țintite pentru fiecare categorie de semnal
  - rulează prin Vtop
  - raportează care semnale SE togglează (realmente reachable)
    și care NU (efectiv tied-off, heuristicile le-au clasificat greșit)

Usage:
    python targeted_reachable_test.py --hits random_baseline_hits.pkl
    python targeted_reachable_test.py --hits random_baseline_hits.pkl --covdat ../../cpu/coverage.dat
"""

import sys, re, pickle, argparse, time
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
L5   = THIS.parent / "level5_real_rtl"
L7   = THIS.parent / "level7_stimulus"
sys.path.insert(0, str(L5))

import cov_parser
from env_l9_v2 import run_program, N_OPS, IMM_BUCKETS

# Importăm explicit din level7_stimulus ca să nu prindă level6_rvc
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "analyze_unreachable_l7",
    str(L7 / "analyze_unreachable.py"),
)
_ar = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ar)
classify                  = _ar.classify
_COUNTER_HIGH_BIT_RE      = _ar._COUNTER_HIGH_BIT_RE
_COUNTER_HIGH_THRESHOLD   = _ar._COUNTER_HIGH_THRESHOLD

# ── Op indices (din codec_l9) ────────────────────────────────────────────────
ADD=0; SUB=1; SLL=2; SLT=3; SLTU=4; XOR=5; SRL=6; SRA=7; OR=8; AND=9
ADDI=10; SLTI=11; SLTIU=12; XORI=13; ORI=14; ANDI=15
SLLI=16; SRLI=17; SRAI=18
LB=19; LH=20; LW=21; LBU=22; LHU=23
SB=24; SH=25; SW=26
CSRRW=27; CSRRS=28; CSRRC=29
MUL=30; MULH=31; MULHSU=32; MULHU=33
DIV=34; DIVU=35; REM=36; REMU=37
BEQ=38; BNE=39; BLT=40; BGE=41; BLTU=42; BGEU=43
JAL=44
AUIPC=61; ECALL=62; EBREAK=63; LUI=64; JALR=65
FENCE=69; MRET=70; WFI=71; FENCE_I=72
NOP = (ADDI, 0, 0, 0, 2)   # addi x0, x0, 0


# ── Programe țintite ──────────────────────────────────────────────────────────

def prog_sext_result(n=200):
    """Sign-extension: LUI valoare negativă + SRAI cu diverse amounts."""
    acts = []
    for rd in range(1, 8):
        # LUI rd, 0xFFFFF → rd = 0xFFFFF000 (negativ, bit31=1)
        acts.append((LUI, rd, 0, 0, 4))          # imm_bucket 4 = 0xFFFFF
        # SRAI rd+8, rd, shamt — shamt buckets: 0,1,7,15,31
        for ib in range(IMM_BUCKETS):
            acts.append((SRAI, (rd+8)%32, rd, 0, ib))
        # SRA cu valoare negativă în rs2
        acts.append((LUI, 16, 0, 0, 3))          # imm_bucket 3 = 0xABCDE
        acts.append((SRA, rd, rd, 16, 0))
        # SRLI pe valoare negativă → zero-extend (bitul 31 devine 0)
        for ib in range(IMM_BUCKETS):
            acts.append((SRLI, (rd+16)%32, rd, 0, ib))
    return (acts * ((n // len(acts)) + 1))[:n]


def prog_branch_target(n=200):
    """Branch target operand bits: branches cu diverse offsets și PC-uri."""
    acts = []
    # Umple registrele cu valori diverse
    for rd in range(1, 16):
        acts.append((LUI, rd, 0, 0, rd % IMM_BUCKETS))
        acts.append((ADDI, rd, rd, 0, rd % IMM_BUCKETS))
    # Branches cu toate combinațiile
    for op in [BEQ, BNE, BLT, BGE, BLTU, BGEU]:
        for rs1 in range(1, 6):
            for rs2 in range(1, 6):
                for ib in range(IMM_BUCKETS):
                    acts.append((op, 0, rs1, rs2, ib))
    # JAL cu diverse offsets
    for ib in range(IMM_BUCKETS):
        acts.append((JAL, 1, 0, 0, ib))
        acts.append((AUIPC, 2, 0, 0, ib))
        acts.append((JALR, 0, 2, 0, ib))
    return (acts * ((n // max(len(acts), 1)) + 1))[:n]


def prog_csr_values(n=200):
    """CSR value bits: scrie valori diverse în CSR-uri (mtvec, mstatus etc.)."""
    acts = []
    # Încarcă valori cu biți diverși setați
    lui_imms = [0, 1, 2, 3, 4]  # LUI_IMM_BUCKETS: 0x00001, 0x12345, 0xABCDE, 0xFFFFF, 0x80000
    for ib in lui_imms:
        acts.append((LUI, 5, 0, 0, ib))
        acts.append((LUI, 6, 0, 0, (ib+1)%5))
        # CSRRW cu registre cu valori mari
        for rs1 in range(5, 10):
            for csr_ib in range(IMM_BUCKETS):
                acts.append((CSRRW, 0, rs1, 0, csr_ib))
                acts.append((CSRRS, 0, rs1, 0, csr_ib))
        # Combină cu ADDI pentru biți mai specifici
        for k in range(4):
            acts.append((ADDI, 7, 5, 0, k))
            acts.append((CSRRW, 0, 7, 0, k))
    return (acts * ((n // max(len(acts), 1)) + 1))[:n]


def prog_counter_bits(n=1024):
    """Counter high bits: program lung de NOP-uri ca să avanseze contoarele."""
    # Bit K necesită 2^K cicluri; 1024 instrucțiuni ≈ 2000-4000 cicluri → bit 11-12
    return [NOP] * n


def prog_mul_div_operands(n=200):
    """MUL/DIV operand bits: operanzi mari cu biți înalți setați."""
    acts = []
    for ib in range(IMM_BUCKETS):
        acts.append((LUI, 10, 0, 0, ib))     # valori mari
        acts.append((LUI, 11, 0, 0, (ib+2)%5))
        acts.append((ADDI, 10, 10, 0, ib))
        acts.append((ADDI, 11, 11, 0, ib))
        for op in [MUL, MULH, MULHSU, MULHU]:
            acts.append((op, 12, 10, 11, 0))
        for op in [DIV, DIVU, REM, REMU]:
            for rs2 in [10, 11]:
                acts.append((op, 13, rs2, 10 if rs2==11 else 11, 0))
    # SUB pentru valori negative
    acts.append((LUI, 14, 0, 0, 4))  # 0xFFFFF000
    acts.append((LUI, 15, 0, 0, 3))  # 0xABCDE000
    for op in [MUL, MULH, DIV]:
        acts.append((op, 16, 14, 15, 0))
        acts.append((op, 16, 15, 14, 0))
    return (acts * ((n // max(len(acts), 1)) + 1))[:n]


def prog_load_store_addr(n=200):
    """Load/store address bits: diverse adrese pentru biții adresei."""
    acts = []
    for ib in range(IMM_BUCKETS):
        acts.append((LUI, 20, 0, 0, ib))
        acts.append((ADDI, 20, 20, 0, ib))
        for op in [LB, LH, LW, LBU, LHU]:
            acts.append((op, 21, 20, 0, ib))
        for op in [SB, SH, SW]:
            acts.append((op, 0, 20, 21, ib))
    # Misaligned accesses (din L10)
    try:
        from codec_l10 import Op as L10Op, N_OPS as L10_N_OPS
        ILLEGAL_INSN = int(L10Op.ILLEGAL_INSN) if hasattr(L10Op, 'ILLEGAL_INSN') else None
        LW_MISALIGN  = int(L10Op.LW_MISALIGN)  if hasattr(L10Op, 'LW_MISALIGN')  else None
        if LW_MISALIGN:
            acts += [(LW_MISALIGN, 0, 0, 0, 0)] * 5
    except ImportError:
        pass
    return (acts * ((n // max(len(acts), 1)) + 1))[:n]


def prog_alu_imd_val(n=200):
    """Verifică dacă alu_imd_val_d[1] poate fi toggleat (RV32B check).
    Dacă NU se togglează cu NICIO instrucțiune → e definitiv RV32B.
    """
    # Încearcă orice secvență care ar putea activa un path multicycle în ALU
    acts = []
    for ib in range(IMM_BUCKETS):
        acts.append((LUI, 1, 0, 0, ib))
        acts.append((LUI, 2, 0, 0, (ib+1)%5))
        for op in [MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU]:
            acts.append((op, 3, 1, 2, 0))
        # CSRRW cu valori mari
        acts.append((CSRRW, 0, 1, 0, ib))
        # Shift operations
        for op in [SLL, SRL, SRA, SLLI, SRLI, SRAI]:
            acts.append((op, 4, 1, 2, ib))
    return (acts * ((n // max(len(acts), 1)) + 1))[:n]


def prog_instr_addr_bits(n=200):
    """Instruction address bits: sare la diverse adrese pentru a togla PC bits."""
    acts = []
    # JAL/JALR cu offsets diverse → schimbă PC
    for ib in range(IMM_BUCKETS):
        acts.append((JAL, 1, 0, 0, ib))
        acts.append((AUIPC, 2, 0, 0, ib))
        acts.append((JALR, 0, 1, 0, ib))
        acts.append((JALR, 0, 2, 0, ib))
    # Branches taken/not-taken
    for op in [BEQ, BNE]:
        for ib in range(IMM_BUCKETS):
            acts.append((op, 0, 0, 0, ib))   # BEQ x0,x0 → always taken
            acts.append((op, 0, 1, 0, ib))   # BEQ x1,x0 → usually not taken
    return (acts * ((n // max(len(acts), 1)) + 1))[:n]


PROGRAMS = {
    "sext_result":      prog_sext_result,
    "branch_target":    prog_branch_target,
    "csr_values":       prog_csr_values,
    "counter_bits":     prog_counter_bits,
    "mul_div_operands": prog_mul_div_operands,
    "load_store_addr":  prog_load_store_addr,
    "alu_imd_val":      prog_alu_imd_val,
    "instr_addr_bits":  prog_instr_addr_bits,
}


# ── Utilitare ──────────────────────────────────────────────────────────────────

_PAGE_RE = re.compile(r"page\x02([^\x01]+)")
_SIG_RE  = re.compile(r"\x01o\x02([^\x01]+)")
_LINE_RE = re.compile(r"\x01l\x02([^\x01]+)")

def stable_id(k):
    pm = _PAGE_RE.search(k); sm = _SIG_RE.search(k); lm = _LINE_RE.search(k)
    return (f"{pm.group(1) if pm else '?'}"
            f"|{sm.group(1) if sm else '?'}"
            f"|{lm.group(1) if lm else '?'}")

def parse_point(key):
    page_m = re.search(r"page\x02([^\x01]+)", key)
    sig_m  = re.search(r"\x01o\x02([^\x01]+)", key)
    line_m = re.search(r"\x01l\x02([^\x01]+)", key)
    page = page_m.group(1) if page_m else "?"
    sig  = sig_m.group(1)  if sig_m  else "?"
    line = line_m.group(1) if line_m else "?"
    return page, sig, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits",   required=True)
    ap.add_argument("--covdat", default="../../cpu/coverage.dat")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Câte repetări per program tip (default 3)")
    args = ap.parse_args()

    # ── Încarcă starea cumulată ───────────────────────────────────────────────
    with open(args.hits, "rb") as f:
        cum_hits: set = pickle.load(f)
    s = cov_parser.parse(args.covdat)

    prefix = "page\x02v_toggle/"
    all_toggle_keys = {k for k in s.points if prefix in k}

    # Mapă stable_id → cheie originală în coverage.dat
    covdat_id_to_key = {stable_id(k): k for k in all_toggle_keys}
    cum_hits_ids = {stable_id(k) for k in cum_hits}

    # ── Identifică semnalele REACHABLE? neacoperite ───────────────────────────
    uncov_reachable = {}   # stable_id → (module, signal, line)
    for sid, k in covdat_id_to_key.items():
        if sid in cum_hits_ids:
            continue
        page, sig, line = parse_point(k)
        tag = classify(sig, k)
        if not tag.startswith("TIED") and not tag.startswith("NEEDS"):
            module = page[len("v_toggle/"):].split("__")[0] if page.startswith("v_toggle/") else page
            uncov_reachable[sid] = (module, sig, line)

    print(f"Semnale REACHABLE? neacoperite de verificat: {len(uncov_reachable)}")
    print(f"Rulăm {len(PROGRAMS)} tipuri de programe × {args.repeats} repetări\n")

    # ── Rulează programele țintite ────────────────────────────────────────────
    newly_covered  = {}   # stable_id → program_type
    still_uncovered = set(uncov_reachable.keys())

    total_runs = 0
    t0 = time.time()

    for prog_name, prog_fn in PROGRAMS.items():
        if not still_uncovered:
            break
        for rep in range(args.repeats):
            if not still_uncovered:
                break
            actions = prog_fn()
            summary = run_program(actions)
            total_runs += 1
            if summary is None:
                print(f"  [{prog_name} rep{rep+1}] vtop FAILED")
                continue

            # Verifică ce semnale s-au toggleat în acest run
            newly_hit = set()
            for k, count in summary.points.items():
                if count > 0 and prefix in k:
                    sid = stable_id(k)
                    if sid in still_uncovered:
                        newly_hit.add(sid)

            for sid in newly_hit:
                newly_covered[sid] = prog_name
                still_uncovered.discard(sid)

            elapsed = time.time() - t0
            print(f"  [{prog_name:20s} rep{rep+1}]  "
                  f"nou acoperite: {len(newly_hit):3d}  "
                  f"rămase: {len(still_uncovered):4d}  "
                  f"({elapsed:.0f}s)", flush=True)

    # ── Raport final ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"REZULTATE FINALE după {total_runs} rulări ({time.time()-t0:.0f}s)")
    print(f"{'='*65}")
    print(f"  REACHABLE? inițial:          {len(uncov_reachable):>5}")
    print(f"  Acoperite de programe țintite:{len(newly_covered):>5}  ← genuinE reachable")
    print(f"  Rămase neacoperite:           {len(still_uncovered):>5}  ← probabil TIED-OFF")
    print()

    # Coverage actualizată
    cum_covered_final = len(cum_hits_ids) + len(newly_covered)
    total_tog = s.by_kind["toggle"][1]
    new_pct = 100.0 * cum_covered_final / total_tog
    print(f"  Coverage cumulat inițial:  {100.*len(cum_hits_ids)/total_tog:.2f}%")
    print(f"  Coverage după test țintit: {new_pct:.2f}%  (+{new_pct - 100.*len(cum_hits_ids)/total_tog:.2f}pp)")

    # Ce program a acoperit ce
    by_prog = defaultdict(list)
    for sid, pname in newly_covered.items():
        by_prog[pname].append(sid)
    if by_prog:
        print(f"\n  Acoperite per tip de program:")
        for pname, sids in sorted(by_prog.items(), key=lambda x: -len(x[1])):
            print(f"    {pname:<22s}: {len(sids):>4} semnale")

    # Semnale rămase neacoperite — clasificare
    print(f"\n=== Semnale rămase NEACOPERITE după toate programele țintite ===")
    print(f"(acestea sunt cu mare probabilitate efectiv TIED-OFF)\n")
    by_module_unc = defaultdict(list)
    for sid in sorted(still_uncovered):
        module, sig, line = uncov_reachable[sid]
        by_module_unc[module].append((sig, line))

    for mod in sorted(by_module_unc, key=lambda m: -len(by_module_unc[m])):
        sigs = by_module_unc[mod]
        print(f"  {mod} ({len(sigs)} semnale):")
        for sig, line in sorted(sigs)[:8]:
            print(f"    line {line:<5s}  {sig}")
        if len(sigs) > 8:
            print(f"    ... și alte {len(sigs)-8}")

    # Estimare plafon corectat
    n_false_reachable = len(still_uncovered)
    corrected_ceiling = 100.0 * (total_tog - (s.by_kind["toggle"][1] - len(all_toggle_keys))
                                 - (4555 + n_false_reachable)) / total_tog
    # Mai simplu:
    tied_off_estimated = 4548 + n_false_reachable  # 4548 din verify_ceiling + cele rămase
    corrected_ceiling2 = 100.0 * (total_tog - tied_off_estimated) / total_tog
    print(f"\n=== Estimare plafon corectat ===")
    print(f"  TIED-OFF din verify_ceiling:    4548")
    print(f"  False REACHABLE? confirmate:  + {n_false_reachable}")
    print(f"  Total TIED-OFF estimat:         {4548 + n_false_reachable}")
    print(f"  Plafon corectat:               {corrected_ceiling2:.2f}%")
    print(f"  Coverage actual:               {100.*len(cum_hits_ids)/total_tog:.2f}%")
    print(f"  Gap real față de plafon:       {corrected_ceiling2 - 100.*len(cum_hits_ids)/total_tog:.2f} pp")


if __name__ == "__main__":
    main()
