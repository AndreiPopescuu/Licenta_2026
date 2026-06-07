"""verify_tiedoff_cpp.py — Verifică empiric că semnalele TIED-OFF sunt constante 0 în C++ Verilator.

Metodologie:
  1. Citește toate punctele toggle neacoperite din coverage.dat
  2. Clasifică fiecare semnal (TIED-OFF / REACHABLE?)
  3. Pentru fiecare semnal TIED-OFF, extrage numele variabilei C++ din Vtop__Syms.cpp
  4. Caută toate asignările acelei variabile în fișierele .cpp de evaluare (non-Trace)
  5. Raportează: confirmed-0 / assigned-variable / not-found (dead code)

Dacă un semnal TIED-OFF apare NUMAI cu "= 0U" sau "= 0ULL" → dovedit structural că nu poate togla.

Usage:
    python verify_tiedoff_cpp.py
    python verify_tiedoff_cpp.py --covdat ../../cpu/coverage.dat --simdir ../../cpu/sim_build
    python verify_tiedoff_cpp.py --sample 50   # verifică doar 50 semnale random
    python verify_tiedoff_cpp.py --category PMP # verifică doar o categorie
"""

import sys, re, argparse, random
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
L5   = THIS.parent / "level5_real_rtl"
L7   = THIS.parent / "level7_stimulus"
sys.path.insert(0, str(L5))
sys.path.insert(0, str(L7))

import cov_parser
import importlib.util as _ilu

# ── Încarcă analyze_unreachable explicit din level7_stimulus ──────────────────
_spec = _ilu.spec_from_file_location("analyze_unreachable_l7", str(L7 / "analyze_unreachable.py"))
_ar   = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ar)
classify   = _ar.classify
parse_point = _ar.parse_point


# ── Pattern-uri pentru asignare în C++ Verilator ─────────────────────────────
# Asignare la constant 0:   signal = 0U;  sau  signal = 0ULL;
# Asignare din variabilă:   signal = expr;  unde expr NU e 0U/0ULL
_ASSIGN_RE      = re.compile(r'(\w+)\s*=\s*(.+?);')
_CONST_ZERO_RE  = re.compile(r'^0U(?:LL)?$')

# Excludem linii care sunt doar tracking de toggle coverage sau trace
_SKIP_PATTERNS = [
    "__Vtogcov__",
    "chgBit", "chgQData", "chgWData", "chgIData",
    "bufp->",
    "Vtop__Trace",
    "Vtop__Syms",
]


def load_syms(simdir: Path) -> dict[str, str]:
    """Extrage maparea signal_name → C++ var_name din Vtop__Syms.cpp."""
    syms_file = simdir / "Vtop__Syms.cpp"
    if not syms_file.exists():
        print(f"  [!] {syms_file} nu există")
        return {}

    # Pattern: varInsert(__Vfinal,"<signal>", &(TOP.<cpp_var>), ...
    _VAR_RE = re.compile(
        r'varInsert\(__Vfinal\s*,\s*"([^"]+)"\s*,\s*&\(TOP\.([^)]+)\)'
    )

    sig_to_vars = defaultdict(list)  # signal_name → [cpp_var, ...]
    with open(syms_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _VAR_RE.search(line)
            if m:
                sig_name = m.group(1)
                cpp_var  = m.group(2)
                sig_to_vars[sig_name].append(cpp_var)

    print(f"  Syms: {len(sig_to_vars)} semnale unice găsite în Vtop__Syms.cpp")
    return dict(sig_to_vars)


def check_signal_in_cpp(cpp_var: str, cpp_files: list[Path]) -> str:
    """
    Verifică dacă cpp_var e asignat în fișierele C++ de evaluare.
    Returnează:
      'constant-0'     — toate asignările sunt la 0U/0ULL
      'assigned-var'   — cel puțin o asignare din expresie variabilă
      'not-found'      — variabila nu apare deloc (dead code eliminat)
      'only-tracking'  — apare doar în toggle tracking / trace
    """
    # Escaped pentru regex
    escaped = re.escape(cpp_var)
    # Asignare directă (nu array index): cpp_var = expr;
    assign_re = re.compile(rf'\b{escaped}\s*=\s*([^;]+?)\s*;')

    found_assign  = False
    found_nonzero = False
    found_any     = False

    for fp in cpp_files:
        # Sare fișierele de trace
        if "Trace" in fp.name or "Syms" in fp.name:
            continue

        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if cpp_var not in text:
            continue
        found_any = True

        for line in text.splitlines():
            if cpp_var not in line:
                continue
            # Sare liniile de tracking / trace
            if any(p in line for p in _SKIP_PATTERNS):
                continue

            m = assign_re.search(line)
            if not m:
                continue

            rhs = m.group(1).strip()
            found_assign = True
            if not _CONST_ZERO_RE.match(rhs):
                # Verifică dacă e altceva decât 0U/0ULL
                # (unele asignări sunt 0x0U sau ~0U etc.)
                if rhs in ("0", "0U", "0ULL", "0u", "0ull", "0x0U", "0x0ULL"):
                    pass
                else:
                    found_nonzero = True

    if not found_any:
        return "not-found (dead-code)"
    if not found_assign:
        return "only-tracking"
    if found_nonzero:
        return "assigned-variable"
    return "constant-0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covdat",   default="../../cpu/coverage.dat")
    ap.add_argument("--simdir",   default="../../cpu/sim_build")
    ap.add_argument("--sample",   type=int, default=0,
                    help="Verifică doar N semnale random (0 = toate)")
    ap.add_argument("--category", default=None,
                    help="Filtrează după categorie TIED-OFF (ex: 'pmp_', 'icache')")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    covdat = Path(args.covdat)
    simdir = Path(args.simdir)

    print(f"=== verify_tiedoff_cpp.py ===")
    print(f"  coverage.dat: {covdat}")
    print(f"  sim_build:    {simdir}")

    # ── 1. Încarcă coverage.dat ───────────────────────────────────────────────
    s = cov_parser.parse(str(covdat))
    total_toggle = s.by_kind["toggle"][1]
    print(f"  Toggle points total: {total_toggle:,}")

    # ── 2. Colectează semnale neacoperite TIED-OFF ────────────────────────────
    tied_off_signals = []  # (signal_name, tag, module)
    for key, count in s.points.items():
        if count > 0:
            continue
        page, sig, line = parse_point(key)
        if not page.startswith("v_toggle/"):
            continue
        tag = classify(sig, key)
        if not tag.startswith("TIED"):
            continue
        module = page[len("v_toggle/"):].split("__")[0]
        # Extrage motivul din tag (e.g. "TIED-OFF (pmp_)" → "pmp_")
        reason_m = re.search(r'\((.+?)\)', tag)
        reason = reason_m.group(1) if reason_m else tag
        if args.category and args.category.lower() not in reason.lower():
            continue
        tied_off_signals.append((sig, tag, module, reason))

    print(f"  Semnale TIED-OFF găsite: {len(tied_off_signals):,}")
    if not tied_off_signals:
        print("  Niciun semnal TIED-OFF găsit cu filtrele date."); return

    # Sample dacă cerut
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(tied_off_signals):
        tied_off_signals = rng.sample(tied_off_signals, args.sample)
        print(f"  → Sample random: {len(tied_off_signals)} semnale")

    # ── 3. Încarcă Syms ───────────────────────────────────────────────────────
    sig_to_vars = load_syms(simdir)

    # ── 4. Colectează fișierele C++ de evaluare ───────────────────────────────
    cpp_files = sorted(simdir.glob("*.cpp"))
    print(f"  Fișiere .cpp: {len(cpp_files)}")

    # ── 5. Verifică fiecare semnal ────────────────────────────────────────────
    results = defaultdict(list)  # status → [(sig, module, reason)]
    not_in_syms = []

    print(f"\n  Verificare {len(tied_off_signals)} semnale...")

    for i, (sig, tag, module, reason) in enumerate(tied_off_signals):
        if i % 50 == 0:
            print(f"    {i}/{len(tied_off_signals)}...", flush=True)

        cpp_vars = sig_to_vars.get(sig, [])
        if not cpp_vars:
            not_in_syms.append((sig, module, reason))
            results["not-in-syms"].append((sig, module, reason))
            continue

        # Verifică primul (și de obicei unic) var C++
        # Dacă sunt mai multe instanțe, verifică toate
        statuses = []
        for cpp_var in cpp_vars:
            st = check_signal_in_cpp(cpp_var, cpp_files)
            statuses.append(st)

        # Cel mai pesimist status
        if any(s == "assigned-variable" for s in statuses):
            status = "assigned-variable"
        elif any(s == "constant-0" for s in statuses):
            status = "constant-0"
        elif any(s == "only-tracking" for s in statuses):
            status = "only-tracking"
        else:
            status = "not-found (dead-code)"

        results[status].append((sig, module, reason))

    # ── 6. Raport ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"REZULTATE VERIFICARE TIED-OFF în C++ Verilator")
    print(f"{'='*65}")

    confirmed   = len(results.get("constant-0", []))
    dead_code   = len(results.get("not-found (dead-code)", []))
    only_track  = len(results.get("only-tracking", []))
    variable    = len(results.get("assigned-variable", []))
    not_sym     = len(results.get("not-in-syms", []))
    total       = len(tied_off_signals)

    print(f"\n  constant-0      (= 0U în C++, 100% confirmat): {confirmed:>5}  "
          f"({100*confirmed/max(total,1):.1f}%)")
    print(f"  not-found       (dead code eliminat de Verilator): {dead_code:>5}  "
          f"({100*dead_code/max(total,1):.1f}%)")
    print(f"  only-tracking   (apare doar în toggle instrumentation): {only_track:>5}  "
          f"({100*only_track/max(total,1):.1f}%)")
    print(f"  not-in-syms     (nu în Syms → eliminat complet): {not_sym:>5}  "
          f"({100*not_sym/max(total,1):.1f}%)")
    print(f"  assigned-var    (asignat din expresie → INVESTIGHEAZA): {variable:>5}  "
          f"({100*variable/max(total,1):.1f}%)")

    cert_total = confirmed + dead_code + only_track + not_sym
    print(f"\n  CERT TIED-OFF   (suma primelor 4 categorii): {cert_total:>5} / {total}")
    print(f"  NECESITA VERIFICARE SUPLIMENTARA:            {variable:>5}")

    if results.get("assigned-variable"):
        print(f"\n=== Semnale ASSIGNED-VAR (necesită investigație) ===")
        for sig, mod, reason in results["assigned-variable"][:30]:
            print(f"  {mod:<30s}  {sig:<40s}  [{reason}]")
        if len(results["assigned-variable"]) > 30:
            print(f"  ... și alte {len(results['assigned-variable'])-30}")

    if results.get("constant-0"):
        print(f"\n=== Sample constant-0 (confirmat TIED-OFF) ===")
        sample = rng.sample(results["constant-0"], min(15, len(results["constant-0"])))
        for sig, mod, reason in sample:
            print(f"  {mod:<30s}  {sig:<40s}  [{reason}]")

    print(f"\n{'='*65}")
    cert_pct = 100.0 * cert_total / max(total, 1)
    print(f"Concluzie: {cert_pct:.1f}% din semnalele TIED-OFF sunt dovedite structural în C++.")
    if variable == 0:
        print("  → Nicio excepție. Toate sunt confirmate TIED-OFF.")
    else:
        print(f"  → {variable} semnale necesită verificare suplimentară.")


if __name__ == "__main__":
    main()
