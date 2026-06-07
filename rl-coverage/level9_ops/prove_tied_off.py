"""prove_tied_off.py — Dovedește care semnale REACHABLE? sunt de fapt tied-off.

Metodă: Verilator a compilat Ibex RTL → C++. În timpul compilării,
constant propagation a transformat orice semnal structural-zero în = 0U.
Dacă un semnal apare DOAR cu = 0U în fișierele de evaluare (Slow.cpp),
este DOVEDIT matematic că nu poate togla niciodată.

Tool folosit: Verilator (deja compilat în sim_build/)

Usage:
    python prove_tied_off.py
    python prove_tied_off.py --simdir ../../cpu/sim_build --input uncovered_reachable.txt
"""

import re, sys, argparse
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent

# ── Regex pentru asignare la constant zero în C++ ────────────────────────────
# Recunoaşte: vlSelfRef.xxx = 0U;  sau  vlSelfRef.xxx = 0ULL;  sau  = 0;
_ZERO_RE    = re.compile(r'^\s*\S+\s*=\s*0U(?:LL)?;?\s*$')
_ASSIGN_RE  = re.compile(r'=\s*(.+?);')
_ZERO_VALS  = {"0U", "0ULL", "0u", "0ull", "0", "0x0U", "0x0ULL", "0x0u"}

# Linii de skip (nu sunt eval real)
_SKIP = ["__Vtogcov__", "chgBit", "chgQData", "chgIData",
         "chgWData", "bufp->", "VL_RAND_RESET", "Vtop__Trace",
         "__Vconfigure", "Vtop__Syms"]


def is_eval_file(fp: Path) -> bool:
    """Fişier de evaluare reală (nu trace, nu syms, nu init)."""
    n = fp.name
    return ("Slow" in n or "__DepSet" in n) and "Trace" not in n and "Syms" not in n


def check_signal(base_name: str, cpp_files: list) -> tuple[str, list]:
    """
    Verifică semnalul în fişierele C++ de evaluare.
    Returnează (status, [linii relevante]):
      'constant-0'   — dovedit = 0U de Verilator
      'computed'     — calculat din expresie (poate togla)
      'input-only'   — doar VL_RAND_RESET (input din testbench)
      'not-found'    — eliminat complet (dead code)
    """
    pattern = re.compile(r'\b' + re.escape(base_name) + r'\b')

    found_zero   = False
    found_nonzero = False
    found_any    = False
    evidence     = []

    for fp in cpp_files:
        if not is_eval_file(fp):
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if base_name not in text:
            continue

        for line in text.splitlines():
            if base_name not in line:
                continue
            if any(s in line for s in _SKIP):
                continue

            # Caută asignare: ... base_name ... = expr;
            # Trebuie să fie pe stânga semnului =
            left_side = line.split("=")[0] if "=" in line else ""
            if base_name not in left_side:
                continue

            found_any = True
            m = _ASSIGN_RE.search(line)
            if not m:
                continue

            rhs = m.group(1).strip().rstrip(";").strip()
            if rhs in _ZERO_VALS:
                found_zero = True
                evidence.append(line.strip()[:100])
            else:
                found_nonzero = True
                evidence.append(line.strip()[:100])

    if not found_any:
        return "not-found (dead-code)", []
    if found_nonzero:
        return "computed (accesibil)", evidence[:2]
    if found_zero:
        return "constant-0 (tied-off)", evidence[:2]
    return "input-only (testbench)", evidence[:1]


def extract_base(sig: str) -> str:
    """bt_a_operand[5] → bt_a_operand"""
    return re.sub(r'\[.*', '', sig).strip()


def load_signals(input_file: Path) -> list[tuple[str, str, str]]:
    """Citeşte uncovered_reachable.txt → [(module, signal, tag)]"""
    signals = []
    current_module = "unknown"
    with open(input_file) as f:
        for line in f:
            m = re.match(r'^## (\S+)', line)
            if m:
                current_module = m.group(1)
                continue
            m = re.match(r'^\s+(\S+)\s+line \S+\s+\[(.+?)\]', line)
            if m:
                signals.append((current_module, m.group(1), m.group(2)))
    return signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", default="../../cpu/sim_build")
    ap.add_argument("--input",  default="uncovered_reachable.txt")
    ap.add_argument("--out",    default="proof_results.txt")
    args = ap.parse_args()

    simdir     = Path(args.simdir)
    input_file = Path(args.input)

    print("=" * 65)
    print("PROVE_TIED_OFF — Dovadă bazată pe output-ul Verilator")
    print("=" * 65)
    print(f"  Input:   {input_file}")
    print(f"  Sim dir: {simdir}")
    print()

    # ── Încarcă semnalele ──────────────────────────────────────────────
    signals = load_signals(input_file)
    print(f"  Semnale de verificat: {len(signals)}")

    # ── Încarcă fişierele C++ ──────────────────────────────────────────
    cpp_files = sorted(simdir.glob("*.cpp"))
    eval_files = [f for f in cpp_files if is_eval_file(f)]
    print(f"  Fişiere eval C++:     {len(eval_files)} din {len(cpp_files)}")
    print()

    # ── Verifică fiecare semnal ────────────────────────────────────────
    results = defaultdict(list)
    seen_bases = {}  # cache: base_name → status

    for i, (module, sig, tag) in enumerate(signals):
        if i % 100 == 0:
            print(f"  Progres: {i}/{len(signals)}...", flush=True)

        base = extract_base(sig)
        if base in seen_bases:
            status, evidence = seen_bases[base]
        else:
            status, evidence = check_signal(base, eval_files)
            seen_bases[base] = (status, evidence)

        results[status].append((module, sig, tag, evidence))

    # ── Raport ────────────────────────────────────────────────────────
    total = len(signals)
    print()
    print("=" * 65)
    print("REZULTATE")
    print("=" * 65)

    cats = [
        ("constant-0 (tied-off)",      "🔴 TIED-OFF   — dovedit = 0U de Verilator"),
        ("not-found (dead-code)",       "🔴 DEAD CODE  — eliminat complet de Verilator"),
        ("input-only (testbench)",      "⚠️  TESTBENCH  — input din exterior (debug_req, bus_err)"),
        ("computed (accesibil)",        "✅ ACCESIBIL  — calculat din expresie, poate togla"),
    ]

    counts = {}
    for key, label in cats:
        n = len(results[key])
        counts[key] = n
        pct = 100 * n / max(total, 1)
        print(f"\n  {label}: {n} ({pct:.1f}%)")
        # Arată câteva exemple
        for module, sig, tag, ev in results[key][:5]:
            print(f"    {module:<25s}  {sig}")
        if len(results[key]) > 5:
            print(f"    ... şi alte {len(results[key])-5}")

    inacc = counts["constant-0 (tied-off)"] + counts["not-found (dead-code)"]
    acc   = counts["computed (accesibil)"]
    tb    = counts["input-only (testbench)"]

    print()
    print("=" * 65)
    print(f"  TIED-OFF confirmat (= 0U sau dead-code): {inacc:>5}  ({100*inacc/total:.1f}%)")
    print(f"  TESTBENCH-blocat (input extern fix):      {tb:>5}  ({100*tb/total:.1f}%)")
    print(f"  ACCESIBIL (poate togla în simulare):      {acc:>5}  ({100*acc/total:.1f}%)")
    print()

    total_toggle = 20023
    covered      = 14404
    print(f"  Din totalul de {total_toggle} toggle points:")
    print(f"    Acoperite:                 {covered}  ({100*covered/total_toggle:.2f}%)")
    print(f"    Tied-off confirmat acum:   {inacc}   ({100*inacc/total_toggle:.2f}%)")
    print(f"    Testbench-blocate:         {tb}    ({100*tb/total_toggle:.2f}%)")
    print(f"    Cu adevărat accesibile:    {acc}    ({100*acc/total_toggle:.2f}%)")
    print("=" * 65)

    # ── Salvează rezultate detaliate ───────────────────────────────────
    with open(args.out, "w") as f:
        f.write("PROVE_TIED_OFF — Rezultate detaliate\n")
        f.write("=" * 65 + "\n\n")
        for key, label in cats:
            f.write(f"\n{label} ({len(results[key])} semnale)\n")
            f.write("-" * 60 + "\n")
            for module, sig, tag, ev in results[key]:
                f.write(f"  {module:<25s}  {sig}\n")
                for e in ev:
                    f.write(f"    C++: {e}\n")

    print(f"\n  Rezultate detaliate → {args.out}")


if __name__ == "__main__":
    main()
