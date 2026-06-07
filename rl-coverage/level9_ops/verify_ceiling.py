"""verify_ceiling.py — Calculează plafonul real față de starea cumulată.

Spre deosebire de analyze_unreachable.py care citește UN singur coverage.dat
(ultimul run), acest script:
  1. Încarcă _cum_hits din checkpoint (starea reală cumulată peste toate ep.)
  2. Încarcă UN coverage.dat pentru a obține lista completă de puncte
  3. Marchează ca "acoperit" orice punct care e în _cum_hits
  4. Rulează clasificarea TIED-OFF pe ce rămâne neacoperit
  5. Raportează plafonul real față de starea cumulată

Asta elimină problema "ultimul run nu acoperă tot ce am acumulat".

Usage:
    python verify_ceiling.py --hits l9_v2_checkpoint_hits.pkl
    python verify_ceiling.py --hits l9_v2_checkpoint_hits.pkl --covdat ../../cpu/coverage.dat
"""

import sys, re, pickle, argparse
from pathlib import Path
from collections import defaultdict

THIS  = Path(__file__).resolve().parent
L5    = THIS.parent / "level5_real_rtl"
L7    = THIS.parent / "level7_stimulus"
sys.path.insert(0, str(L5))
sys.path.insert(0, str(L7))

import cov_parser
from analyze_unreachable import (
    TIED_OFF_SUBSTRINGS, REACHABLE_BUT_NEEDS,
    _COUNTER_HIGH_BIT_RE, _COUNTER_HIGH_THRESHOLD,
    classify, parse_point,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits",   required=True,
                    help="Fișier .pkl cu _cum_hits (ex: l9_v2_checkpoint_hits.pkl)")
    ap.add_argument("--covdat", default="../../cpu/coverage.dat",
                    help="coverage.dat pentru lista completă de puncte")
    args = ap.parse_args()

    # ── Încarcă cum_hits ──────────────────────────────────────────────────────
    with open(args.hits, "rb") as f:
        cum_hits: set = pickle.load(f)
    print(f"Loaded {len(cum_hits):,} hits din {args.hits}")

    # ── Încarcă coverage.dat ─────────────────────────────────────────────────
    s = cov_parser.parse(args.covdat)
    total_toggle = s.by_kind["toggle"][1]
    print(f"Loaded coverage.dat: {len(s.points):,} puncte, "
          f"{total_toggle:,} toggle points total")

    # ── Identificator stabil pentru fiecare toggle point ─────────────────────
    # Cheile din coverage.dat conțin câmpuri variabile între rulări/mașini:
    #   \x01f\x02<filepath>  — calea fișierului RTL (poate diferi)
    #   \x01n\x02<num>       — număr intern Verilator (se schimbă la recompilare)
    #   \x01h\x02<hier>      — ierarhia instanței (poate diferi cu dots vs _)
    # Identificatorul stabil: (page/modul, signal_name, line_number)
    # Acestea NU se schimbă dacă RTL-ul rămâne același.

    _PAGE_RE = re.compile(r"page\x02([^\x01]+)")
    _SIG_RE  = re.compile(r"\x01o\x02([^\x01]+)")
    _LINE_RE = re.compile(r"\x01l\x02([^\x01]+)")

    def stable_id(k: str) -> str:
        pm = _PAGE_RE.search(k)
        sm = _SIG_RE.search(k)
        lm = _LINE_RE.search(k)
        return (f"{pm.group(1) if pm else '?'}"
                f"|{sm.group(1) if sm else '?'}"
                f"|{lm.group(1) if lm else '?'}")

    # Construiește set toggle keys din coverage.dat
    prefix = "page\x02v_toggle/"
    all_toggle_keys = {k for k in s.points if prefix in k}
    print(f"Toggle points în coverage.dat: {len(all_toggle_keys):,}")

    # Mapă stable_id → cheie originală (pentru acces la s.points)
    covdat_id_to_key = {stable_id(k): k for k in all_toggle_keys}
    print(f"IDs unice în coverage.dat:     {len(covdat_id_to_key):,}")

    # Construiește set stable_ids din pkl
    cum_hits_ids = {stable_id(k) for k in cum_hits}
    print(f"IDs unice în pkl:              {len(cum_hits_ids):,}")

    # Câte pkl IDs se regăsesc în coverage.dat?
    overlap = cum_hits_ids & set(covdat_id_to_key.keys())
    print(f"Overlap pkl ∩ coverage.dat:    {len(overlap):,}  "
          f"({'%.1f' % (100*len(overlap)/max(len(cum_hits_ids),1))}% din pkl)")

    if len(overlap) < len(cum_hits_ids) * 0.5:
        print("\n  [ATENTIE] Overlap mic — cheile din pkl par incompatibile cu")
        print("  coverage.dat curent. Posibil Vtop recompilat între rulări.")
        print("  Continuăm cu coverage.dat single-run ca fallback.")

    covered   = 0
    uncovered = []
    for sid, k in covdat_id_to_key.items():
        if sid in cum_hits_ids or s.points.get(k, 0) > 0:
            covered += 1
        else:
            uncovered.append(k)

    cum_pct = 100.0 * covered / max(total_toggle, 1)
    print(f"\nStare cumulată: {covered:,}/{total_toggle:,} = {cum_pct:.2f}%")
    print(f"Neacoperite cumulat: {len(uncovered):,}")

    # ── Clasifică semnalele neacoperite ──────────────────────────────────────
    by_module   = defaultdict(list)
    tag_totals  = defaultdict(int)
    tied_reasons = defaultdict(int)

    for key in uncovered:
        page, sig, line = parse_point(key)
        module = page[len("v_toggle/"):].split("__")[0] if page.startswith("v_toggle/") else page
        tag = classify(sig, key)
        by_module[module].append((tag, sig, line))
        bucket = ("TIED-OFF" if tag.startswith("TIED") else
                  "NEEDS"    if tag.startswith("NEEDS") else "REACHABLE?")
        tag_totals[bucket] += 1
        if bucket == "TIED-OFF":
            # extrage motivul din tag (e.g. "TIED-OFF (pmp_)" → "pmp_")
            m = re.search(r'\((.+)\)', tag)
            tied_reasons[m.group(1) if m else tag] += 1

    # ── Raport per modul ──────────────────────────────────────────────────────
    print("\n=== Semnale neacoperite cumulat, per modul ===")
    rows = []
    for m, pts in by_module.items():
        tied  = sum(1 for (t,_,_) in pts if t.startswith("TIED"))
        needs = sum(1 for (t,_,_) in pts if t.startswith("NEEDS"))
        reach = len(pts) - tied - needs
        rows.append((len(pts), tied, needs, reach, m))
    rows.sort(reverse=True)
    print(f"{'module':<30s} {'uncov':>6s} {'tied':>6s} {'needs':>6s} {'?':>6s}")
    print("-" * 60)
    for uncov, tied, needs, reach, m in rows[:20]:
        print(f"{m:<30s} {uncov:>6d} {tied:>6d} {needs:>6d} {reach:>6d}")
    if len(rows) > 20:
        print(f"  … {len(rows)-20} alte module")

    # ── Totals ────────────────────────────────────────────────────────────────
    print(f"\nTotals (față de starea cumulată {cum_pct:.2f}%):")
    print(f"  TIED-OFF   (nerealizabile în config): {tag_totals['TIED-OFF']:>5}")
    print(f"  NEEDS      (cu stimul cunoscut):       {tag_totals['NEEDS']:>5}")
    print(f"  REACHABLE? (pot fi acoperite):         {tag_totals['REACHABLE?']:>5}")

    ceiling = 100.0 * (total_toggle - tag_totals['TIED-OFF']) / total_toggle
    gap     = ceiling - cum_pct
    print(f"\n  Plafon calculat (tied-off excluse): {ceiling:.2f}%")
    print(f"  Coverage cumulat actual:            {cum_pct:.2f}%")
    print(f"  Gap față de plafon:                 {gap:+.2f} pp")

    if gap < 0:
        print(f"\n  [ATENTIE] Plafonul e sub coverage-ul actual ({gap:.2f} pp)!")
        print(f"  Inseamna ca {-gap:.2f}% din semnalele acoperite sunt")
        print(f"  clasificate gresit ca TIED-OFF. Top motive:")
        for reason, cnt in sorted(tied_reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"    {reason:<45s}: {cnt:>4} semnale")
    else:
        print(f"\n  [OK] Plafonul e deasupra coverage-ului actual.")
        print(f"  Raman {tag_totals['REACHABLE?']} semnale teoretic accesibile.")

    # ── Sample REACHABLE? ─────────────────────────────────────────────────────
    import random
    rng = random.Random(42)
    reachable = [(m, sig, line) for m, pts in by_module.items()
                 for (tag, sig, line) in pts if not tag.startswith("TIED")]
    rng.shuffle(reachable)
    print(f"\n=== Sample semnale REACHABLE? / NEEDS neacoperite cumulat ===")
    for m, sig, line in reachable[:20]:
        print(f"  {m:<28s}  line {line:<5s}  {sig}")


if __name__ == "__main__":
    main()
