"""export_uncovered_reachable.py — Exportă bins-urile REACHABLE/NEEDS neacoperite
pentru a fi folosite ca input pentru un LLM care generează programe țintite.

Usage:
    python export_uncovered_reachable.py --hits ../level10_ops/l10_focused_checkpoint_hits.pkl
    python export_uncovered_reachable.py --hits ../level10_ops/l10_focused_checkpoint_hits.pkl --out uncovered.txt
"""

import sys, re, pickle, argparse
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
L5   = THIS.parent / "level5_real_rtl"
L7   = THIS.parent / "level7_stimulus"
sys.path.insert(0, str(L5))
sys.path.insert(0, str(L7))

import cov_parser
from analyze_unreachable import classify, parse_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits",   required=True)
    ap.add_argument("--covdat", default="../../cpu/coverage.dat")
    ap.add_argument("--out",    default="uncovered_reachable.txt")
    args = ap.parse_args()

    # ── Încarcă hits cumulate ─────────────────────────────────────────────────
    with open(args.hits, "rb") as f:
        cum_hits: set = pickle.load(f)
    print(f"Hits încărcate: {len(cum_hits):,}")

    # ── Încarcă coverage.dat ──────────────────────────────────────────────────
    s = cov_parser.parse(args.covdat)
    total_toggle = s.by_kind["toggle"][1]
    print(f"Toggle points total: {total_toggle:,}")

    # ── Stable ID matching ────────────────────────────────────────────────────
    _PAGE_RE = re.compile(r"page\x02([^\x01]+)")
    _SIG_RE  = re.compile(r"\x01o\x02([^\x01]+)")
    _LINE_RE = re.compile(r"\x01l\x02([^\x01]+)")

    def stable_id(k):
        pm = _PAGE_RE.search(k)
        sm = _SIG_RE.search(k)
        lm = _LINE_RE.search(k)
        return (f"{pm.group(1) if pm else '?'}"
                f"|{sm.group(1) if sm else '?'}"
                f"|{lm.group(1) if lm else '?'}")

    prefix = "page\x02v_toggle/"
    all_toggle_keys   = {k for k in s.points if prefix in k}
    covdat_id_to_key  = {stable_id(k): k for k in all_toggle_keys}
    cum_hits_ids      = {stable_id(k) for k in cum_hits}

    # ── Identifică bins neacoperite REACHABLE / NEEDS ─────────────────────────
    reachable = []  # (module, signal, line, tag)

    for sid, k in covdat_id_to_key.items():
        if sid in cum_hits_ids or s.points.get(k, 0) > 0:
            continue  # acoperit
        page, sig, line = parse_point(k)
        tag = classify(sig, k)
        if tag.startswith("TIED"):
            continue  # inaccessibil structural
        module = page[len("v_toggle/"):].split("__")[0] if page.startswith("v_toggle/") else page
        reachable.append((module, sig, line, tag))

    reachable.sort(key=lambda x: (x[0], x[1]))
    print(f"Bins REACHABLE/NEEDS neacoperite: {len(reachable):,}")

    # ── Export ────────────────────────────────────────────────────────────────
    by_module = defaultdict(list)
    for module, sig, line, tag in reachable:
        by_module[module].append((sig, line, tag))

    with open(args.out, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UNCOVERED REACHABLE TOGGLE BINS — Ibex RISC-V\n")
        f.write(f"Total: {len(reachable)} bins across {len(by_module)} modules\n")
        f.write("=" * 70 + "\n\n")
        f.write("FORMAT: module | signal | line | category\n")
        f.write("NEEDS    = known stimulus exists (exception, CSR write, etc.)\n")
        f.write("REACHABLE? = heuristic cannot confirm as tied-off\n\n")

        for module, pts in sorted(by_module.items(), key=lambda x: -len(x[1])):
            f.write(f"\n## {module}  ({len(pts)} bins)\n")
            f.write("-" * 60 + "\n")
            for sig, line, tag in sorted(pts):
                f.write(f"  {sig:<45s}  line {line:<6s}  [{tag}]\n")

    print(f"\nExportat → {args.out}")
    print(f"Poți folosi acest fișier ca input pentru un LLM care generează")
    print(f"programe assembly țintite pentru bins-urile neacoperite.")


if __name__ == "__main__":
    main()
