"""Parser for Verilator's coverage.dat.

Each line is one coverage point in the form:
    C '<KV-encoded metadata>' <count>

Metadata fields use ASCII 0x01 as field separator and 0x02 as key/value
separator. Relevant keys:
  page  -> v_toggle/<module>, v_branch/<module>, or v_line/<module>
  f     -> file path
  l     -> line number
  o     -> signal name (for toggle) or coverpoint label
  h     -> hierarchy

Returns per-page covered/total counts and a per-point dict of (key -> count)
for delta computation between runs.
"""

import re
from collections import defaultdict
from dataclasses import dataclass


KIND_TOGGLE = "toggle"
KIND_BRANCH = "branch"
KIND_LINE = "line"


@dataclass
class CovSummary:
    by_kind: dict[str, tuple[int, int]]  # kind -> (covered, total)
    by_page: dict[str, tuple[int, int]]  # page name -> (covered, total)
    points: dict[str, int]                # point key -> count

    def kind_pct(self, kind: str) -> float:
        cov, tot = self.by_kind.get(kind, (0, 0))
        return 100.0 * cov / tot if tot else 0.0


def _kind_of(page: str) -> str | None:
    if page.startswith("v_toggle/"): return KIND_TOGGLE
    if page.startswith("v_branch/"): return KIND_BRANCH
    if page.startswith("v_line/"):   return KIND_LINE
    return None


_PAGE_RE = re.compile(r"\x01page\x02([^\x01\x27]+)")


def parse(path: str) -> CovSummary:
    by_kind_cov = defaultdict(int)
    by_kind_tot = defaultdict(int)
    by_page_cov = defaultdict(int)
    by_page_tot = defaultdict(int)
    points: dict[str, int] = {}

    with open(path) as f:
        for line in f:
            if not line.startswith("C '"):
                continue
            m = _PAGE_RE.search(line)
            if not m:
                continue
            page = m.group(1)
            kind = _kind_of(page)
            if kind is None:
                continue
            # The count is the last whitespace-separated token on the line.
            # The metadata is everything between the first "'" and the last "'".
            try:
                end_quote = line.rindex("'")
                metadata = line[3:end_quote]
                count = int(line[end_quote + 1:].strip())
            except (ValueError, IndexError):
                continue
            # Deduplicate points (same metadata string can appear twice with
            # different module hierarchies; we treat each unique key separately).
            key = metadata
            if key in points:
                points[key] = max(points[key], count)
                continue
            points[key] = count
            by_kind_tot[kind] += 1
            by_page_tot[page] += 1
            if count > 0:
                by_kind_cov[kind] += 1
                by_page_cov[page] += 1

    by_kind = {k: (by_kind_cov[k], by_kind_tot[k])
               for k in (KIND_TOGGLE, KIND_BRANCH, KIND_LINE)}
    by_page = {p: (by_page_cov[p], by_page_tot[p]) for p in by_page_tot}
    return CovSummary(by_kind=by_kind, by_page=by_page, points=points)


def hit_set(summary: CovSummary, kind: str | None = None) -> set[str]:
    """Set of point-keys that fired (count > 0)."""
    if kind is None:
        return {k for k, v in summary.points.items() if v > 0}
    prefix = f"\x01page\x02v_{kind}/"
    return {k for k, v in summary.points.items() if v > 0 and prefix in ("\x01" + k)}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "../../cpu/coverage.dat"
    s = parse(path)
    print(f"Coverage summary for {path}")
    for k in (KIND_TOGGLE, KIND_BRANCH, KIND_LINE):
        c, t = s.by_kind[k]
        print(f"  {k:7s}: {c:>5}/{t:>5}  ({s.kind_pct(k):.2f}%)")
    print(f"\nTop pages by uncovered count:")
    rows = []
    for page, (c, t) in s.by_page.items():
        rows.append((t - c, page, c, t))
    rows.sort(reverse=True)
    for missed, page, c, t in rows[:15]:
        print(f"  {missed:>4} uncovered  {c:>4}/{t:<4} {100*c/t:5.1f}%  {page}")
