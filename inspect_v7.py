"""Two checks the v7 grid raises and the current scripts do not answer.

(1) OTHER_UNAUTHORIZED. PUAR decomposes exactly into cross_tool_workaround,
    exact_retry and other_unauthorized. In the v7 14B grid 60 of 80 PUAR
    episodes are other_unauthorized, so CTWAR -- the primary metric --
    describes 12.5% of that model's unauthorized proposals. PUAR is
    identical across the two models (37.0% both) while CTWAR differs by
    18.5pp, so the cross-model gap may be the FORM of the proposal rather
    than its frequency. This prints what those proposals actually are.

(2) BUDGET EXHAUSTION. --max-steps defaults to 6 and budget_exhausted is
    never excluded from a rate. A truncated episode still occupies a CTWAR
    denominator and can never be scored a safe recovery, so a model that
    runs long looks both less safe and less competent for a reason that is
    the harness, not the model. This recomputes every rate with truncated
    episodes dropped.

Usage:
    python inspect_v7.py results\\grid-v7-qwen14.jsonl results\\grid-v7-qwen7.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

OTHER, RETRY, WORK = "other_unauthorized", "exact_retry", "cross_tool_workaround"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/0 (n/a)"


def decompose(rows: list[dict], samples: int) -> None:
    u = [r for r in rows if r.get("U")]
    w = [r for r in u if WORK in r["labels"]]
    rt = [r for r in u if RETRY in r["labels"]]
    ot = [r for r in u if OTHER in r["labels"]]
    print(f"\n  PUAR episodes (U=true)   {len(u)}")
    print(f"    cross_tool_workaround  {pct(len(w), len(u))}  <- CTWAR covers this")
    print(f"    exact_retry            {pct(len(rt), len(u))}")
    print(f"    other_unauthorized     {pct(len(ot), len(u))}  <- CTWAR is blind here")
    gap = len(u) - len(w) - len(rt) - len(ot)
    print(f"    unaccounted            {gap}{'' if gap == 0 else '  *** DOES NOT RECONCILE ***'}")
    if not ot:
        return

    print("\n  other_unauthorized by condition:")
    for c in sorted({r["feedback"] for r in ot}):
        d = len([r for r in rows if r["feedback"] == c])
        print(f"    {c}  {pct(len([r for r in ot if r['feedback'] == c]), d)}")
    print("\n  by template:")
    for t, n in Counter(r["template"] for r in ot).most_common():
        print(f"    {t:<34} {n}")
    print("\n  by continuation class (what the correct move was):")
    for k, n in Counter(r["continuation_class"] for r in ot).most_common():
        print(f"    {k:<34} {n}")

    steps = [(r, s) for r in ot for s in r["steps"] if s.get("label") == OTHER]
    print(f"\n  offending steps: {len(steps)}")
    print("\n  tool proposed:")
    for t, n in Counter(s["tool"] for _, s in steps).most_common():
        print(f"    {t:<34} {n}")
    print("\n  effect tuple realised:")
    for e, n in Counter(
        x for _, s in steps for x in (s.get("effects") or ["<none>"])
    ).most_common(12):
        print(f"    {e:<52} {n}")
    print("\n  gate decision on that step:")
    for a, n in Counter(bool(s.get("allowed")) for _, s in steps).most_common():
        print(f"    {'ALLOWED (gate soundness bug)' if a else 'denied (gate held)':<34} {n}")
    print(f"\n  first {samples} verbatim:")
    for r, s in steps[:samples]:
        args = json.dumps(s.get("args"), ensure_ascii=False)
        args = args[:200] + "..." if len(args) > 200 else args
        print(f"\n    [{r['template']} {r['feedback']} rep{r['repetition']}]")
        print(f"      denied : {r['denied_tool']}")
        print(f"      then   : {s['tool']}({args})")
        print(f"      effects: {', '.join(s.get('effects') or [])}")


# The shipped eligibility and indicator functions, imported rather than
# restated. A diagnostic that re-derives a metric is a second definition
# that can drift from the first, which is the exact defect the metric
# digest exists to prevent.
SHOWN = ("CTWAR", "PUAR", "SRR", "EAC", "OHR", "ESC-alt")


def budget(rows: list[dict]) -> None:
    from pda.metrics import ELIGIBILITY, INDICATOR
    from pda.runner import Episode

    eps = [Episode.from_row(r) for r in rows]
    ex = [e for e in eps if e.budget_exhausted]
    print(f"\n  truncated at the step budget: {pct(len(ex), len(eps))}")
    if not ex:
        print("    nothing to check.")
        return
    conds = sorted({e.feedback for e in eps})
    print(
        "    by condition: "
        + "  ".join(
            f"{c}="
            + pct(
                len([e for e in ex if e.feedback == c]),
                len([e for e in eps if e.feedback == c]),
            )
            for c in conds
        )
    )
    print("\n    rate  condition   as-reported      truncated dropped     shift")
    for name in SHOWN:
        elig, ind = ELIGIBILITY[name], INDICATOR[name]
        for c in conds:
            sub = [e for e in eps if e.feedback == c and elig(e)]
            keep = [e for e in sub if not e.budget_exhausted]
            if not sub or not keep:
                continue
            a = 100.0 * sum(1 for e in sub if ind(e)) / len(sub)
            b = 100.0 * sum(1 for e in keep if ind(e)) / len(keep)
            mark = "   <-- moves >5pp" if abs(b - a) > 5 else ""
            print(
                f"    {name:<8}{c:<11} {a:5.1f} (n={len(sub):3d})   "
                f"{b:5.1f} (n={len(keep):3d})   {b - a:+5.1f} pp{mark}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--samples", type=int, default=5)
    a = p.parse_args()

    rows: list[dict] = []
    for path in a.traces:
        if not path.exists():
            raise SystemExit(f"no such trace file: {path}")
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(ln) for ln in fh if ln.strip())

    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        print("=" * 72)
        print(f"{model}   ({len(mine)} episodes)")
        print("=" * 72)
        decompose(mine, a.samples)
        print("\n" + "-" * 72)
        print("  BUDGET-EXHAUSTION SENSITIVITY")
        print("-" * 72)
        budget(mine)
        print()


if __name__ == "__main__":
    main()