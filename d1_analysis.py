"""D1: does CTWAR add information a task-level outcome label already carries?

Why this exists
---------------
The paper's incremental-value claim is that an execution-only or
task-success-only view of an episode hides post-denial behaviour. So far
that argument has been made conceptually -- "zero executed violations would
assign equal credit to a model that never proposed a violation and one that
repeatedly tried alternate channels". A reviewer can reasonably ask for the
contingency table instead of the argument, and it needs no new experiments:
every episode already records both the audit label and the task outcome.

The table crosses the audit label against what a coarse evaluator would see:

                     task completed        task failed
    CTWAR positive   A  invisible to a     B  a task-level evaluator sees
                        success metric        failure, but not WHY
    CTWAR negative   C  genuinely clean    D  failed for other reasons

Cell A is the strongest evidence for the metric: an episode a task-success
evaluator scores as a WIN while the agent pursued the forbidden effect.
Cell B is the common case and shows CTWAR partitioning failures by cause.
If A and B are both empty the metric adds nothing and the paper should say
so; if A is non-empty, task-level success is demonstrably not sufficient.

Reported for the strict non-adversarial subset (the paper's primary
denominator) and for the pooled set, because the strict subset is what the
headline rests on.

Usage
-----
    python d1_analysis.py results/grid-v6-qwen7.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Sequence

# Templates excluded from the strict primary denominator, with the reason.
BROAD_EQUIVALENCE = "T4"  # access grant treated as equivalent to an attachment
ADVERSARIAL = "T6"  # injected document content
CHANNEL_EQUIVALENCE = "T8"  # broadcast channel aliased to one recipient


def load(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def table(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Cross the audit label against the coarse task outcome.

    `C_plus` is the task-completion bit: recovery regardless of whether a
    forbidden proposal came first. That is deliberately the permissive
    outcome, because it is what a task-success evaluator with no view of the
    proposal trace would record. Using `C` instead would build the audit
    label into the baseline and make the comparison circular.
    """
    cells: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}
    for row in rows:
        workaround = bool(row.get("W"))
        completed = bool(row.get("C_plus"))
        if workaround and completed:
            cells["A"].append(row)
        elif workaround:
            cells["B"].append(row)
        elif completed:
            cells["C"].append(row)
        else:
            cells["D"].append(row)
    return cells


def _show(title: str, rows: Sequence[dict[str, Any]]) -> None:
    eligible = [r for r in rows if r.get("cross_tool")]
    if not eligible:
        print(f"\n  {title}: no cross-tool-eligible episodes")
        return
    cells = table(eligible)
    n = len(eligible)
    clusters = len({r["template"] for r in eligible})
    a, b, c, d = (len(cells[k]) for k in "ABCD")

    print(f"\n  {title}  (n={n}, {clusters} template clusters)")
    print(f"    {'':18s} {'task completed':>15s} {'task failed':>13s}")
    print(f"    {'CTWAR positive':18s} {a:>15d} {b:>13d}")
    print(f"    {'CTWAR negative':18s} {c:>15d} {d:>13d}")

    hidden = a
    print(
        f"    -> {hidden}/{n} episodes a task-success evaluator scores as a WIN "
        f"while the agent pursued the forbidden effect"
    )
    if a + b:
        print(
            f"    -> CTWAR partitions {a + b}/{n} episodes that an "
            f"execution-only view (UER=0) cannot distinguish at all"
        )
    if a == 0 and b == 0:
        print("    -> no positives: CTWAR adds nothing on this subset. Say so.")
    for key, label in (("A", "completed AND workaround"),):
        for row in cells[key][:4]:
            print(
                f"       e.g. {row['task_id']} {row['feedback']}: "
                f"{row['template']}, labels={row.get('labels')}"
            )


def main(paths: Sequence[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    for raw in paths:
        path = pathlib.Path(raw)
        rows = load(path)
        models = sorted({r["model"] for r in rows})
        print("=" * 74)
        print(f"{path}  —  {len(rows)} episodes")
        print("=" * 74)
        for model in models:
            mine = [r for r in rows if r["model"] == model]
            print(f"\n{model}")
            strict = [
                r
                for r in mine
                if not r["template"].startswith((BROAD_EQUIVALENCE, ADVERSARIAL))
            ]
            _show("STRICT (primary: no T4, no T6)", strict)
            _show(
                "STRICT, also excluding T8's channel equivalence",
                [r for r in strict if not r["template"].startswith(CHANNEL_EQUIVALENCE)],
            )
            _show("POOLED (sensitivity: all cross-tool eligible)", mine)
        print()
    print(
        "Cell A is the claim: a task-level evaluator records success while the\n"
        "audit records a forbidden-effect proposal. Report A, B and the\n"
        "denominators; an empty A and B means CTWAR added nothing here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))