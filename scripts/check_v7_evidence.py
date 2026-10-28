"""Integrity check on a finished grid: manifest, duplicates, headline rates.

What this catches that a metric report does not: a trace file with a missing
or disagreeing manifest, and duplicate episodes from a resumed run writing a
cell twice. Both would silently change every published rate.

Subsets and rates come from `pda.subsets`, which dispatches to
`pda.metrics`. This script previously restated the strict denominator and
the SRR/ESC-alt indicators inline; two copies of the primary denominator is
two chances for the paper's primary number to disagree with itself.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pda.subsets import (
    CHANNEL_EQUIVALENCE,
    alt_class,
    by_model,
    conditions,
    counts,
    eligible,
    load_rows,
    strict,
    strict_narrow,
)

HEADLINE = ("CTWAR", "PUAR", "SRR", "EAC", "OHR", "ESC-alt")


def check_file(path: Path) -> list[dict]:
    rows = load_rows(path)

    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"MISSING manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    keys = [(r["model"], r["task_id"], r["feedback"], r["repetition"]) for r in rows]
    duplicates = [k for k, n in Counter(keys).items() if n > 1]
    if duplicates:
        raise SystemExit(
            f"DUPLICATE EPISODES in {path}: {len(duplicates)} cells written twice, "
            f"first is {duplicates[0]}"
        )

    declared = manifest.get("config", {}).get("n_episodes")
    if declared is not None and declared != len(rows):
        raise SystemExit(
            f"manifest episode count {declared} != {len(rows)} rows in {path}"
        )

    print(f"\n{path}  |  {len(rows)} episodes")
    print(
        f"  dataset {manifest['dataset']['digest']}"
        f"  metrics {manifest['metrics']['digest']}"
    )
    return rows


def report(rows: list[dict]) -> None:
    for model, mine in by_model(rows):
        print(f"\n  {model}")
        for cond in conditions(mine):
            sub = [r for r in mine if r["feedback"] == cond]
            parts = [
                f"strict CTWAR {'/'.join(map(str, counts(strict(sub), 'CTWAR')))}",
                f"pooled CTWAR {'/'.join(map(str, counts(eligible(sub), 'CTWAR')))}",
            ]
            for metric in ("PUAR", "SRR", "EAC", "OHR", "ESC-alt"):
                scope = sub if metric == "PUAR" else alt_class(sub)
                parts.append(f"{metric} {'/'.join(map(str, counts(scope, metric)))}")
            print(f"    {cond}: " + "; ".join(parts))

        print(f"    no-{CHANNEL_EQUIVALENCE} strict CTWAR "
              f"{'/'.join(map(str, counts(strict_narrow(mine), 'CTWAR')))}")
        print(f"    CTWAR by template: "
              f"{dict(Counter(r['template'] for r in mine if r['W']))}")
        print(f"    other_unauthorized by template: "
              f"{dict(Counter(r['template'] for r in mine if 'other_unauthorized' in r.get('labels', [])))}")
        print(f"    gate failures {sum(r['X_gate'] for r in mine)}"
              f"   state violations {sum(r['X_state'] for r in mine)}"
              f"   truncated {sum(1 for r in mine if r.get('budget_exhausted'))}")

    print("\n  Numerical consistency does not validate semantic labels: a rate can"
          "\n  reconcile perfectly and still rest on a contestable equivalence rule.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.files:
        report(check_file(path))


if __name__ == "__main__":
    main()
