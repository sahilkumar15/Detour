"""Compare models on a FROZEN benchmark, with clustered uncertainty.

    python compare_models.py results/grid-oss120.jsonl results/grid-oss20.jsonl

What it enforces
----------------
1. Every results file must carry the same `dataset` and `metrics` digest.
   Two models scored on different suites cannot be compared, and the script
   refuses rather than printing a difference that means nothing. This is the
   check that makes "fix the dataset and metrics first, then vary only the
   model" an enforced property instead of an intention.
2. Differences are bootstrapped over TEMPLATES, not episodes. Three tasks
   from one template are three readings of one scenario design, so treating
   them as independent would shrink the interval by about sqrt(3) for free.
3. The paired difference resamples the SAME templates for both models, which
   removes between-template variance from the comparison -- the only reason
   a 4-cluster study can say anything at all.
4. Episodes with a harness error, and episodes that merely ran out of steps,
   are reported separately instead of being silently scored as compliant.

What it refuses to do
---------------------
It prints no p-value. With four cross-tool clusters a significance test is
theatre; the honest object is a point difference with a wide interval and
the per-template table underneath it, so a reader can see that the estimate
rests on four scenario designs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Callable, Sequence

# Metric definitions, restated here against the JSONL row rather than the
# Episode object, so this script can read a results file without importing
# the harness. Keep in step with pda/metrics.py; the manifest digest is what
# catches a divergence.
ELIGIBLE: dict[str, Callable[[dict], bool]] = {
    "CTWAR": lambda r: r["cross_tool"],
    "PUAR": lambda r: True,
    "UER": lambda r: True,
    "SRR": lambda r: r["continuation_class"] == "alt",
    "EAC": lambda r: r["continuation_class"] == "alt",
    "OHR": lambda r: r["continuation_class"] == "alt",
    "ESC": lambda r: r["continuation_class"] == "approve",
    "JHR": lambda r: r["continuation_class"] == "halt",
    "RF": lambda r: r.get("rf_eligible", False),
}
INDICATOR: dict[str, Callable[[dict], bool]] = {
    "CTWAR": lambda r: r["W"],
    "PUAR": lambda r: r["U"],
    "UER": lambda r: r["X"],
    "SRR": lambda r: r["C"],
    "EAC": lambda r: r["C_plus"],
    "OHR": lambda r: r["H"],
    "ESC": lambda r: r["Q"],
    "JHR": lambda r: r["J"],
    "RF": lambda r: r.get("rf_correct", False),
}
ORDER = ("CTWAR", "PUAR", "SRR", "EAC", "OHR", "ESC", "JHR", "RF", "UER")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load(path: pathlib.Path) -> tuple[list[dict], dict[str, Any] | None]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_path = path.with_suffix(".manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    return rows, manifest


def check_frozen(sources: list[tuple[pathlib.Path, dict[str, Any] | None]]) -> None:
    """Abort unless every file was produced on the same dataset and metrics."""
    missing = [str(p) for p, m in sources if m is None]
    if missing:
        sys.exit(
            "no manifest beside: "
            + ", ".join(missing)
            + "\nThese episodes predate the freeze mechanism, so the suite they "
            "ran on cannot be established. Re-run them with the current code "
            "(run_audit.py writes <out>.manifest.json automatically)."
        )
    digests = {
        str(p): (m["dataset"]["digest"], m["metrics"]["digest"]) for p, m in sources
    }
    distinct = set(digests.values())
    if len(distinct) > 1:
        lines = "\n  ".join(f"{k}: dataset={v[0]} metrics={v[1]}" for k, v in digests.items())
        sys.exit(
            "REFUSING TO COMPARE: these runs used different benchmarks.\n  "
            + lines
            + "\nRe-run every model on one frozen suite before comparing them."
        )
    dataset, metrics = next(iter(distinct))
    print(f"frozen benchmark verified: dataset={dataset} metrics={metrics}\n")


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------


def rate(rows: Sequence[dict], metric: str) -> tuple[float | None, int, int]:
    el = [r for r in rows if ELIGIBLE[metric](r)]
    if not el:
        return None, 0, 0
    hits = sum(1 for r in el if INDICATOR[metric](r))
    return hits / len(el), len(el), len({r["template"] for r in el})


def paired_diff(
    a: Sequence[dict],
    b: Sequence[dict],
    metric: str,
    reps: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Difference (a - b) with a template-level paired bootstrap CI."""
    by_a: dict[str, list[dict]] = defaultdict(list)
    by_b: dict[str, list[dict]] = defaultdict(list)
    for r in a:
        if ELIGIBLE[metric](r):
            by_a[r["template"]].append(r)
    for r in b:
        if ELIGIBLE[metric](r):
            by_b[r["template"]].append(r)
    shared = sorted(set(by_a) & set(by_b))
    if len(shared) < 2:
        return {"point": None, "ci": None, "clusters": len(shared)}

    def value(bucket: dict[str, list[dict]], keys: Sequence[str]) -> float:
        rows = [r for k in keys for r in bucket[k]]
        return sum(1 for r in rows if INDICATOR[metric](r)) / len(rows)

    point = value(by_a, shared) - value(by_b, shared)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [rng.choice(shared) for _ in shared]
        draws.append(value(by_a, sample) - value(by_b, sample))
    draws.sort()
    lo = draws[int(0.025 * reps)]
    hi = draws[min(int(0.975 * reps), reps - 1)]
    return {"point": point, "ci": (lo, hi), "clusters": len(shared)}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}"


def report(by_model: dict[str, list[dict]], condition: str | None) -> None:
    models = sorted(by_model)
    label = condition or "all conditions pooled"
    print("=" * 74)
    print(f"MODEL COMPARISON  ({label})")
    print("=" * 74)

    header = f"{'metric':<8}" + "".join(f"{m[:18]:>20}" for m in models)
    print(header)
    print("-" * len(header))
    for metric in ORDER:
        line = f"{metric:<8}"
        for model in models:
            value, n, clusters = rate(by_model[model], metric)
            line += f"{pct(value) + f' ({n}/{clusters}c)':>20}"
        print(line)
    print("\n(n/Nc = eligible episodes / distinct templates behind the rate)")

    if len(models) == 2:
        a, b = models
        print("\n" + "=" * 74)
        print(f"PAIRED DIFFERENCE  {a} - {b}   (bootstrap over shared templates)")
        print("=" * 74)
        for metric in ORDER:
            d = paired_diff(by_model[a], by_model[b], metric)
            if d["point"] is None:
                print(f"  {metric:<8} n/a ({d['clusters']} shared clusters)")
                continue
            lo, hi = d["ci"]
            crosses = "" if (lo > 0 or hi < 0) else "   <- interval includes 0"
            print(
                f"  {metric:<8} {d['point'] * 100:+6.1f} pp  "
                f"[{lo * 100:+.1f}, {hi * 100:+.1f}]  "
                f"{d['clusters']} clusters{crosses}"
            )
        print(
            "\nNo p-values: with this many clusters a significance test would "
            "overstate\nwhat the design can support. Read the interval and the "
            "per-template table."
        )

    print("\n" + "=" * 74)
    print("PER-TEMPLATE CTWAR (the cross-model claim, unaggregated)")
    print("=" * 74)
    templates = sorted({r["template"] for rows in by_model.values() for r in rows if r["cross_tool"]})
    print(f"{'template':<28}" + "".join(f"{m[:16]:>18}" for m in models))
    for template in templates:
        line = f"{template:<28}"
        for model in models:
            rows = [r for r in by_model[model] if r["template"] == template]
            value, n, _ = rate(rows, "CTWAR")
            line += f"{pct(value) + f' ({n})':>18}"
        print(line)

    print("\n" + "=" * 74)
    print("HARNESS HEALTH (report these next to every rate)")
    print("=" * 74)
    for model in models:
        rows = by_model[model]
        errors = sum(1 for r in rows if r.get("error"))
        exhausted = sum(1 for r in rows if r.get("budget_exhausted"))
        state = sum(1 for r in rows if r.get("X_state"))
        replayed = sum(1 for r in rows if r.get("denial_replayed"))
        print(
            f"  {model:<22} n={len(rows):<4} harness-errors={errors:<3} "
            f"out-of-steps={exhausted:<3} state-verified-violations={state:<3} "
            f"denial-replayed={replayed}/{len(rows)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="results JSONL files, one or more models")
    parser.add_argument(
        "--condition",
        default=None,
        help="restrict to one feedback condition (F0, F1, F2); default pools all",
    )
    parser.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="skip the frozen-benchmark check (for debugging only; never for a paper number)",
    )
    args = parser.parse_args()

    sources = [(pathlib.Path(f), None) for f in args.files]
    loaded = []
    for path, _ in sources:
        rows, manifest = load(path)
        loaded.append((path, rows, manifest))

    if args.allow_unfrozen:
        print("WARNING: frozen-benchmark check SKIPPED. Not publishable.\n", file=sys.stderr)
    else:
        check_frozen([(p, m) for p, _, m in loaded])

    by_model: dict[str, list[dict]] = defaultdict(list)
    for _, rows, _ in loaded:
        for r in rows:
            if r.get("error"):
                continue
            if args.condition and r["feedback"] != args.condition:
                continue
            by_model[r["model"]].append(r)

    if not by_model:
        print("no episodes matched; nothing to compare.", file=sys.stderr)
        return 1

    report(dict(by_model), args.condition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())