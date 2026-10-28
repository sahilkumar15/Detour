"""The one definition of every analysis subset, loader and rate helper.

Before this module the strict primary denominator -- cross-tool eligible,
non-adversarial, excluding T4's broad equivalence -- was written out in four
places (`rq4_analysis`, `d1_analysis`, `scripts/check_v7_evidence`,
`scripts/plot_ctwar_feedback`), the trace loader in five with four different
implementations, and `pct`/`rate` in five. Four copies of a denominator is
four chances for the paper's primary number to disagree with itself, and a
reader cannot tell which copy produced a published rate.

Everything here is about reading traces back. The metric definitions
themselves stay in `pda.metrics`: `rate` dispatches through its ELIGIBILITY
and INDICATOR tables rather than restating any predicate, so a diagnostic
cannot drift from the definitions the metric digest covers.

Template roles live here because they are facts about the suite, the same
kind of annotation as the alias map, and every consumer needs them to agree.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .runner import Episode

# --------------------------------------------------------------------------
# Template roles. A template is named here when the analysis treats it
# specially, with the reason it is treated that way.
# --------------------------------------------------------------------------

#: Access grant treated as equivalent to an attachment. The broadest
#: equivalence claim in the suite, so the primary denominator excludes it and
#: reports it as a sensitivity instead.
BROAD_EQUIVALENCE = "T4"

#: Injected instruction in the document body. Excluded from the primary
#: denominator because the workaround is planted rather than volunteered.
ADVERSARIAL = "T6"

#: Channel broadcast scored as equivalent to an individual recipient. Its
#: audience is not the same size, so it gets its own sensitivity.
CHANNEL_EQUIVALENCE = "T8"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSONL trace file into raw rows."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no such trace file: {p}")
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_all(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read several trace files into one row list."""
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(load_rows(p))
    if not rows:
        raise SystemExit("no episodes loaded")
    return rows


def as_episodes(rows: Sequence[dict[str, Any]]) -> list[Episode]:
    """Rehydrate rows so the shipped metric predicates can score them."""
    return [Episode.from_row(r) for r in rows]


def by_model(rows: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group rows by model, in a stable order."""
    return [
        (m, [r for r in rows if r["model"] == m])
        for m in sorted({r["model"] for r in rows})
    ]


def conditions(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Feedback conditions present, in a stable order."""
    return sorted({r["feedback"] for r in rows})


# --------------------------------------------------------------------------
# Subsets. Each takes and returns raw rows, so callers can compose them.
# --------------------------------------------------------------------------


def _is(row: dict[str, Any], template_prefix: str) -> bool:
    return str(row["template"]).startswith(template_prefix)


def eligible(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-tool eligible: an effect-equivalent alternate route exists.

    CTWAR's denominator. An episode with no alternate route cannot evidence
    a choice not to take one.
    """
    return [r for r in rows if r["cross_tool"]]


def strict(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The paper's PRIMARY denominator: eligible, no T6, no T4."""
    return [
        r
        for r in eligible(rows)
        if not r["adversarial"]
        and not _is(r, BROAD_EQUIVALENCE)
        and not _is(r, ADVERSARIAL)
    ]


def strict_narrow(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict, and also without T8's channel-to-recipient equivalence."""
    return [r for r in strict(rows) if not _is(r, CHANNEL_EQUIVALENCE)]


def broad(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-adversarial but including T4. A sensitivity, not the headline."""
    return [r for r in eligible(rows) if not r["adversarial"]]


def adversarial(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The adversarial arm, reported on its own."""
    return [r for r in eligible(rows) if r["adversarial"]]


def alt_class(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Episodes whose correct continuation is permitted local work."""
    return [r for r in rows if r["continuation_class"] == "alt"]


#: Name -> subset function, for tools that iterate specifications.
SUBSETS: dict[str, Callable[[Sequence[dict[str, Any]]], list[dict[str, Any]]]] = {
    "strict": strict,
    "strict_narrow": strict_narrow,
    "broad": broad,
    "eligible": eligible,
    "adversarial": adversarial,
}


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------


def pct(numerator: int, denominator: int) -> str:
    """`n/d (p%)`, and never a ZeroDivisionError on an empty cell."""
    if not denominator:
        return f"{numerator}/0 (n/a)"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def rate(rows: Sequence[dict[str, Any]], metric: str) -> tuple[float | None, int, int]:
    """`(percent, n_eligible, n_clusters)` for a metric over these rows.

    Dispatches through `pda.metrics`, so this is the shipped definition
    rather than a second one. `percent` is None when nothing is eligible,
    which a caller must render rather than print as a zero.
    """
    from .metrics import ELIGIBILITY, INDICATOR

    elig, ind = ELIGIBILITY[metric], INDICATOR[metric]
    eps = [e for e in as_episodes(rows) if elig(e)]
    if not eps:
        return None, 0, 0
    hits = sum(1 for e in eps if ind(e))
    return 100.0 * hits / len(eps), len(eps), len({e.template for e in eps})


def counts(rows: Sequence[dict[str, Any]], metric: str) -> tuple[int, int]:
    """`(numerator, denominator)` for a metric over these rows."""
    from .metrics import ELIGIBILITY, INDICATOR

    elig, ind = ELIGIBILITY[metric], INDICATOR[metric]
    eps = [e for e in as_episodes(rows) if elig(e)]
    return sum(1 for e in eps if ind(e)), len(eps)


def cluster_ci(
    rows: Sequence[dict[str, Any]],
    metric: str = "CTWAR",
    *,
    reps: int = 5000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% cluster-bootstrap interval in percentage points, or None.

    Clusters are templates, never episodes: three entity-renamed variants of
    one template are not three independent designs, and resampling episodes
    would report an interval far narrower than the design supports. Returns
    None below two clusters, where resampling cannot say anything -- a
    caller must print that rather than an interval.

    Generic over `metric` because this was previously a CTWAR-only helper in
    `rq4_analysis`; every other rate was then published without an interval
    or with one computed somewhere else.
    """
    import random
    from collections import defaultdict

    from .metrics import ELIGIBILITY, INDICATOR

    elig, ind = ELIGIBILITY[metric], INDICATOR[metric]
    grouped: dict[str, list[bool]] = defaultdict(list)
    for e in as_episodes(rows):
        if elig(e):
            grouped[e.template].append(bool(ind(e)))
    keys = list(grouped)
    if len(keys) < 2:
        return None

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(reps):
        picked = [x for k in (rng.choice(keys) for _ in keys) for x in grouped[k]]
        draws.append(100.0 * sum(picked) / len(picked))
    draws.sort()
    lo = draws[int(0.025 * reps)]
    hi = draws[min(int(0.975 * reps), reps - 1)]
    return lo, hi
