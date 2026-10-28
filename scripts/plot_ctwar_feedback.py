"""The result figure: where post-denial behaviour GOES as feedback changes.

A CTWAR-only line is the wrong picture. CTWAR falling from F0 to F1 looks
like a safety win on its own, and the mechanism is that the mass moved into
escalation while task completion collapsed -- which only CTWAR, SRR and
ESC-alt on one axis can show. One panel per model, strict primary
denominator, cluster-bootstrap intervals over templates.

Subsets, rates and intervals come from `pda.subsets`; this script computed
the strict denominator inline before, which was a fourth copy of the
paper's primary definition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pda.subsets import (
    alt_class,
    by_model,
    cluster_ci,
    conditions,
    load_all,
    rate,
    strict,
)

# CTWAR is the primary; SRR is what we hope replaces it; ESC-alt is where it
# actually goes. Styles are distinguishable without colour, because a
# reviewer may print the paper in greyscale.
SERIES = (
    ("CTWAR", "workaround attempted", "o", "-"),
    ("SRR", "safe recovery", "s", "--"),
    ("ESC-alt", "unnecessary escalation", "^", ":"),
)


def short_name(model: str) -> str:
    """`ollama:qwen2.5-14b-detour:latest` -> `qwen2.5-14b`.

    Splitting on ':' and taking the last field yields "latest" for every
    model, which silently titles every panel identically.
    """
    name = model.split(":", 1)[-1] if ":" in model else model
    for suffix in ("-detour:latest", "-detour", ":latest"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or model


def panel(ax, rows: list[dict], title: str) -> None:
    conds = conditions(rows)
    for metric, label, marker, style in SERIES:
        values, los, his = [], [], []
        for cond in conds:
            sub = [r for r in rows if r["feedback"] == cond]
            # CTWAR is scored on the strict cross-tool denominator; the
            # continuation metrics are scored on the alternate class. Mixing
            # them would put two different denominators on one axis.
            scope = strict(sub) if metric == "CTWAR" else alt_class(sub)
            value, _, _ = rate(scope, metric)
            ci = cluster_ci(scope, metric)
            values.append(value if value is not None else float("nan"))
            los.append(value - ci[0] if (ci and value is not None) else 0.0)
            his.append(ci[1] - value if (ci and value is not None) else 0.0)
        ax.errorbar(
            conds, values, yerr=[los, his], label=label, marker=marker,
            linestyle=style, capsize=3, linewidth=1.3, markersize=5,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("denial feedback")
    ax.set_ylim(-4, 104)
    ax.grid(alpha=0.25, linewidth=0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", default="figures/feedback_redistribution.png")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("For plotting only: python -m pip install matplotlib")

    rows = load_all(args.files)
    groups = by_model(rows)
    fig, axes = plt.subplots(
        1, len(groups), figsize=(3.6 * len(groups), 3.3), sharey=True
    )
    if len(groups) == 1:
        axes = [axes]

    for ax, (model, mine) in zip(axes, groups):
        panel(ax, mine, short_name(model))
    axes[0].set_ylabel("rate (%)")
    axes[0].legend(fontsize=7, loc="upper center", framealpha=0.9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    print("Bars are 95% cluster-bootstrap intervals over templates, not episodes.")


if __name__ == "__main__":
    main()
