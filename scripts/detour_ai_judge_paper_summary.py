#!/usr/bin/env python3
"""Four-AI-judge research report. NOT human annotation."""

import argparse
import json
from pathlib import Path

NAMES = {
    "A": "GPT-OSS 20B",
    "B": "Mistral 24B",
    "C": "Llama 3.3 70B",
    "D": "Qwen3.5 35B-A3B",
}

# Exploratory shortlist, not human-ground-truth ranking.
SELECTED = ("A", "D", "B")


def latex_escape(value):
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "annotations/private_oracle_v2/"
            "four_judge_final_audit/"
            "four_judge_metrics_PRIVATE.json"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/paper_ai_judge_analysis"),
    )

    args = parser.parse_args()

    if not args.metrics.is_file():
        parser.error(f"Missing metrics: {args.metrics}")

    stats = json.loads(
        args.metrics.read_text(encoding="utf-8")
    )

    comparisons = stats["machine_comparisons"]
    distribution = stats["model_distributions"]
    pairwise = stats["pairs"]

    for role in NAMES:
        if role not in comparisons or role not in distribution:
            parser.error(f"Missing Reviewer {role}")

        counts = distribution[role]["equivalent"]

        n = sum(
            counts.get(k, 0)
            for k in ("yes", "no", "uncertain")
        )

        if n != 259:
            parser.error(
                f"Reviewer {role}: expected 259, got {n}"
            )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def pct(x):
        return f"{x:.1f}" if x is not None else "NA"

    def coeff(x):
        return f"{x:.3f}" if x is not None else "NA"

    lines = [
        "# Detour Four-AI-Judge Sensitivity",
        "",
        "NOT independent human annotation.",
        "NOT accuracy against established human ground truth.",
        "259 non-pilot selected action proposals.",
        "",
        "## Complete four-model results",
        "",
        "| Model | Eq yes | Eq agreement | "
        "Oracle-positive overlap | Eq kappa | "
        "Forbidden agreement | Forbidden kappa |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for role in ("A", "D", "B", "C"):
        e = comparisons[role]["equivalent"]
        f = comparisons[role]["forbidden"]
        yes = distribution[role]["equivalent"].get("yes", 0)

        lines.append(
            f"| {NAMES[role]} ({role}) "
            f"| {yes}/259 "
            f"| {pct(e['agreement_all_pct'])}% "
            f"| {pct(e['positive_agreement_all_pct'])}% "
            f"| {coeff(e['kappa_binary'])} "
            f"| {pct(f['agreement_all_pct'])}% "
            f"| {coeff(f['kappa_binary'])} |"
        )

    lines.extend([
        "",
        "## Exploratory three-model shortlist",
        "A: GPT-OSS 20B — strongest oracle alignment.",
        "D: Qwen3.5 35B — second equivalence candidate.",
        "B: Mistral 24B — contrasting forbiddenness judge.",
        "",
        "This selection is post-hoc, not a human quality ranking.",
        "Llama C remains in the complete four-model analysis.",
        "",
        "## Private discrepancy priorities",
        str(
            stats.get("consensus", {}).get(
                "review_priorities", {}
            )
        ),
        "",
        "Do not treat AI consensus as human ground truth.",
    ])

    (out / "THREE_JUDGE_RESEARCH_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    tex = [
        r"% Automated reviewer sensitivity, NOT human labels.",
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Automated evaluator sensitivity on "
        r"259 non-pilot proposals. Percentages measure "
        r"agreement with Detour's frozen machine oracle, "
        r"not independent human-ground-truth accuracy. "
        r"Binary $\kappa$ excludes uncertain judgments.}",
        r"\label{tab:ai_judge_sensitivity}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"AI judge & Eq. agree (\%) & "
        r"Eq. positive overlap (\%) & "
        r"Forbidden agree (\%) & Eq. $\kappa$ \\",
        r"\midrule",
    ]

    for role in SELECTED:
        e = comparisons[role]["equivalent"]
        f = comparisons[role]["forbidden"]

        row = (
            f"{latex_escape(NAMES[role])} & "
            f"{pct(e['agreement_all_pct'])} & "
            f"{pct(e['positive_agreement_all_pct'])} & "
            f"{pct(f['agreement_all_pct'])} & "
            f"{coeff(e['kappa_binary'])}"
        )

        tex.append(row + r" \\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    (out / "ai_judge_sensitivity.tex").write_text(
        "\n".join(tex) + "\n",
        encoding="utf-8",
    )

    # Four-reviewer agreement heatmap.
    try:
        import matplotlib
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import numpy as np

        roles = ["A", "B", "C", "D"]
        matrix = np.eye(4)

        for i, a in enumerate(roles):
            for j, b in enumerate(roles):
                if i == j:
                    continue

                key = "".join(sorted((a, b)))

                matrix[i, j] = pairwise[key][
                    "equivalent"
                ]["kappa_3label"]

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(
            matrix,
            vmin=0,
            vmax=1,
            cmap="Blues",
        )

        ax.set_xticks(
            range(4),
            [NAMES[x] for x in roles],
            rotation=25,
            ha="right",
        )

        ax.set_yticks(
            range(4),
            [NAMES[x] for x in roles],
        )

        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                )

        ax.set_title(
            "AI-to-AI Effect-Equivalence Agreement"
        )

        fig.colorbar(
            im, ax=ax, label="Cohen's kappa"
        )

        fig.tight_layout()

        fig.savefig(
            out / "ai_equivalence_agreement.pdf"
        )

        plt.close(fig)

    except ImportError:
        print(
            "WARNING: matplotlib/numpy missing; "
            "figure skipped"
        )

    print("\n".join(lines))
    print("\nPASS: Paper analysis generated at", out)


if __name__ == "__main__":
    main()
