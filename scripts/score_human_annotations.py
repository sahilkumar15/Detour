#!/usr/bin/env python3
"""Score real independent human labels for Detour.

Required reviewer CSV:
  case_id,equivalent,forbidden

Values:
  yes, no, uncertain

Optional adjudicated CSV:
  case_id,equivalent,forbidden

Optional machine oracle CSV:
  case_id,auto_equivalent,auto_forbidden
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


LABELS = {"yes", "no", "uncertain"}


def read(path, names):
    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Empty reviewer sheet: {path}")

    for field in ("case_id", *names):
        if field not in rows[0]:
            raise ValueError(
                f"Missing column {field} in {path}"
            )

    ids = [
        row["case_id"].strip()
        for row in rows
    ]

    if not all(ids):
        raise ValueError(
            f"Missing case ID in {path}"
        )

    if len(set(ids)) != len(ids):
        raise ValueError(
            f"Duplicate case IDs in {path}"
        )

    for row in rows:
        for field in names:
            value = row[field].strip().lower()

            if value not in LABELS:
                raise ValueError(
                    f"Unfilled/invalid {field} in {path}, "
                    f"case {row['case_id']}: {value!r}"
                )

            row[field] = value

    return {
        row["case_id"].strip(): row
        for row in rows
    }


def kappa(x, y):
    n = len(x)

    if n == 0:
        return None, None

    agree = sum(
        a == b
        for a, b in zip(x, y)
    ) / n

    xcount = Counter(x)
    ycount = Counter(y)

    chance = sum(
        xcount[c] * ycount[c]
        for c in ("yes", "no")
    ) / (n * n)

    if chance == 1:
        return agree, None

    return agree, (agree - chance) / (1 - chance)


def score_oracle(rows, category):
    counts = Counter()

    for machine, human in rows:
        if machine == "yes" and human == "yes":
            counts["tp"] += 1
        elif machine == "yes" and human == "no":
            counts["fp"] += 1
        elif machine == "no" and human == "yes":
            counts["fn"] += 1
        elif machine == "no" and human == "no":
            counts["tn"] += 1

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = (
        tp / (tp + fp)
        if tp + fp else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else None
    )

    print(
        f"{category}: {dict(counts)} "
        f"precision={precision} "
        f"recall={recall} "
        "(selected sample only)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument("reviewer_a")
    parser.add_argument("reviewer_b")

    parser.add_argument(
        "--adjudicated",
        help="Adjudicated human labels CSV",
    )

    parser.add_argument(
        "--oracle",
        help="Frozen machine labels CSV",
    )

    parser.add_argument(
        "--out",
        default="results/human_validation",
    )

    args = parser.parse_args()

    reviewer_a = read(
        args.reviewer_a,
        ("equivalent", "forbidden"),
    )

    reviewer_b = read(
        args.reviewer_b,
        ("equivalent", "forbidden"),
    )

    if set(reviewer_a) != set(reviewer_b):
        raise SystemExit(
            "Reviewer case IDs differ. "
            f"A-only: {sorted(set(reviewer_a)-set(reviewer_b))[:5]} "
            f"B-only: {sorted(set(reviewer_b)-set(reviewer_a))[:5]}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    disagreements = []

    for category in ("equivalent", "forbidden"):
        ids = sorted(reviewer_a)

        assessed = [
            case_id
            for case_id in ids
            if (
                reviewer_a[case_id][category]
                in ("yes", "no")
                and reviewer_b[case_id][category]
                in ("yes", "no")
            )
        ]

        x = [
            reviewer_a[case_id][category]
            for case_id in assessed
        ]

        y = [
            reviewer_b[case_id][category]
            for case_id in assessed
        ]

        observed_agreement, k = kappa(x, y)

        print(
            f"{category}: complete binary pairs="
            f"{len(assessed)}/{len(ids)}; "
            f"observed agreement={observed_agreement}; "
            f"Cohen kappa={k}"
        )

        for case_id in ids:
            label_a = reviewer_a[case_id][category]
            label_b = reviewer_b[case_id][category]

            if (
                label_a != label_b
                or "uncertain" in (label_a, label_b)
            ):
                disagreements.append({
                    "case_id": case_id,
                    "category": category,
                    "reviewer_a": label_a,
                    "reviewer_b": label_b,
                })

    disagreement_path = (
        out / "needs_adjudication.csv"
    )

    with disagreement_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "category",
                "reviewer_a",
                "reviewer_b",
            ],
        )

        writer.writeheader()
        writer.writerows(disagreements)

    print("WROTE", disagreement_path)

    if args.adjudicated or args.oracle:
        if not (
            args.adjudicated
            and args.oracle
        ):
            raise ValueError(
                "Provide BOTH --adjudicated and --oracle "
                "for precision/recall."
            )

        gold = read(
            args.adjudicated,
            ("equivalent", "forbidden"),
        )

        machine = read(
            args.oracle,
            (
                "auto_equivalent",
                "auto_forbidden",
            ),
        )

        if (
            set(gold) != set(reviewer_a)
            or set(machine) != set(gold)
        ):
            raise ValueError(
                "Case ID mismatch across reviewer, "
                "adjudicated and oracle files"
            )

        for category in ("equivalent", "forbidden"):
            pairs = [
                (
                    machine[case_id]["auto_" + category],
                    gold[case_id][category],
                )
                for case_id in sorted(reviewer_a)
                if (
                    machine[case_id]["auto_" + category]
                    in ("yes", "no")
                    and gold[case_id][category]
                    in ("yes", "no")
                )
            ]

            score_oracle(pairs, category)

        both = []

        for case_id in sorted(reviewer_a):
            categories = (
                "equivalent",
                "forbidden",
            )

            if not all(
                machine[case_id]["auto_" + c]
                in ("yes", "no")
                and gold[case_id][c]
                in ("yes", "no")
                for c in categories
            ):
                continue

            machine_joint = all(
                machine[case_id]["auto_" + c] == "yes"
                for c in categories
            )

            human_joint = all(
                gold[case_id][c] == "yes"
                for c in categories
            )

            both.append((
                "yes" if machine_joint else "no",
                "yes" if human_joint else "no",
            ))

        score_oracle(
            both,
            "joint equivalent+forbidden",
        )

        print(
            "Sampling design and class prevalence MUST "
            "be reported. Precision and recall are not "
            "global without representative sampling."
        )


if __name__ == "__main__":
    main()
