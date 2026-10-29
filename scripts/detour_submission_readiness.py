#!/usr/bin/env python3
"""Read-only Detour readiness checks. AI != human annotation."""

import csv
import re
from collections import Counter
from pathlib import Path

root = Path.cwd()
ai = root / "annotations/ai_review_v3"
human = root / "annotations/submission_blinded_v2"
semantic = root / "annotations/semantic_v3"

pilots = {
    "DT-06e647b2c6dbce70",
    "DT-13ce2c1b988df604",
    "DT-f6ea6a01f46013e3",
    "DT-8b9234b5a040d5c2",
}

def load(path):
    if not path.exists():
        print("MISSING:", path)
        return None
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

print("=== AUTOMATED JUDGE STATUS ===")
judges = {}

for role in "ABCD":
    rows = load(ai / f"ai_{role.lower()}.csv")
    if rows is None:
        continue

    valid = [r for r in rows if r.get("status") == "ok"]
    ids = {r["case_id"] for r in rows}

    print(
        f"Stream {role}: {len(valid)}/263 valid, "
        f"{len(ids)} unique IDs"
    )

    judges[role] = {
        r["case_id"]: r for r in valid
    }

print("\n=== ACTUAL HUMAN REVIEW STATUS ===")

for role in "ab":
    rows = load(human / f"reviewer_{role}.csv")
    if rows is None:
        continue

    accepted = {"yes", "no", "uncertain", "y", "n", "u"}

    finished = sum(
        str(r.get("equivalent", "")).strip().lower()
            in accepted
        and str(r.get("forbidden", "")).strip().lower()
            in accepted
        for r in rows
    )

    print(
        f"Human {role.upper()}: "
        f"{finished}/{len(rows)} completed"
    )

print("\n=== SEMANTIC ARTIFACT REVIEW ===")

for role in "ab":
    rows = load(semantic / f"reviewer_{role}.csv")
    if rows is None:
        continue

    fields = (
        "relevant", "faithful",
        "authorized", "complete"
    )

    finished = sum(
        all(
            str(r.get(k, "")).strip().lower()
                in {"yes", "no", "uncertain", "y", "n", "u"}
            for k in fields
        )
        for r in rows
    )

    print(
        f"Semantic {role.upper()}: "
        f"{finished}/{len(rows)} completed"
    )

print("\n=== SELECTED A/D/C AUTOMATED COMPARISON ===")

if all(r in judges for r in "ADC"):
    common = (
        set(judges["A"])
        & set(judges["D"])
        & set(judges["C"])
    ) - pilots

    print("Non-calibration cases:", len(common))

    for field in ("equivalent", "forbidden"):
        unanimous = sum(
            len({
                judges[r][cid][field]
                for r in "ADC"
            }) == 1
            for cid in common
        )

        for role in "ADC":
            counts = Counter(
                judges[role][cid][field]
                for cid in common
            )
            print(role, field, dict(counts))

        print(
            f"{field}: unanimous on "
            f"{unanimous}/{len(common)}"
        )

print("\n=== CURRENT CODE CHECKS ===")

runner = root / "pda/runner.py"

if runner.is_file():
    source = runner.read_text(encoding="utf-8")

    for line in source.splitlines():
        if re.match(r"\s*ep\.H\s*=", line):
            print("H definition:", line.strip())

            if (
                "not completed" in line
                and "not violated" in line
            ):
                print(
                    "WARNING: H may include clean "
                    "incompletion without an explicit halt."
                )

scenarios = root / "pda/scenarios.py"

if scenarios.is_file():
    source = scenarios.read_text(encoding="utf-8")
    print(
        "Filename-based recovery checks present:",
        "sb.filenames()" in source
    )

manuscript = (
    root / "manuscript/trust2027_reconciled_draft.tex"
)

print(
    "Manuscript:",
    "FOUND" if manuscript.is_file() else "MISSING"
)

print("\nDONE: Read-only checks completed.")
print("Automated labels are NOT human annotations.")
