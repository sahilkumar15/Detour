#!/usr/bin/env python3
"""Run additional AI-only Detour judges using the frozen V3 rubric."""

import argparse
from pathlib import Path

import run_detour_ai_review_v3 as v3

parser = argparse.ArgumentParser()
parser.add_argument("--role", choices=["C", "D"], required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

if args.limit < 0:
    parser.error("--limit must be nonnegative")

cases = v3.read_cases()

# Preserve the original effect-equivalence rubric.
v3.PROMPTS[args.role] = v3.A_PROMPT.replace(
    "AI Reviewer A",
    f"AI Reviewer {args.role}",
)

# No fallback: attribution remains to the selected model.
v3.MODELS[args.role] = (
    args.model,
    "__NO_FALLBACK__",
)

out = Path("annotations/ai_review_v3").resolve()

v3.run_role(
    args.role,
    cases,
    out,
    args.limit,
)

path = out / f"ai_{args.role.lower()}.csv"
_, rows = v3.read_csv(path)

valid = sum(row["status"] == "ok" for row in rows)

print(f"\nAI Reviewer {args.role}: {valid}/263 valid")

if args.limit == 0 and valid != 263:
    raise SystemExit(
        "Incomplete: inspect needs_review cases and rerun."
    )
