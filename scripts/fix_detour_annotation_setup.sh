#!/usr/bin/env bash
set -Eeuo pipefail

cd /scratch/sahil/projects/ACM_Trust/pda_audit

if [[ ! -f scripts/prepare_detour_annotations.py ||
      ! -f pda/scenarios.py ]]; then
    echo "ERROR: Missing Detour generator or pda package."
    exit 2
fi

mkdir -p logs results/human_validation

PUBLIC='annotations/submission_blinded_v2'
PRIVATE='annotations/private_oracle_v2'

mkdir -p "$PUBLIC" "$PRIVATE"
chmod 700 "$PRIVATE"

echo "========== REPAIRING PYTHON IMPORT =========="

python - <<'PY'
from pathlib import Path

p = Path("scripts/prepare_detour_annotations.py")
source = p.read_text(encoding="utf-8")

marker = "from __future__ import annotations\n"

patch = (
    "\n# Allow imports from the Detour project root.\n"
    "import sys as _detour_sys\n"
    "from pathlib import Path as _detour_Path\n"
    "_detour_sys.path.insert(0, str("
    "_detour_Path(__file__).resolve().parents[1]))\n"
)

if marker not in source:
    raise SystemExit(
        "ERROR: Unexpected generator format. No changes made."
    )

if "_detour_sys.path.insert" not in source:
    backup = p.with_name(
        "prepare_detour_annotations.py.before_import_fix"
    )

    if not backup.exists():
        backup.write_text(source, encoding="utf-8")

    p.write_text(
        source.replace(marker, marker + patch, 1),
        encoding="utf-8"
    )
    print("PASS: Import fixed. Backup:", backup)
else:
    print("PASS: Import already fixed")
PY

python -m py_compile \
  scripts/prepare_detour_annotations.py \
  scripts/score_human_annotations.py

echo "========== VERIFYING FROZEN BENCHMARK =========="

python - <<'PY'
from pda.scenarios import build_suite
from pda.freeze import verify_manifest

suite = build_suite()

assert len(suite) == 24, (
    f"Expected 24 scenarios; found {len(suite)}"
)

warnings = verify_manifest(
    "results/grid-v8-qwen7.manifest.json",
    suite
)

for warning in warnings:
    print("WARNING:", warning)

print("PASS: 24 scenarios loaded")
print("PASS: Frozen dataset and metrics verified")
PY

echo "========== PREPARING HUMAN ANNOTATION =========="

if [[ -f "$PUBLIC/cases.csv" &&
      -f "$PUBLIC/reviewer_a.csv" &&
      -f "$PUBLIC/reviewer_b.csv" &&
      -f "$PRIVATE/machine_labels.csv" &&
      -f "$PRIVATE/sampling_manifest.json" ]]; then

    echo "PASS: Existing v2 queue found."
    echo "Preserving all previously generated files."

elif [[ -n "$(find "$PUBLIC" "$PRIVATE" \
    -mindepth 1 -maxdepth 1 -type f -print -quit)" ]]; then

    echo "ERROR: Partial v2 outputs detected."
    echo "Inspect them before regenerating."
    exit 2

else
    python -u scripts/prepare_detour_annotations.py \
      results/grid-v8-qwen7.jsonl \
      results/grid-v8-qwen14.jsonl \
      results/grid-v9-mistral24.jsonl \
      results/grid-v8-gptoss20-ollama.jsonl \
      --controls 100 \
      --seed 20261028 \
      --out "$PUBLIC" \
      --private-out "$PRIVATE" \
      | tee logs/annotation_queue_v2.log
fi

echo "========== VERIFYING GENERATED FILES =========="

PUBLIC="$PUBLIC" PRIVATE="$PRIVATE" python - <<'PY'
import csv
import json
import hashlib
import os
from pathlib import Path

public = Path(os.environ["PUBLIC"])
private = Path(os.environ["PRIVATE"])

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"Empty file: {path}"
    return rows

cases = read_csv(public / "cases.csv")
review_a = read_csv(public / "reviewer_a.csv")
review_b = read_csv(public / "reviewer_b.csv")
machine = read_csv(private / "machine_labels.csv")

manifest = json.loads(
    (private / "sampling_manifest.json").read_text()
)

case_ids = [r["case_id"] for r in cases]

assert len(case_ids) == len(set(case_ids)), (
    "Duplicate case IDs"
)

for name, records in [
    ("Reviewer A", review_a),
    ("Reviewer B", review_b),
    ("Private machine oracle", machine),
]:
    assert len(records) == len(cases)
    assert {r["case_id"] for r in records} == set(case_ids)
    print(f"PASS: {name}: {len(records)} matching cases")

allowed = {"", "yes", "no", "uncertain"}

for record in review_a + review_b:
    assert record["equivalent"].strip().lower() in allowed
    assert record["forbidden"].strip().lower() in allowed

for field in cases[0]:
    assert not field.startswith(("auto_", "_"))
    assert field not in ("model", "feedback", "stratum")

assert manifest["total_cases"] == len(cases)

checksum = hashlib.sha256(
    (public / "cases.csv").read_bytes()
).hexdigest()

assert checksum == manifest["case_file_sha256"]

for name, records in [
    ("A", review_a),
    ("B", review_b),
]:
    filled = sum(
        r["equivalent"].strip().lower()
            in {"yes", "no", "uncertain"}
        and r["forbidden"].strip().lower()
            in {"yes", "no", "uncertain"}
        for r in records
    )
    print(f"Reviewer {name}: {filled}/{len(records)} completed")

print("==============================================")
print("PASS: ANNOTATION DATASET VALIDATED")
print("==============================================")
print("Completed audit episodes:", manifest["episodes_read"])
print("Human-review cases:", len(cases))
print("Denied proposals:", manifest["denied_steps_included"])
print("Allowed controls:", manifest["allowed_controls_selected"])
print("Machine labels: PRIVATE")
print("No human labels generated automatically.")
PY

echo "========== ALL FILES READY =========="
ls -lh "$PUBLIC"
ls -lh "$PRIVATE"

echo "SUCCESS: Annotation setup completed."
echo "Do not share the private_oracle_v2 directory."
echo "Do not share the old submission_blinded directory:"
echo "it contains a legacy answer_key_private.csv."
