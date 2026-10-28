#!/usr/bin/env bash
set -Eeuo pipefail

cd /scratch/sahil/projects/ACM_Trust/pda_audit

PREFIX='results/grid-v8-gptoss20-ollama'
REF='results/grid-v8-qwen7.manifest.json'
MODEL='ollama:gptoss20-detour:latest'
MAX_ATTEMPTS=3
RESET_MODEL_ON_ERROR="${RESET_MODEL_ON_ERROR:-0}"

mkdir -p logs results

# Lock against another copy of this launcher.
# Do not run while an older manually started audit is active.
exec 9>results/.gptoss20-ollama-resume.lock
flock -n 9 || {
  echo 'Another recovery launcher is running.' >&2
  exit 2
}

# Check required files.
for path in \
    "$REF" \
    "${PREFIX}.jsonl" \
    "${PREFIX}.manifest.json"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing $path" >&2
    exit 2
  fi
done

# Check Ollama connection.
curl -fsS --max-time 10 \
  http://127.0.0.1:11434/api/version >/dev/null || {
  echo 'ERROR: Ollama is unreachable on this node.' >&2
  exit 2
}

# Verify that saved records belong to this frozen experiment.
python - "$PREFIX" "$REF" <<'PY'
import json
import pathlib
import sys
from collections import Counter

from pda.freeze import verify_manifest
from pda.scenarios import build_suite

prefix, reference = sys.argv[1:]
suite = build_suite()

for path in (reference, prefix + ".manifest.json"):
    warnings = verify_manifest(path, suite)
    print("Freeze verified:", path)
    for warning in warnings:
        print("WARNING:", warning)

manifest = json.loads(
    pathlib.Path(prefix + ".manifest.json").read_text()
)

run = manifest.get("run", {})

assert run.get("agents") == [
    "ollama:gptoss20-detour:latest"
], "Wrong model"

assert run.get("conditions") == [
    "F0", "F1", "F2"
], "Wrong feedback conditions"

assert run.get("reps") == 3
assert run.get("max_steps") == 10

rows = [
    json.loads(line)
    for line in pathlib.Path(prefix + ".jsonl")
                       .read_text().splitlines()
    if line.strip()
]

keys = [
    (
        r["model"],
        r["feedback"],
        r["task_id"],
        r["repetition"],
    )
    for r in rows
]

assert len(keys) == len(set(keys)), "Duplicate episodes"
assert len(rows) <= 216, "Too many episodes"

assert all(
    r["model"] == "ollama:gptoss20-detour:latest"
    and not r.get("error")
    for r in rows
), "Mixed-model or error records"

counts = Counter(r["feedback"] for r in rows)
assert all(counts[f] <= 72 for f in ("F0", "F1", "F2"))

print("Saved:", len(rows), "/216")
print("By condition:", dict(counts))
print("PASS: Existing episodes are consistent.")
PY

# Resume the same frozen experiment.
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do

  LOG="logs/gptoss20-ollama-resume-attempt${attempt}.log"

  echo "========================================="
  echo "RESUME ATTEMPT $attempt / $MAX_ATTEMPTS"
  echo "========================================="

  if python run_audit.py \
      --agents "$MODEL" \
      --conditions F0,F1,F2 \
      --reps 3 \
      --max-steps 10 \
      --max-tokens 900 \
      --retries 4 \
      --verify-freeze "$REF" \
      --out "$PREFIX" \
      2>&1 | tee "$LOG"; then

    echo "Runner completed successfully."
    break

  fi

  echo "Runner stopped; see $LOG" >&2

  # Only automatically retry the identified repeat-token failure.
  if ! grep -q 'token repeat limit reached' "$LOG"; then
    echo "Unexpected failure; stopping for diagnosis." >&2
    exit 2
  fi

  if [[ "$attempt" -eq "$MAX_ATTEMPTS" ]]; then
    echo "Repeat-token failure persists." >&2
    echo "Saved traces are preserved." >&2
    exit 2
  fi

  # Optional recovery when this is your exclusive Ollama model.
  if [[ "$RESET_MODEL_ON_ERROR" == '1' ]]; then
    echo "Unloading model from memory..."
    ollama stop gptoss20-detour:latest || true
  else
    echo "Retrying without changing the loaded model."
  fi

done

# Never accept an incomplete run as final results.
python - "$PREFIX" <<'PY'
import json
import pathlib
import sys
from collections import Counter

base = pathlib.Path(sys.argv[1])

rows = [
    json.loads(line)
    for line in base.with_suffix(".jsonl")
                    .read_text().splitlines()
    if line.strip()
]

manifest = json.loads(
    base.with_suffix(".manifest.json").read_text()
)

keys = [
    (
        r["model"],
        r["task_id"],
        r["feedback"],
        r["repetition"],
    )
    for r in rows
]

counts = Counter(r["feedback"] for r in rows)

assert len(rows) == 216, f"Only {len(rows)}/216 completed"
assert len(set(keys)) == 216, "Duplicate episode keys"
assert all(
    counts[f] == 72
    for f in ("F0", "F1", "F2")
), "Incomplete feedback conditions"

assert not any(r.get("error") for r in rows)
assert manifest["run"]["complete"] is True

print("=" * 50)
print("PASS: COMPLETE DETOUR AUDIT")
print("Episodes: 216/216")
print("F0: 72/72")
print("F1: 72/72")
print("F2: 72/72")
print("Manifest: complete")
print("=" * 50)
PY
