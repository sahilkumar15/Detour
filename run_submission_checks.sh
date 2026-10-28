#!/usr/bin/env bash
set -Eeuo pipefail

cd /scratch/sahil/projects/ACM_Trust/pda_audit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/submission_review

echo '=== Detour self-tests ==='
python run_audit.py --self-test

echo '=== Detour-owned pytest tests (not vendored AgentDojo) ==='
python -m pytest -q tests

echo '=== Verify available episode files ==='
FILES=()
for prefix in results/grid-v8-qwen7 results/grid-v8-qwen14 results/grid-v9-mistral24; do
  if [[ -f "${prefix}.jsonl" ]]; then
     FILES+=("${prefix}.jsonl")
     echo "FOUND: ${prefix}.jsonl"
  else
     echo "MISSING: ${prefix}.jsonl"
  fi
done
if [[ "${#FILES[@]}" -eq 0 ]];then echo 'ERROR: no experiment traces found';exit 1;fi

python scripts/submission_audit.py "${FILES[@]}" --out results/submission_review
python scripts/prepare_blinded_annotations.py "${FILES[@]}" --out annotations/submission_blinded --allowed-controls 100
python scripts/prepare_recovery_review.py "${FILES[@]}" --out annotations/recovery_review --max 150

echo
printf '%s\n' 'Review results/submission_review/review_report.md and the 2 independent blinded annotation sheets.'
printf '%s\n' 'Do NOT claim annotation precision or recall until independent reviewers complete them.'
