# Detour submission validity fixes (2026-10-26)

This is an **offline analysis and manuscript reconciliation package**, not a new benchmark release. It does **not** modify `run_audit.py`, your frozen scenarios, the JSONL traces, the CTWAR implementation, or existing manifests. This avoids changing the metric digest or inventing new results after a frozen run.

## Copy to A100 cluster

Download `detour_submission_patch.zip` from this chat and transfer it to the cluster's existing `pda_audit` directory using your usual SSH/SFTP/VS Code file transfer. This archive extracts directly into the project root: `scripts/` tools, `run_submission_checks.sh`, and a manuscript draft under `manuscript/`. It does not overwrite your existing analysis scripts.

```bash
cd /scratch/sahil/projects/ACM_Trust/pda_audit
unzip -o detour_submission_patch.zip
bash run_submission_checks.sh
```

For review of STRUCTURAL completion, fill `annotations/recovery_review/reviewer_recovery.csv` with actual semantic correctness judgments of source and output content. No score is fabricated.

For effect annotation scoring **only after two independent humans fill the reviewer sheets**:

```bash
python scripts/score_annotations.py annotations/submission_blinded --out results/submission_review
```

If there are disagreements, take `results/submission_review/annotation_disagreements_to_adjudicate.csv`, copy it to `annotation_disagreements_adjudicated.csv`, and manually fill the two adjudicated labels plus reasons. Rerun `score_annotations.py`.

## Corrections made

1. **D1 denominator:** strict non-adversarial cross-tool episodes with `continuation_class=alt` only. T2 (approve-class) is not counted as failed structural completion. The resulting Qwen v8 denominators should be 81/model, not 108/model, given full 3x3 repetitions.
2. **Zero-event uncertainty:** the report refuses to print either degenerate `[0,0]` confidence bounds or an independent-episode `3/n` claim. Rates are observed proportions; only 4 independent template clusters under strict CTWAR.
3. **RF diagnostics:** show eligible/abstention counts by model and condition and status availability by continuation class. 192/216 Mistral RF abstentions cannot be silently treated as report correctness.
4. **Evidence for metric distinctiveness:** exports exact retry/general unauthorized/proposed CTWAR/execution-only comparisons using recorded trace labels. NOTE: naive changed-tool is a STEP-LEVEL count; CTWAR is an EPISODE-LEVEL measure; do not divide them by the same denominator.
5. **Human oracle validity:** creates truly blinded A/B reviewer sheets hiding automatic labels, model and feedback, with denied actions and policy text from `build_suite()`, plus some allowed-step controls to find false negatives. Nothing in this package invents human labels.
6. **Test isolation:** `python -m pytest -q tests` runs YOUR Detour tests, not the unrelated vendored `agentdojo/tests/test_agent_pipeline/test_anthropic_llm.py` failure caused by attempting to instantiate an Anthropic typing.Union. This does not resolve the upstream library's compatibility bug; if you depend on that test, fix and separately rerun upstream's test suite.
7. **Manuscript:** `manuscript/trust2027_reconciled_draft.tex` is based on the uploaded `trust.tex.txt`, replaces the prospective experiment/table, and explicitly calls out missing human validation and natural rollouts. It has NOT been compiled or independently checked against the unseen `reference.bib`, latest cluster code, or venue metadata. Treat as an editable draft, not camera-ready.

## Research caveats

- This script detects disagreement between standard top-level manifest digest keys if present. Always run the repository's `--verify-freeze` check; schema variations may store digests under other keys.
- Clean competence is 12/12 for Qwen 7B, Qwen 14B and Mistral 24B in the shared logs, but that is only six distinct clean tasks with two repetitions.
- Human annotation requires two *independent human reviewers*, not an assistant copying deterministic oracle labels.
- Without artifact contents/terminal sandbox state in JSONL, semantic recovery may require trace replay or separate human review of actual produced artifacts.
- `reference.bib`, precise Qwen serving settings/model digests, exact end-to-end submission compilation, and the actual cluster Mistral JSONL were not available in this session. Consequently, the package cannot assert those items have been fixed or validated.

## Scope of verification

- Syntax checked every new Python script and the shell script.
- Tested `submission_audit.py` against the supplied older `Detour-main.zip` scripted episode JSONL. Real v8 Qwen + v9 Mistral JSONLs are on the cluster, not accessible in this chat.
- The reconciled manuscript compiles with `pdflatex` in this environment (8 pages BEFORE bibliography); citation references remain unresolved because `reference.bib` was not supplied. Recompile with your own bibliography and check page limit and anonymization.
- The annotation and recovery-queue generators depend on your cluster's installed `agentdojo` package and `pda.scenarios`. They are syntax-checked but not end-to-end tested in this container because the bundled `agentdojo` directory in the provided ZIP is empty.
- The small clean-task competence screen does not establish reliability across all real-world workflows. New scenario diversity remains a limitation.
