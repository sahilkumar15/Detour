# Detour submission evidence: offline diagnostic

648 recorded episodes, 3 model identities.

## Main metrics (observed proportions; DO NOT assume episode independence)
- **Mistral Small 3.2 24B / F0**: CTWAR 0/36 (0.0%); PUAR 15/72 (20.8%); SRR 9/54 (16.7%); ESC-alt 0/54 (0.0%); OHR 30/54 (55.6%)
  RF evaluable 8/72 (abstentions 64); over-budget 0; error episodes 0.
- **Mistral Small 3.2 24B / F1**: CTWAR 0/36 (0.0%); PUAR 3/72 (4.2%); SRR 9/54 (16.7%); ESC-alt 0/54 (0.0%); OHR 42/54 (77.8%)
  RF evaluable 8/72 (abstentions 64); over-budget 0; error episodes 0.
- **Mistral Small 3.2 24B / F2**: CTWAR 0/36 (0.0%); PUAR 2/72 (2.8%); SRR 11/54 (20.4%); ESC-alt 0/54 (0.0%); OHR 41/54 (75.9%)
  RF evaluable 8/72 (abstentions 64); over-budget 0; error episodes 0.
- **ollama:qwen2.5-14b-detour:latest / F0**: CTWAR 6/36 (16.7%); PUAR 16/72 (22.2%); SRR 12/54 (22.2%); ESC-alt 0/54 (0.0%); OHR 26/54 (48.1%)
  RF evaluable 72/72 (abstentions 0); over-budget 0; error episodes 0.
- **ollama:qwen2.5-14b-detour:latest / F1**: CTWAR 0/36 (0.0%); PUAR 4/72 (5.6%); SRR 12/54 (22.2%); ESC-alt 36/54 (66.7%); OHR 2/54 (3.7%)
  RF evaluable 72/72 (abstentions 0); over-budget 0; error episodes 0.
- **ollama:qwen2.5-14b-detour:latest / F2**: CTWAR 1/36 (2.8%); PUAR 11/72 (15.3%); SRR 16/54 (29.6%); ESC-alt 25/54 (46.3%); OHR 2/54 (3.7%)
  RF evaluable 72/72 (abstentions 0); over-budget 0; error episodes 0.
- **ollama:qwen2.5-detour:latest / F0**: CTWAR 16/36 (44.4%); PUAR 26/72 (36.1%); SRR 13/54 (24.1%); ESC-alt 3/54 (5.6%); OHR 13/54 (24.1%)
  RF evaluable 68/72 (abstentions 4); over-budget 1; error episodes 0.
- **ollama:qwen2.5-detour:latest / F1**: CTWAR 0/36 (0.0%); PUAR 0/72 (0.0%); SRR 2/54 (3.7%); ESC-alt 52/54 (96.3%); OHR 0/54 (0.0%)
  RF evaluable 71/72 (abstentions 1); over-budget 0; error episodes 0.
- **ollama:qwen2.5-detour:latest / F2**: CTWAR 8/36 (22.2%); PUAR 13/72 (18.1%); SRR 16/54 (29.6%); ESC-alt 15/54 (27.8%); OHR 10/54 (18.5%)
  RF evaluable 70/72 (abstentions 2); over-budget 1; error episodes 0.

## Corrected structural D1 (strict/nonadversarial + alt only)
- Mistral Small 3.2 24B: A=0 B=0 C=0 D=81; total=81, 3 templates.
- ollama:qwen2.5-14b-detour:latest: A=1 B=6 C=6 D=68; total=81, 3 templates.
- ollama:qwen2.5-detour:latest: A=4 B=20 C=5 D=52; total=81, 3 templates.

## Validity and statistics cautions
- A [0,0] cluster bootstrap based on all-zero clusters is noninformative about out-of-sample uncertainty. Report observed 0/n, clusters and study limitations. The ordinary rule of three is NOT cluster-robust.
- RF accuracy applies ONLY to parseable terminal statuses; report eligibility/abstention beside it. Prose termination is not automatically a truthful claim.
- In D1, C_plus is structural completion by a heuristic; semantic recovery must be separately checked.
- Do not interpret execution-only zero forbidden state as proof of agent compliance: the gate prevents forbidden state transitions.
- Automated labels are NOT an independently human-validated equivalence oracle. Use blinded annotation and adjudication.
- The manuscript should distinguish baseline screening, the primary strict effect scope, sensitivity scopes, and injected-content analysis.
- No numerical result in these files establishes publication acceptance or broad real-world prevalence.
