# Detour Four-AI-Judge Sensitivity

NOT independent human annotation.
NOT accuracy against established human ground truth.
259 non-pilot selected action proposals.

## Complete four-model results

| Model | Eq yes | Eq agreement | Oracle-positive overlap | Eq kappa | Forbidden agreement | Forbidden kappa |
|---|---:|---:|---:|---:|---:|---:|
| GPT-OSS 20B (A) | 57/259 | 81.8% | 55.9% | 0.654 | 88.4% | 0.836 |
| Qwen3.5 35B-A3B (D) | 41/259 | 75.3% | 38.7% | 0.527 | 79.5% | 0.599 |
| Mistral 24B (B) | 11/259 | 64.5% | 6.5% | 0.045 | 88.8% | 0.753 |
| Llama 3.3 70B (C) | 6/259 | 66.4% | 6.5% | 0.081 | 71.4% | 0.298 |

## Exploratory three-model shortlist
A: GPT-OSS 20B — strongest oracle alignment.
D: Qwen3.5 35B — second equivalence candidate.
B: Mistral 24B — contrasting forbiddenness judge.

This selection is post-hoc, not a human quality ranking.
Llama C remains in the complete four-model analysis.

## Private discrepancy priorities
{'P0': 15, 'P1': 122, 'P2': 11, 'P3': 111}

Do not treat AI consensus as human ground truth.
