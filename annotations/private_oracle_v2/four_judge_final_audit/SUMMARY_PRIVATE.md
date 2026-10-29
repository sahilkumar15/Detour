# Detour AI Judge Validation (NOT human annotation)

259 non-pilot action proposals; four previously discussed calibration cases excluded.

## Model-label distributions
- Reviewer A: equiv {'yes': 57, 'no': 190, 'uncertain': 12}; forbidden {'yes': 136, 'no': 113, 'uncertain': 10}; actual models {'gptoss20-detour:latest': 259}
- Reviewer B: equiv {'no': 246, 'yes': 11, 'uncertain': 2}; forbidden {'yes': 181, 'no': 78}; actual models {'mistral24-detour:latest': 259}
- Reviewer C: equiv {'no': 253, 'yes': 6}; forbidden {'yes': 233, 'no': 25, 'uncertain': 1}; actual models {'llama3.3:70b': 259}
- Reviewer D: equiv {'uncertain': 22, 'no': 196, 'yes': 41}; forbidden {'yes': 200, 'no': 46, 'uncertain': 13}; actual models {'qwen3.5:35b': 259}

## Agreement with machine oracle (machine is NOT ground truth)
| Judge | Equivalence all-case agreement | Equivalence binary kappa | Oracle-positive equivalence agreement | Forbiddenness all-case agreement | Forbiddenness binary kappa |
|---|---:|---:|---:|---:|---:|
| A | 81.85% | 0.6539923954372623 | 55.91% | 88.42% | 0.8362596172815151 |
| B | 64.48% | 0.04468858417313724 | 6.45% | 88.8% | 0.7529195039310503 |
| C | 66.41% | 0.08122324159021412 | 6.45% | 71.43% | 0.2981292390251175 |
| D | 75.29% | 0.5267211867630279 | 38.71% | 79.54% | 0.5993485342019543 |

## Pairwise AI judge comparison
| Pair | Equivalence exact agreement | Equivalence 3-label kappa | Forbidden exact agreement | Forbidden 3-label kappa |
|---|---:|---:|---:|---:|
| AB | 72.2% | 0.05292026409344862 | 81.08% | 0.622864105078601 |
| AC | 75.68% | 0.12598425196850396 | 62.16% | 0.22040665888568092 |
| AD | 73.36% | 0.34394273127753305 | 70.27% | 0.4228286979422916 |
| BC | 93.44% | 0.07829181494661913 | 78.76% | 0.37951912187472775 |
| BD | 73.36% | 0.027163854109961826 | 81.08% | 0.5350089766606824 |
| CD | 77.22% | 0.11399083898649062 | 81.08% | 0.3430479345687961 |

## Private discrepancy triage
{'P0': 15, 'P1': 122, 'P2': 11, 'P3': 111}

Machine-label disagreements are NOT adjudicated model errors.
The 259 selected proposals do not estimate global accuracy on 864 episodes.
A consensus of AI models is not independent ground truth.
Provide real blinded human A/B labels, then independent adjudication.
Do NOT release private reports to blinded human annotators.
