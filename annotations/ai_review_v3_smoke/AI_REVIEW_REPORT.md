# Detour AI judge analysis (NOT human review)

All source cases: 263; four discussed pilot cases excluded from agreement.

Reviewer A: 5/263 valid, 258 unresolved; models {'gptoss20-detour:latest': 5}; fallback 0
Reviewer B: 5/263 valid, 258 unresolved; models {'mistral24-detour:latest': 5}; fallback 0

Paired nonpilot valid cases: 1
Cases with AI-AI disagreements: 1
equivalent: exact agreement=0.0; Cohen kappa (yes/no only)=0.0 on 1 pairs
forbidden: exact agreement=1.0; Cohen kappa (yes/no only)=None on 1 pairs

This tests AI-generated annotation sensitivity, not correctness against human ground truth.
Overall equivalence is derived from four dimensions.
Check policy reasoning and grounding manually.
Human-label files, machine oracle and experiment traces are unchanged.
