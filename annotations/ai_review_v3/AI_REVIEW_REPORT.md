# Detour AI judge analysis (NOT human review)

All source cases: 263; four discussed pilot cases excluded from agreement.

Reviewer A: 263/263 valid, 0 unresolved; models {'gptoss20-detour:latest': 263}; fallback 0
Reviewer B: 263/263 valid, 0 unresolved; models {'mistral24-detour:latest': 263}; fallback 0

Paired nonpilot valid cases: 259
Cases with AI-AI disagreements: 226
equivalent: exact agreement=0.722007722007722; Cohen kappa (yes/no only)=0.03267528931245756 on 245 pairs
forbidden: exact agreement=0.8108108108108109; Cohen kappa (yes/no only)=0.6755537736794629 on 249 pairs

This tests AI-generated annotation sensitivity, not correctness against human ground truth.
Overall equivalence is derived from four dimensions.
Check policy reasoning and grounding manually.
Human-label files, machine oracle and experiment traces are unchanged.
