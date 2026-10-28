# Detour repository layout
- Model definitions are under `configs/models/`.
- Paper data remain in `results/grid-v7-*` and `results/clean-v7-*`.
- Older experiments are preserved in `results/archive/`; NEVER mix their fingerprints.
- Utilities are under `scripts/`. Research modules/metrics are deliberately unchanged.
- Example: `ollama create qwen2.5-detour -f configs/models/Modelfile`.
- Example: `python scripts/check_v7_evidence.py results/grid-v7-qwen7.jsonl results/grid-v7-qwen14.jsonl`.
- Never modify v7 logs in place. New metric/scenario changes require a new frozen run name.
