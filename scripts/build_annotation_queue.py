"""Build the blinded queue for independent effect-equivalence labelling.

CTWAR counts an episode when the oracle says a later proposal realizes the
denied effect. That is the claim a reviewer will not take on trust, so it
needs human labels produced WITHOUT sight of the oracle's answer.

The blinded sheet therefore carries only what a human needs to judge
equivalence for themselves: the denied call, the later call, the policy rule
in force, and the protected object's markers. It deliberately omits
`denied_effects`, the step's `effects`, the step label and the episode's W --
those are the machine's own conclusions, and an annotator shown two
identical effect tuples has been told the answer rather than asked the
question. The private key carries them for scoring afterwards.

Usage:
    python scripts/build_annotation_queue.py results/grid-v8-*.jsonl
    python scripts/build_annotation_queue.py results/grid-v8-qwen7.jsonl --sample 120
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from pda.subsets import load_all

# Only a tool CHANGE can be a cross-tool workaround, so a same-tool
# re-attempt is not an equivalence question and does not belong in the queue.
BLIND_FIELDS = [
    "case_id",
    "task",
    "policy_rule",
    "protected_markers",
    "denied_call",
    "later_call",
    "human_same_effect",      # yes / no / unsure
    "human_still_forbidden",  # yes / no / unsure
    "human_notes",
]
KEY_FIELDS = [
    "case_id",
    "source",
    "model",
    "feedback",
    "task_id",
    "repetition",
    "template",
    "automatic_step_label",
    "automatic_denied_effects",
    "automatic_proposed_effects",
    "episode_W",
]


def call_text(tool: str, args: dict) -> str:
    return f"{tool}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"


def scenario_context() -> dict[str, dict]:
    """task_id -> the goal, denied call and markers the annotator needs.

    A trace row records `denied_tool` and `denied_effects` but neither the
    goal nor `denied_args`, so these have to be joined back from the suite.
    Without them the sheet renders the denied call as `send_email({})` and
    names the task by id, which is not enough to judge equivalence.
    """
    from pda.scenarios import build_suite

    out: dict[str, dict] = {}
    for spec in build_suite():
        out[spec.task_id] = {
            "goal": spec.goal,
            "denied_tool": spec.denied_tool,
            "denied_args": dict(spec.denied_args),
            "markers": "; ".join(
                f"{obj}: {', '.join(ms)}" for obj, ms in spec.ctx.object_markers.items()
            ),
        }
    return out


def build(rows: list[dict], sample: int | None, seed: int) -> tuple[list[dict], list[dict]]:
    blind: list[dict] = []
    key: list[dict] = []
    context = scenario_context()
    missing: set[str] = set()
    for r in rows:
        rule = "; ".join(
            f"{x}" for x in (r.get("denied_effects") or [])
        )
        spec = context.get(r.get("task_id", ""))
        if spec is None:
            missing.add(r.get("task_id", "?"))
            spec = {"goal": "", "denied_tool": r.get("denied_tool", ""),
                    "denied_args": {}, "markers": ""}
        for step in r.get("steps", []):
            if step.get("allowed") is not False:
                continue
            if step.get("tool") == r.get("denied_tool"):
                continue
            idx = len(blind) + 1
            cid = f"CTW-{idx:05d}"
            blind.append({
                "case_id": cid,
                "task": spec["goal"],
                "policy_rule": step.get("rule_id") or "",
                "protected_markers": spec["markers"],
                "denied_call": call_text(spec["denied_tool"], spec["denied_args"]),
                "later_call": call_text(step.get("tool", ""), dict(step.get("args") or {})),
                "human_same_effect": "",
                "human_still_forbidden": "",
                "human_notes": "",
            })
            key.append({
                "case_id": cid,
                "source": r.get("_source", ""),
                "model": r.get("model"),
                "feedback": r.get("feedback"),
                "task_id": r.get("task_id"),
                "repetition": r.get("repetition"),
                "template": r.get("template"),
                "automatic_step_label": step.get("label"),
                "automatic_denied_effects": rule,
                "automatic_proposed_effects": "; ".join(step.get("effects") or []),
                "episode_W": r.get("W"),
            })

    if missing:
        print(f"  WARNING: {len(missing)} task_id(s) not in the current suite "
              f"(e.g. {sorted(missing)[0]}); their goal and denied arguments are "
              f"blank. These traces predate the current dataset digest.")

    rng = random.Random(seed)
    if sample and sample < len(blind):
        picked = sorted(rng.sample(range(len(blind)), sample))
        blind = [blind[i] for i in picked]
        chosen = {b["case_id"] for b in blind}
        key = [k for k in key if k["case_id"] in chosen]

    # Shuffle the blinded sheet only. Presentation order otherwise tracks
    # model and condition, which is itself a hint about the answer.
    order = list(range(len(blind)))
    rng.shuffle(order)
    blind = [blind[i] for i in order]
    return blind, key


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path}  {len(rows)} cases")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("annotations/v8"))
    ap.add_argument("--sample", type=int, default=None,
                    help="draw this many cases instead of every one")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.files:
        for row in load_all([path]):
            row["_source"] = str(path)
            rows.append(row)

    blind, key = build(rows, args.sample, args.seed)
    if not blind:
        raise SystemExit("no tool-changing denied steps found; nothing to annotate")

    write(args.out / "review_blinded.csv", blind, BLIND_FIELDS)
    write(args.out / "answer_key_private.csv", key, KEY_FIELDS)
    print("\n  The blinded sheet carries no machine effect labels, no step label and")
    print("  no episode W. Keep answer_key_private.csv away from annotators, have at")
    print("  least two label independently, and report agreement plus an adjudication")
    print("  ledger. This script builds the queue; it does not validate equivalence.")


if __name__ == "__main__":
    main()
