
import json
from pathlib import Path

files = [
    Path("results/grid-v8-qwen7.jsonl"),
    Path("results/grid-v8-qwen14.jsonl"),
]

total_unauthorized = 0
total_denied_steps = 0

for path in files:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    unauthorized = [r for r in rows if r.get("U")]

    denied_steps = [
        (r, s)
        for r in rows
        for s in r.get("steps", [])
        if s.get("allowed") is False
    ]

    missing = [
        r for r in unauthorized
        if not any(
            s.get("allowed") is False
            for s in r.get("steps", [])
        )
    ]

    missing_workarounds = [
        r for r in rows
        if r.get("W") and not any(
            s.get("label") == "cross_tool_workaround"
            for s in r.get("steps", [])
        )
    ]

    print(f"\n{path.name}")
    print("Episodes:", len(rows))
    print("U-positive episodes:", len(unauthorized))
    print("Denied steps:", len(denied_steps))
    print("U episodes missing denied steps:", len(missing))
    print("W episodes missing CTWAR steps:", len(missing_workarounds))

    for r in missing[:10]:
        print("MISSING U:", r["task_id"], r["feedback"], r["repetition"])

    total_unauthorized += len(unauthorized)
    total_denied_steps += len(denied_steps)

print("\nCOMBINED")
print("U-positive episodes:", total_unauthorized)
print("Denied steps:", total_denied_steps)
print("Expected queue cases:", total_denied_steps)
