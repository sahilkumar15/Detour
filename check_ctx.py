import json
import re
import sys
import urllib.request

models = sys.argv[1:]

for model in models:
    payload = json.dumps({"model": model}).encode()

    request = urllib.request.Request(
        "http://localhost:11434/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.load(response)

    params = info.get("parameters", "")
    match = re.search(r"(?m)^\s*num_ctx\s+(\d+)", params)

    ctx = int(match.group(1)) if match else None
    tools = "tools" in info.get("capabilities", [])

    print(f"\nModel: {model}")
    print(f"Configured context: {ctx}")
    print(f"Tool capability: {tools}")

    if ctx != 8192 or not tools:
        raise SystemExit(f"Configuration check failed: {model}")

print("\nAll model configurations passed.")
