"""Preflight a local (or any OpenAI-compatible) model before spending a grid on it.

    python check_local.py ollama:qwen2.5:7b
    python check_local.py local:mock-not-a-model
    python check_local.py groq:openai/gpt-oss-120b

Four checks, in the order that fails fastest:

  1. server reachable
  2. model present on that server
  3. emits a TOOL CALL when one is required
  4. answers a tool RESULT with a second tool call (multi-turn protocol)

Check 3 is the one that matters. A model that cannot emit tool calls will
produce CTWAR 0% -- a perfect safety score -- while never having been
capable of the behaviour the metric measures. That is the single most
dangerous failure mode in this study, so it is checked explicitly, before
any episode runs, rather than being inferred from a suspiciously clean
result afterwards.

Check 4 catches the subtler case: some models emit a first tool call and
then fall back to prose once a `role: "tool"` message appears. Detour needs
up to six chained steps, so a model that cannot continue after a tool
result is unusable even though check 3 passed.

Passing this is necessary, NOT sufficient. The clean-task arm
(`run_audit.py --clean`) is still what decides whether a model is competent
enough for its safety rates to mean anything.
"""

from __future__ import annotations

from typing import Any

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_audit import PROVIDERS  # noqa: E402

PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_city",
            "description": "Look up a city record.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _client(spec: str):
    kind, _, model = spec.partition(":")
    if kind not in PROVIDERS:
        raise SystemExit(f"unknown provider {kind!r}; choose from {sorted(PROVIDERS)}")
    base_url, key_env = PROVIDERS[kind]
    candidates = [v.strip() for v in key_env.split(",") if v.strip()]
    key = next((os.environ[n] for n in candidates if os.environ.get(n, "").strip()), "")
    if not key and candidates:
        raise SystemExit(f"{' or '.join(candidates)} is not set in .env")
    from openai import OpenAI

    return OpenAI(api_key=key or "local", base_url=base_url, timeout=300.0), model, base_url


def _report_budget(client: Any, model: str) -> None:
    """Print every rate-limit header the provider returns.

    Header names differ between providers and change over time, so every
    `x-ratelimit-*` key is printed rather than a guessed subset. Reading the
    wrong name would be worse than reading none: it would report a budget
    that is not the binding one.
    """
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception as exc:
        print(f"  note  budget probe failed ({type(exc).__name__}); skipping")
        return

    headers = {
        k.lower(): v for k, v in dict(raw.headers).items() if "ratelimit" in k.lower()
    }
    if not headers:
        print("  note  provider returned no rate-limit headers")
        return

    print("\n  provider budget:")
    for key in sorted(headers):
        print(f"      {key:<42} {headers[key]}")

    # The daily token allowance is what decides whether a grid fits today.
    for key, value in headers.items():
        if "remaining" in key and "token" in key and "day" in key:
            try:
                remaining = float(str(value).rstrip("s"))
            except ValueError:
                continue
            episodes = int(remaining // 4300)  # ~4.3k tokens per episode
            print(
                f"\n  ~{remaining:,.0f} tokens left today "
                f"-> roughly {episodes} episodes "
                f"({episodes // 24} full conditions of a 24-task suite)"
            )
            if episodes < 24:
                print("  NOT enough for one condition. Wait for the daily reset.")
    print()


def main(spec: str) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    client, model, base_url = _client(spec)
    print(f"probing {model!r} at {base_url}\n")
    failures = 0

    # 1 + 2 -------------------------------------------------------------
    try:
        available = sorted(m.id for m in client.models.list())
    except Exception as exc:
        print(f"  FAIL  server unreachable: {type(exc).__name__}: {exc}")
        if "localhost" in base_url:
            print("        Is the local server running? For Ollama, check that")
            print("        the app is started, then run:  ollama list")
        return 1
    print(f"  PASS  server reachable ({len(available)} models)")

    if model in available:
        print(f"  PASS  model present: {model}")
    else:
        print(f"  FAIL  model {model!r} not on this server.")
        shown = ", ".join(available[:12]) or "(none)"
        print(f"        available: {shown}")
        print("        For Ollama:  ollama pull <model>")
        return 1

    # 2.5 budget ---------------------------------------------------------
    # A provider's remaining DAILY allowance is the single most useful number
    # before starting a grid, and it is only visible in response headers. A
    # ~20-token probe buys it, which is a great deal against discovering the
    # limit after twenty minutes of flow-control waiting.
    _report_budget(client, model)

    # 3 -----------------------------------------------------------------
    messages = [
        {"role": "user", "content": "Look up the record for Paris using the tool."}
    ]
    call = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=PROBE_TOOLS,
            tool_choice="required",
            temperature=0.0,
        )
        call = (response.choices[0].message.tool_calls or [None])[0]
    except Exception as exc:
        # tool_choice="required" is not universally supported; retry without
        # it before concluding the model cannot call tools at all.
        print(f"  note  tool_choice=required rejected ({type(exc).__name__}); retrying without it")
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=PROBE_TOOLS, temperature=0.0
            )
            call = (response.choices[0].message.tool_calls or [None])[0]
        except Exception as inner:
            print(f"  FAIL  tool-calling request errored: {type(inner).__name__}: {inner}")
            return 1

    if call is None:
        print("  FAIL  model answered in prose instead of calling a tool.")
        print("        It CANNOT be used as an agent here. Used anyway it would")
        print("        score CTWAR 0% -- a perfect safety score it never earned.")
        print("        Pick a model tagged for tools (Ollama: ollama.com, 'Tools' filter).")
        return 1
    print(f"  PASS  emits a tool call: {call.function.name}({call.function.arguments})")

    # 4 -----------------------------------------------------------------
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": "lookup_city completed. Result:\nParis, FR. population 2.1M.",
        }
    )
    messages.append({"role": "user", "content": "Now look up Berlin with the tool."})
    try:
        follow = client.chat.completions.create(
            model=model, messages=messages, tools=PROBE_TOOLS, temperature=0.0
        )
        second = (follow.choices[0].message.tool_calls or [None])[0]
    except Exception as exc:
        print(f"  FAIL  multi-turn tool protocol errored: {type(exc).__name__}: {exc}")
        return 1

    if second is None:
        print("  FAIL  stopped calling tools after a tool result arrived.")
        print("        Detour chains up to 6 steps; this model cannot continue.")
        failures += 1
    else:
        print(f"  PASS  continues after a tool result: {second.function.name}")

    usage = getattr(follow, "usage", None)
    if usage:
        print(f"\n  tokens on a 2-turn probe: {getattr(usage, 'total_tokens', '?')}")

    print()
    if failures:
        print("NOT USABLE as an agent in this harness.")
        return 1
    print("Tool calling works. Next, the check that actually gates the paper:")
    print(f"  python run_audit.py --clean --agents {spec} --reps 1 --out results/clean-probe")
    print("A model that cannot chain two tools on a BENIGN task must not")
    print("appear in the grid, however clean its safety rates look.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python check_local.py <provider>:<model>")
    raise SystemExit(main(sys.argv[1]))