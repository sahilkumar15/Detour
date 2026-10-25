"""Where am I, and what do I run next?

    python status.py

Checks the whole setup and prints the single next command. Nothing has to
be running first: every probe has a short timeout and reports plainly
instead of failing.

Sections:
  1. code health      -- self-test, unit tests, construct validity, digest
  2. local models     -- is the Ollama server up, which models are usable
  3. provider budget  -- Groq daily tokens left, and whether a run fits
  4. results on disk  -- which runs exist, under which benchmark
  5. next command     -- the one thing to do now
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable
TOKENS_PER_EPISODE = 4260  # measured: ~1065 tokens x ~4 calls


def run(args: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [PY, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 1, "(timed out)"
    except Exception as exc:  # pragma: no cover
        return 1, f"({type(exc).__name__}: {exc})"


def head(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------


def code_health() -> bool:
    head("1. CODE HEALTH  (no server, no key, no network)")
    ok = True

    code, out = run(["run_audit.py", "--self-test"])
    line = next((l for l in out.splitlines() if "checks passed" in l), "no result")
    print(f"  self-test         {line.strip()}")
    ok &= code == 0

    code, out = run(["-m", "pytest", "tests", "-q"])
    line = next((l for l in out.splitlines() if "passed" in l or "failed" in l), "no result")
    print(f"  unit tests        {line.strip()}")
    ok &= code == 0

    code, out = run(["validate_metrics.py"])
    suite = next((l for l in out.splitlines() if l.startswith("suite:")), "")
    verdict = next((l for l in out.splitlines() if "PASSED" in l or "FAILED" in l), "no result")
    print(f"  construct validity {verdict.strip()}")
    print(f"  {suite.strip()}")
    ok &= code == 0

    code, out = run(["run_audit.py", "--freeze-digest"])
    try:
        manifest = json.loads(out)
        dataset = manifest["dataset"]
        print(
            f"  benchmark         dataset={dataset['digest']}  "
            f"{dataset['n_tasks']} tasks / {dataset['n_templates']} templates / "
            f"{dataset['n_cross_tool_clusters']} cross-tool clusters"
        )
    except Exception:
        print("  benchmark         could not read digest")
        ok = False

    print("\n  " + ("ALL GREEN -- the measurement is sound." if ok
                    else "SOMETHING IS BROKEN above. Fix that before running any model."))
    return ok


def local_models() -> list[str]:
    head("2. LOCAL MODELS  (Ollama -- no quota, no key)")
    try:
        from openai import OpenAI

        client = OpenAI(api_key="local", base_url="http://localhost:11434/v1", timeout=5.0)
        names = sorted(m.id for m in client.models.list())
    except Exception:
        print("  Ollama server NOT reachable at localhost:11434")
        print("  -> launch Ollama from the Start menu (llama icon in the tray).")
        print("     The server is separate from the `ollama` command; you do NOT")
        print("     need `ollama` on PATH for the harness to work.")
        return []

    print(f"  Ollama server UP. {len(names)} model(s):")
    for name in names:
        note = ""
        if "detour" in name:
            note = "   <- your num_ctx 8192 build, use this one"
        elif ":cloud" in name:
            note = "   (cloud model, needs sign-in -- not local)"
        print(f"      {name}{note}")
    if not any("detour" in n for n in names):
        print("\n  No *-detour model yet. Build one so the context window is big")
        print("  enough (a small context silently drops the denial):")
        print("      ollama create qwen2.5-detour -f Modelfile")
    return names


def provider_budget() -> None:
    head("3. PROVIDER BUDGET  (Groq)")
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("  GROQ_API_KEY not set; skipping.")
        return
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=20.0)
        raw = client.chat.completions.with_raw_response.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        headers = {k.lower(): v for k, v in dict(raw.headers).items() if "ratelimit" in k.lower()}
    except Exception as exc:
        text = str(exc)
        if "tokens per day" in text or "TPD" in text:
            print("  DAILY TOKEN BUDGET EXHAUSTED. Nothing will run on Groq until it resets.")
            print("  -> use the local Ollama model tonight (section 2).")
        else:
            print(f"  probe failed: {type(exc).__name__}: {text[:160]}")
        return

    for key_name in sorted(headers):
        print(f"      {key_name:<40} {headers[key_name]}")
    for name, value in headers.items():
        if "remaining" in name and "token" in name and "day" in name:
            try:
                remaining = float(str(value).rstrip("s"))
            except ValueError:
                continue
            episodes = int(remaining // TOKENS_PER_EPISODE)
            print(f"\n  ~{remaining:,.0f} tokens left today -> about {episodes} episodes")
            print(f"  (one condition of the 24-task suite needs 24 episodes, "
                  f"~{24 * TOKENS_PER_EPISODE:,} tokens)")
            if episodes < 24:
                print("  NOT enough for one condition. Use the local model instead.")


def results_on_disk() -> None:
    head("4. RESULTS ON DISK")
    results = ROOT / "results"
    if not results.exists():
        print("  no results/ directory yet.")
        return
    rows = sorted(results.glob("*.jsonl"))
    if not rows:
        print("  results/ exists but holds no .jsonl runs.")
        return
    for path in rows:
        try:
            n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception:
            n = -1
        manifest = path.with_suffix(".manifest.json")
        digest = "no manifest (predates the freeze -- not comparable)"
        if manifest.exists():
            try:
                digest = json.loads(manifest.read_text())["dataset"]["digest"]
            except Exception:
                digest = "unreadable manifest"
        print(f"  {path.name:<30} {n:>4} episodes   dataset={digest}")
    print("\n  Only runs sharing ONE dataset digest may be compared.")


def next_command(code_ok: bool, models: list[str]) -> None:
    head("5. WHAT TO RUN NEXT")
    if not code_ok:
        print("  Fix the failures in section 1 first. Nothing else is meaningful.")
        return

    detour = next((m for m in models if "detour" in m), None)
    if detour:
        clean = ROOT / "results" / "clean-local.jsonl"
        if not clean.exists():
            print("  Competence first -- a model that cannot chain tools scores")
            print("  CTWAR 0% and looks perfectly safe. Run:\n")
            print(f"    python run_audit.py --clean --agents ollama:{detour} \\")
            print("        --reps 2 --max-tokens 900 --out results/clean-local\n")
            print("  Time it: 12 episodes tells you whether 72 is minutes or hours.")
        else:
            print("  Clean arm exists. If it showed the model chaining tools, run the grid:\n")
            print(f"    python run_audit.py --agents ollama:{detour} \\")
            print("        --conditions F0,F1,F2 --reps 1 --max-tokens 900 \\")
            print("        --out results/grid-local\n")
            print("  No quota locally, so keep all three conditions.")
        return

    if models:
        print("  Build the big-context model first (a small context silently")
        print("  drops the denial from the transcript):\n")
        print("    ollama create qwen2.5-detour -f Modelfile\n")
        print("  Modelfile contents:")
        print("    FROM qwen2.5:7b")
        print("    PARAMETER num_ctx 8192")
        return

    print("  Start the Ollama app (tray icon), then run this script again.")
    print("  Meanwhile the pipeline can be exercised with no model at all:\n")
    print("    python mock_server.py                     # terminal 1")
    print("    python run_audit.py --agents local:mock-not-a-model \\")
    print("        --conditions F0,F1,F2 --out results/mock    # terminal 2")


def main() -> int:
    print("Detour -- setup status")
    print(f"project: {ROOT}")
    ok = code_health()
    models = local_models()
    provider_budget()
    results_on_disk()
    next_command(ok, models)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())