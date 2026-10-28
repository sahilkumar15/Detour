#!/usr/bin/env python3
"""Detour single-writer, quota-aware Groq experiment launcher.

Preserves the old contaminated run. No modifications to frozen
scenario, gate, effect, metric, or sandbox implementations.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "results/grid-v8-gptoss20-groq-cleanroom"

REF = ROOT / "results/grid-v8-qwen7.manifest.json"

MODEL = "groq:openai/gpt-oss-20b"

CONDITIONS = ("F0", "F1", "F2")

TOTAL = 216

ARGS = (
    "--agents", "oss20",
    "--conditions", "F0,F1,F2",
    "--reps", "3",
    "--max-steps", "10",
    "--max-tokens", "900",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_snapshot(frozen: dict) -> dict:
    packages = {}

    for name in ("openai", "httpx", "Brotli"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"

    return {
        "model": MODEL,
        "conditions": list(CONDITIONS),
        "reps": 3,
        "max_steps": 10,
        "max_tokens": 900,
        "runner_args": list(ARGS),
        "dataset_digest": frozen["dataset"]["digest"],
        "metrics_digest": frozen["metrics"]["digest"],
        "code_digest": frozen["code"]["digest"],
        "files_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "run_audit.py",
                "pda/agents.py",
                "models.json",
            )
        },
        "python": sys.version.split()[0],
        "packages": packages,
    }


def record_provenance(snapshot: dict) -> None:
    path = OUT.with_suffix(".provenance.json")

    if path.exists():
        old = json.loads(
            path.read_text(encoding="utf-8")
        )

        if old != snapshot:
            raise RuntimeError(
                "Code or environment changed since this run began. "
                "Do not combine different inference configurations."
            )
        return

    if OUT.with_suffix(".jsonl").exists():
        raise RuntimeError(
            "Clean-room JSONL exists without provenance. "
            "Refusing an unverified resume."
        )

    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Provenance recorded:", path, flush=True)


def existing_groq_processes() -> list[int]:
    result = []

    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue

        if int(proc.name) == os.getpid():
            continue

        try:
            args = [
                os.fsdecode(x)
                for x in
                (proc / "cmdline").read_bytes().split(b"\0")
                if x
            ]

            if not any(
                Path(x).name == "run_audit.py"
                for x in args
            ):
                continue

            if "--agents" not in args:
                continue

            agent = args[args.index("--agents") + 1]

            if agent in ("oss20", MODEL):
                result.append(int(proc.name))

        except (OSError, IndexError, ValueError):
            continue

    return sorted(result)


def check_results(suite):
    from pda.freeze import verify_manifest

    expected = {
        (MODEL, feedback, rep, spec.task_id)
        for feedback in CONDITIONS
        for rep in range(3)
        for spec in suite
    }

    assert len(expected) == TOTAL, (
        "Unexpected scenario/task identities"
    )

    data = OUT.with_suffix(".jsonl")

    seen = set()
    counts = Counter()

    if data.exists():
        for line_no, line in enumerate(
            data.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)

                key = (
                    row["model"],
                    row["feedback"],
                    row["repetition"],
                    row["task_id"],
                )

            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeError(
                    f"Invalid row {line_no}: {exc}"
                ) from exc

            if key not in expected:
                raise RuntimeError(
                    f"Unexpected episode row {line_no}: {key}"
                )

            if key in seen:
                raise RuntimeError(
                    f"DUPLICATE episode row {line_no}: {key}"
                )

            if row.get("error"):
                raise RuntimeError(
                    f"Failed episode at row {line_no}"
                )

            seen.add(key)
            counts[row["feedback"]] += 1

    manifest_path = OUT.with_suffix(".manifest.json")
    complete = False

    if manifest_path.exists():
        verify_manifest(
            manifest_path,
            suite,
            strict_code=True,
        )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        info = manifest.get("run", {})

        expected_run = {
            "agents": [MODEL],
            "conditions": list(CONDITIONS),
            "reps": 3,
            "max_steps": 10,
            "max_tokens": 900,
        }

        for field, value in expected_run.items():
            if info.get(field) != value:
                raise RuntimeError(
                    f"Run configuration mismatch: {field}"
                )

        complete = info.get("complete") is True

        if complete and (
            len(seen) != TOTAL
            or info.get("n_episodes") != TOTAL
        ):
            raise RuntimeError(
                "Manifest says complete, but coverage disagrees."
            )

    print(
        f"Validated {len(seen)}/{TOTAL}: "
        + ", ".join(
            f"{f}={counts[f]}/72"
            for f in CONDITIONS
        )
        + f"; manifest complete={complete}",
        flush=True,
    )

    return len(seen), complete


def parse_wait(message: str):
    match = re.search(
        r"please try again in\s+"
        r"(?:(\d+(?:\.\d+)?)h)?"
        r"(?:(\d+(?:\.\d+)?)m)?"
        r"(?:(\d+(?:\.\d+)?)s)?",
        message,
        re.IGNORECASE,
    )

    if not match or not any(match.groups()):
        return None

    hours, minutes, seconds = (
        float(v or 0)
        for v in match.groups()
    )

    return hours * 3600 + minutes * 60 + seconds


def error_class(message: str):
    text = message.lower()

    is_limit = (
        "ratelimiterror" in text
        or ("429" in text and "rate limit" in text)
    )

    if is_limit:
        if any(
            x in text
            for x in (
                "tokens per day",
                "requests per day",
                "(tpd)",
                "(rpd)",
            )
        ):
            return "DAILY_QUOTA"

        if any(
            x in text
            for x in (
                "tokens per minute",
                "requests per minute",
                "(tpm)",
                "(rpm)",
            )
        ):
            return "MINUTE_LIMIT"

    if any(
        x in text
        for x in (
            "internalservererror",
            "apiconnectionerror",
            "apitimeouterror",
        )
    ):
        return "TRANSIENT"

    return None


def execute(round_id: int):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log = (
        ROOT / "logs"
        / f"groq-cleanroom-{stamp}-{round_id}.log"
    )

    command = [
        sys.executable,
        "-u",
        "run_audit.py",
        *ARGS,
        "--verify-freeze", str(REF),
        "--out", str(OUT),
    ]

    print(
        f"\n=== Groq attempt {round_id} ===",
        flush=True,
    )

    print("Log:", log, flush=True)

    tail = deque(maxlen=120)

    with log.open("w", encoding="utf-8") as file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in process.stdout:
                print(line, end="", flush=True)

                file.write(line)
                file.flush()

                tail.append(line)

        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

        return process.wait(), "".join(tail)


def cooldown(seconds: float):
    remaining = max(1.0, seconds)

    while remaining > 0:
        print(
            f"Groq cooldown: {remaining / 60:.1f} "
            "minutes; no requests sent.",
            flush=True,
        )

        wait = min(60.0, remaining)
        time.sleep(wait)
        remaining -= wait


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Validate saved results without calling Groq",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test quota parser without calling Groq",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1000,
    )

    options = parser.parse_args()

    if options.self_test:
        assert abs(
            parse_wait("Please try again in 6m24.48s")
            - 384.48
        ) < 1e-8

        assert parse_wait(
            "Please try again in 1h2m3s"
        ) == 3723

        assert error_class(
            "RateLimitError: 429 tokens per day (TPD)"
        ) == "DAILY_QUOTA"

        assert error_class(
            "RateLimitError: 429 tokens per minute (TPM)"
        ) == "MINUTE_LIMIT"

        print("PASS: Cooldown and quota parser")
        return 0

    if options.max_rounds < 1:
        parser.error("--max-rounds must be positive")

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    sys.path.insert(0, str(ROOT))

    from pda.freeze import build_manifest, verify_manifest
    from pda.scenarios import build_suite

    suite = build_suite()

    # Strict verification prevents a changed benchmark
    # from being mixed with earlier model results.
    verify_manifest(
        REF,
        suite,
        strict_code=True,
    )

    frozen = build_manifest(suite)
    snapshot = repo_snapshot(frozen)

    lock_path = (
        ROOT / "results/.gptoss20-groq-cleanroom.lock"
    )

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

        except BlockingIOError:
            raise RuntimeError(
                "Another clean-room launcher is active."
            )

        running = existing_groq_processes()

        if running:
            raise RuntimeError(
                f"Another Groq audit is running: {running}. "
                "Do not start concurrent Groq experiments."
            )

        if (
            options.status
            and not OUT.with_suffix(
                ".provenance.json"
            ).exists()
        ):
            print("Clean-room run not started yet.")
            return 0

        record_provenance(snapshot)

        count, complete = check_results(suite)

        if options.status:
            return 0

        if complete and count == TOTAL:
            print(
                "PASS: Already completed 216/216"
            )
            return 0

        consecutive_transient = 0

        for round_id in range(
            1, options.max_rounds + 1
        ):
            # Detect inference-configuration changes.
            if repo_snapshot(
                build_manifest(suite)
            ) != snapshot:
                raise RuntimeError(
                    "Code or environment changed during the run."
                )

            # Also detect manually launched raw workers.
            if existing_groq_processes():
                raise RuntimeError(
                    "Another Groq worker started. "
                    "Stopping to protect result integrity."
                )

            return_code, tail = execute(round_id)

            new_count, complete = check_results(suite)

            if complete and new_count == TOTAL:
                print(
                    "\nPASS: COMPLETE GROQ AUDIT"
                    "\nEpisodes: 216/216"
                    "\nF0: 72/72"
                    "\nF1: 72/72"
                    "\nF2: 72/72"
                    "\nNo duplicate keys."
                    "\nFrozen manifest verified.",
                    flush=True,
                )
                return 0

            if return_code == 0:
                raise RuntimeError(
                    "Runner returned success "
                    "without complete coverage."
                )

            # Classify the FINAL failure only, not
            # earlier recovered exceptions.
            failure = (
                tail[tail.rfind("STOPPED at"):]
                if "STOPPED at" in tail
                else tail
            )

            category = error_class(failure)
            hint = parse_wait(failure)

            if category == "DAILY_QUOTA":
                consecutive_transient = 0

                delay = max(
                    65.0,
                    (hint or 900.0) + 45.0,
                )

                print(
                    f"Daily quota reached at "
                    f"{new_count}/{TOTAL}. "
                    "Resuming after cooldown.",
                    flush=True,
                )

            elif category == "MINUTE_LIMIT":
                consecutive_transient = 0

                delay = max(
                    65.0,
                    (hint or 60.0) + 10.0,
                )

                print(
                    "Minute-rate limit encountered.",
                    flush=True,
                )

            elif category == "TRANSIENT":
                consecutive_transient = (
                    consecutive_transient + 1
                    if new_count == count
                    else 1
                )

                if consecutive_transient > 3:
                    raise RuntimeError(
                        "Repeated API failures without progress. "
                        "Inspect the saved log."
                    )

                delay = (
                    90.0 * consecutive_transient
                )

                print(
                    "Transient API error; "
                    f"recovery {consecutive_transient}/3",
                    flush=True,
                )

            else:
                raise RuntimeError(
                    "Non-quota failure. "
                    "Inspect the last Groq log."
                )

            count = new_count
            cooldown(delay)

        raise RuntimeError(
            "Maximum restart count reached. "
            "All valid episodes remain saved."
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Saved episodes preserved.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except (
        RuntimeError,
        AssertionError,
        OSError,
        ValueError,
    ) as exc:
        print(
            "ERROR:", exc,
            file=sys.stderr,
        )
        raise SystemExit(2)
