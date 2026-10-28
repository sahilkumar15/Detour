#!/usr/bin/env python3
"""Quota-aware, single-writer resume for frozen Detour Groq grid."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "results/grid-v8-gptoss20-groq"
REF = ROOT / "results/grid-v8-qwen7.manifest.json"

MODEL = "groq:openai/gpt-oss-20b"
FEEDBACK = ("F0", "F1", "F2")
TOTAL = 216


def quota_delay(text):
    match = re.search(
        r"please try again in\s+"
        r"(?:(\d+(?:\.\d+)?)h)?"
        r"(?:(\d+(?:\.\d+)?)m)?"
        r"(?:(\d+(?:\.\d+)?)s)?",
        text,
        re.I,
    )

    if not match or not any(match.groups()):
        return None

    h, m, s = (
        float(v or 0)
        for v in match.groups()
    )

    return 3600 * h + 60 * m + s


def daily_limit(text):
    lower = text.lower()

    return (
        ("429" in lower or "ratelimiterror" in lower)
        and any(
            term in lower
            for term in (
                "tokens per day",
                "requests per day",
                "(tpd)",
                "(rpd)",
            )
        )
    )


def conflicting_pids():
    hits = []

    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue

        if int(entry.name) == os.getpid():
            continue

        try:
            args = [
                part.decode(errors="replace")
                for part in
                (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]

            if not any(
                a == "run_audit.py"
                or a.endswith("/run_audit.py")
                for a in args
            ):
                continue

            if "--out" not in args:
                continue

            i = args.index("--out")
            target = Path(args[i + 1])

            if not target.is_absolute():
                target = (entry / "cwd").resolve() / target

            if target.resolve() == OUT.resolve():
                hits.append(int(entry.name))

        except (OSError, IndexError):
            continue

    return hits


def inspect():
    sys.path.insert(0, str(ROOT))

    from pda.freeze import verify_manifest
    from pda.scenarios import build_suite

    suite = build_suite()

    assert REF.exists(), f"Missing reference: {REF}"
    assert len(suite) == 24, "Expected 24 tasks"

    verify_manifest(REF, suite)

    manifest_path = OUT.with_suffix(".manifest.json")
    manifest = None

    if manifest_path.exists():
        verify_manifest(manifest_path, suite)

        manifest = json.loads(
            manifest_path.read_text()
        )

        info = manifest.get("run", {})

        expected_run = {
            "agents": [MODEL],
            "conditions": list(FEEDBACK),
            "reps": 3,
            "max_steps": 10,
            "max_tokens": 900,
        }

        for key, expected in expected_run.items():
            assert info.get(key) == expected, (
                f"Manifest configuration mismatch: {key}"
            )

    expected = {
        (MODEL, condition, rep, task.task_id)
        for condition in FEEDBACK
        for rep in range(3)
        for task in suite
    }

    seen = set()
    counts = Counter()

    data = OUT.with_suffix(".jsonl")

    if data.exists():
        for line_no, line in enumerate(
            data.read_text().splitlines(), 1
        ):
            if not line.strip():
                continue

            try:
                r = json.loads(line)
                key = (
                    r["model"],
                    r["feedback"],
                    r["repetition"],
                    r["task_id"],
                )

            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid row {line_no}: {exc}"
                ) from exc

            if key not in expected:
                raise RuntimeError(
                    f"Unexpected row {line_no}: {key}"
                )

            if key in seen:
                raise RuntimeError(
                    f"DUPLICATE row {line_no}: {key}. "
                    "Do not overwrite or cherry-pick outcomes."
                )

            if r.get("error"):
                raise RuntimeError(
                    f"Error row {line_no}: {r['error']}"
                )

            seen.add(key)
            counts[r["feedback"]] += 1

    complete = bool(
        manifest
        and manifest.get("run", {}).get("complete") is True
    )

    if complete and len(seen) != TOTAL:
        raise RuntimeError(
            "Manifest says complete but row count disagrees."
        )

    print(
        f"Validated {len(seen)}/{TOTAL}: "
        + ", ".join(
            f"{fb}={counts[fb]}/72"
            for fb in FEEDBACK
        ),
        flush=True,
    )

    return len(seen), complete


def invoke(number):
    command = [
        sys.executable,
        "-u",
        "run_audit.py",
        "--agents", "oss20",
        "--conditions", "F0,F1,F2",
        "--reps", "3",
        "--max-steps", "10",
        "--max-tokens", "900",
        "--verify-freeze", str(REF),
        "--out", str(OUT),
    ]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log = (
        ROOT / "logs"
        / f"groq-resume-{stamp}-{os.getpid()}-{number}.log"
    )

    print(f"Log: {log}", flush=True)

    chunks = []

    with log.open("w", encoding="utf-8") as file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                file.write(line)
                file.flush()
                chunks.append(line)

        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise

        rc = proc.wait()

    return rc, "".join(chunks)


def pause(seconds):
    end = time.monotonic() + seconds

    while time.monotonic() < end:
        remaining = end - time.monotonic()

        print(
            f"Quota cooldown: {remaining / 60:.1f} "
            "minutes; no API calls.",
            flush=True,
        )

        time.sleep(min(60, remaining))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--max-pauses",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    if args.self_test:
        assert quota_delay(
            "Please try again in 6m24.48s"
        ) == 384.48

        assert quota_delay(
            "Please try again in 1h2m3s"
        ) == 3723

        assert daily_limit(
            "RateLimitError: 429 on tokens per day (TPD)"
        )

        assert not daily_limit(
            "RateLimitError: 429 tokens per minute (TPM)"
        )

        print("PASS: cooldown parser")
        return 0

    if args.max_pauses < 1:
        parser.error("--max-pauses must be positive")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    lock_path = ROOT / "results/.groq-audit-resume.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

        except BlockingIOError:
            print(
                "ERROR: Another recovery launcher is running.",
                file=sys.stderr,
            )
            return 2

        active = conflicting_pids()

        if active:
            print(
                "ERROR: Another writer is running. "
                f"PIDs: {active}. Do not run concurrently.",
                file=sys.stderr,
            )
            return 2

        n, done = inspect()

        if args.check_only:
            return 0 if n == TOTAL and done else 3

        if done and n == TOTAL:
            print("PASS: Groq audit already complete.")
            return 0

        for attempt in range(1, args.max_pauses + 2):
            active = conflicting_pids()

            if active:
                raise RuntimeError(
                    f"Another writer started: {active}. Stopping."
                )

            print(
                f"=== Resume attempt {attempt} ===",
                flush=True,
            )

            rc, log = invoke(attempt)

            n, done = inspect()

            if done and n == TOTAL:
                print(
                    "PASS: COMPLETE 216/216, unique episodes, "
                    "verified manifest."
                )
                return 0

            if rc == 0:
                raise RuntimeError(
                    "Runner exited 0 but audit is incomplete."
                )

            if not daily_limit(log):
                raise RuntimeError(
                    "Non-quota failure. Inspect the latest log; "
                    "do not retry blindly."
                )

            if attempt > args.max_pauses:
                print(
                    "Quota-pause limit reached; "
                    "saved rows preserved.",
                    file=sys.stderr,
                )
                return 2

            hinted = quota_delay(log)

            wait = max(
                60,
                (hinted if hinted is not None else 1800) + 45,
            )

            print(
                f"Groq daily quota reached. "
                f"Waiting {wait:.0f}s then resuming "
                f"from {n}/{TOTAL}. "
                "This does not increase your quota.",
                flush=True,
            )

            pause(wait)

    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (RuntimeError, AssertionError) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
