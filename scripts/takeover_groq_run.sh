#!/usr/bin/env bash
set -Eeuo pipefail

cd /scratch/sahil/projects/ACM_Trust/pda_audit

mkdir -p scripts logs results/backups

if [[ ! -f scripts/finish_groq_audit.py ]]; then
    echo "ERROR: Groq recovery launcher is missing."
    exit 2
fi

echo "=========================================="
echo "DETOUR GROQ SAFE PROCESS TAKEOVER"
echo "=========================================="

python - <<'PY'
import os
import signal
import time
from pathlib import Path

root = Path.cwd().resolve()
target = (
    root / "results/grid-v8-gptoss20-groq"
).resolve()

expected = {
    "--agents": "oss20",
    "--conditions": "F0,F1,F2",
    "--reps": "3",
    "--max-steps": "10",
    "--max-tokens": "900",
}

pids = []

for proc in Path("/proc").iterdir():

    if not proc.name.isdigit():
        continue

    pid = int(proc.name)

    if pid == os.getpid():
        continue

    try:
        raw = (proc / "cmdline").read_bytes()

        args = [
            os.fsdecode(a)
            for a in raw.split(b"\0")
            if a
        ]

        if not any(
            Path(a).name == "run_audit.py"
            for a in args
        ):
            continue

        if "--out" not in args:
            continue

        i = args.index("--out")
        output = Path(args[i + 1])

        cwd = (proc / "cwd").resolve()

        if not output.is_absolute():
            output = cwd / output

        if output.resolve() != target:
            continue

        status = (proc / "status").read_text()

        uid_line = next(
            x for x in status.splitlines()
            if x.startswith("Uid:")
        )

        uid = int(uid_line.split()[1])

        if uid != os.getuid():
            raise SystemExit(
                f"ERROR: PID {pid} belongs to another user."
            )

        for flag, value in expected.items():

            if (
                flag not in args
                or args[args.index(flag) + 1] != value
            ):
                raise SystemExit(
                    f"ERROR: PID {pid} has different "
                    f"settings for {flag}."
                )

        state = next(
            x for x in status.splitlines()
            if x.startswith("State:")
        )

        if "Z (zombie)" in state:
            raise SystemExit(
                f"ERROR: PID {pid} is a zombie process. "
                "Its parent must reap it."
            )

        print(
            f"Verified matching Groq audit PID: {pid}",
            flush=True
        )

        pids.append(pid)

    except (OSError, IndexError, StopIteration):
        continue


# Graceful termination: preserve the already-flushed JSONL.
for pid in pids:

    print(
        f"Sending SIGTERM to PID {pid}",
        flush=True
    )

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


if pids:

    for _ in range(30):

        alive = []

        for pid in pids:

            proc = Path("/proc") / str(pid)

            if not proc.exists():
                continue

            try:
                status = (proc / "status").read_text()

                state = next(
                    x for x in status.splitlines()
                    if x.startswith("State:")
                )

                if "Z (zombie)" not in state:
                    alive.append(pid)

            except (OSError, StopIteration):
                continue

        if not alive:

            print(
                "PASS: Old audit process stopped.",
                flush=True
            )

            break

        time.sleep(1)

    else:
        raise SystemExit(
            f"ERROR: Processes still active: {alive}. "
            "Refusing to start another writer."
        )

else:
    print(
        "No old matching audit process found.",
        flush=True
    )

PY

echo
echo "=========================================="
echo "BACKING UP EXISTING GROQ RESULTS"
echo "=========================================="

STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP="results/backups/groq-${STAMP}"

mkdir -p "$BACKUP"

for suffix in jsonl manifest.json txt tex; do

    src="results/grid-v8-gptoss20-groq.${suffix}"

    if [[ -f "$src" ]]; then
        cp -p -- "$src" "$BACKUP/"
    fi

done

echo "Backup saved to: $BACKUP"

echo
echo "=========================================="
echo "CHECKING EXISTING EXPERIMENT DATA"
echo "=========================================="

# The launcher returns:
# 0: already complete
# 3: valid but incomplete
# 2: invalid data or conflicting process

if python scripts/finish_groq_audit.py --check-only; then

    echo "PASS: Groq experiment already complete."
    exit 0

else

    rc=$?

    if [[ "$rc" -ne 3 ]]; then

        echo "ERROR: Data or process validation failed."
        echo "Original files preserved in: $BACKUP"
        exit "$rc"

    fi

fi

echo
echo "=========================================="
echo "STARTING AUTOMATIC GROQ RECOVERY"
echo "=========================================="

exec python -u scripts/finish_groq_audit.py

