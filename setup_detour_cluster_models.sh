#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/sahil/projects/ACM_Trust/pda_audit"
ENV_FILE="/scratch/sahil/ollama_env.sh"

cd "$PROJECT"

# ============================================================
# 1. Activate user-installed Ollama (no sudo required)
# ============================================================

if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

export PATH="/scratch/sahil/apps/ollama/bin:$PATH"
export OLLAMA_MODELS="/scratch/sahil/models/ollama"
export OLLAMA_HOST="http://127.0.0.1:11434"

mkdir -p "$OLLAMA_MODELS" configs/models logs results

command -v ollama >/dev/null || {
    echo "ERROR: Ollama executable not found."
    exit 1
}

command -v curl >/dev/null || {
    echo "ERROR: curl not found."
    exit 1
}

command -v python >/dev/null || {
    echo "ERROR: Python not found."
    exit 1
}

echo "========================================"
echo "OLLAMA INSTALLATION"
echo "========================================"

command -v ollama
ollama --version || true

echo
echo "========================================"
echo "MODEL STORAGE"
echo "========================================"

df -h "$OLLAMA_MODELS"

# ============================================================
# 2. Check Ollama server; start only if not running
# ============================================================

if ! curl -fsS --connect-timeout 2 --max-time 4 \
    http://127.0.0.1:11434/api/version >/dev/null 2>&1
then
    echo "Starting Ollama server..."

    nohup ollama serve \
        > logs/ollama-server.log 2>&1 < /dev/null &

    server_pid=$!
    ready=0

    for ((i=0; i<120; i++)); do
        if curl -fsS --connect-timeout 2 --max-time 3 \
            http://127.0.0.1:11434/api/version \
            >/dev/null 2>&1
        then
            ready=1
            break
        fi

        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi

        sleep 1
    done

    if [[ "$ready" != 1 ]]; then
        echo "ERROR: Ollama server failed."
        tail -n 70 logs/ollama-server.log
        exit 1
    fi
fi

echo
echo "OLLAMA SERVER READY:"
curl -fsS http://127.0.0.1:11434/api/version
echo

# ============================================================
# 3. Download a model only when missing
# ============================================================

download_if_missing() {
    local model="$1"
    local attempt

    if ollama show "$model" >/dev/null 2>&1; then
        echo "ALREADY INSTALLED: $model"
        return 0
    fi

    for attempt in 1 2 3; do
        echo
        echo "========================================"
        echo "DOWNLOADING: $model"
        echo "Attempt: $attempt / 3"
        echo "========================================"

        if ollama pull "$model"; then
            if ollama show "$model" >/dev/null 2>&1; then
                echo "DOWNLOAD SUCCESSFUL: $model"
                return 0
            fi
        fi

        if [[ "$attempt" != 3 ]]; then
            echo "Download interrupted. Retrying..."
            sleep 10
        fi
    done

    echo "ERROR: Unable to download $model"
    echo "Rerun this script to retry/resume."
    return 1
}

# ============================================================
# 4. Create Detour-specific model configuration
# ============================================================

configure_model() {
    local base_model="$1"
    local detour_name="$2"
    local modelfile="configs/models/Modelfile.${detour_name}"

    download_if_missing "$base_model"

    cat > "$modelfile" <<EOF
FROM $base_model
PARAMETER num_ctx 8192
EOF

    echo
    echo "CREATING DETOUR MODEL: $detour_name"

    ollama create "$detour_name" -f "$modelfile"

    ollama show "$detour_name" \
        > "logs/${detour_name}-ollama-show.txt"

    echo "CREATED: $detour_name"
}

# ============================================================
# 5. Download and configure all three research models
# ============================================================

configure_model \
    "gpt-oss:20b" \
    "gptoss20-detour"

configure_model \
    "mistral-small3.2:24b" \
    "mistral24-detour"

configure_model \
    "llama3.3:70b" \
    "llama70-detour"

# ============================================================
# 6. Verify context window and tool-calling support
# ============================================================

echo
echo "========================================"
echo "VERIFYING ALL MODELS"
echo "========================================"

python - <<'PY'
import json
import re
import urllib.request

models = [
    "gptoss20-detour:latest",
    "mistral24-detour:latest",
    "llama70-detour:latest",
]

for model in models:
    payload = json.dumps({"model": model}).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        info = json.load(response)

    capabilities = info.get("capabilities", [])
    parameters = info.get("parameters") or ""

    tools_ok = "tools" in capabilities

    match = re.search(
        r"(?m)^\s*num_ctx\s+(\d+)\s*$",
        parameters,
    )

    context = int(match.group(1)) if match else None
    context_ok = context == 8192

    print()
    print("Model:", model)
    print("Tool calling:", tools_ok)
    print("Configured context:", context)

    if not tools_ok or not context_ok:
        raise SystemExit(
            "VALIDATION FAILED: " + model
        )

    print("STATUS: PASSED")

print()
print("ALL THREE MODELS VALIDATED SUCCESSFULLY")
PY

# ============================================================
# 7. Save model inventory for paper reproducibility
# ============================================================

echo
echo "========================================"
echo "FINAL INSTALLED MODEL LIST"
echo "========================================"

ollama list | tee logs/ollama-model-inventory.txt

echo
echo "========================================"
echo "MODEL SETUP COMPLETE"
echo "========================================"
echo "Configured models:"
echo "  gptoss20-detour:latest"
echo "  mistral24-detour:latest"
echo "  llama70-detour:latest"
echo
echo "Next: run clean-task competence tests on"
echo "an authorized A100 GPU allocation."
echo
