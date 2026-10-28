#!/usr/bin/env bash
# User-space Ollama installation for Detour on a Linux A100 cluster.
# Does not start Ollama: launch the server ONLY inside an allocated GPU job.
# Usage: bash install_ollama_no_sudo.sh
set -Eeuo pipefail

STORE="${DETOUR_SCRATCH:-/scratch/sahil}"
ROOT="${OLLAMA_INSTALL_ROOT:-$STORE/apps/ollama}"
MODEL_STORE="${OLLAMA_MODELS:-$STORE/models/ollama}"
DOWNLOADS="$STORE/downloads"
ARCH="$(uname -m)"

if [[ "$(uname -s)" != 'Linux' ]]; then
    echo 'ERROR: This installer is intended for Linux.' >&2
    exit 1
fi
case "$ARCH" in
    x86_64) PACKAGE='ollama-linux-amd64.tar.zst' ;;
    aarch64) PACKAGE='ollama-linux-arm64.tar.zst' ;;
    *) echo "ERROR: Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

mkdir -p "$ROOT" "$MODEL_STORE" "$DOWNLOADS"
ARCHIVE="$DOWNLOADS/$PACKAGE"

if [[ -x "$ROOT/bin/ollama" ]]; then
    echo "Ollama already installed: $ROOT/bin/ollama"
else
    command -v curl >/dev/null || { echo 'ERROR: curl is required.' >&2; exit 1; }
    echo "Downloading official Ollama archive to $ARCHIVE"
    curl --fail --location --show-error --retry 3 --continue-at - \
        "https://ollama.com/download/$PACKAGE" -o "$ARCHIVE"
    echo "Extracting official Ollama archive into $ROOT"
    if command -v zstd >/dev/null 2>&1; then
        zstd -dc "$ARCHIVE" | tar -xf - -C "$ROOT"
    else
        echo 'zstd executable missing; using Python zstandard in the active environment.'
        python -c 'import zstandard' 2>/dev/null || python -m pip install zstandard
        python - "$ARCHIVE" <<'PY'
import sys
import zstandard
source = sys.argv[1]
with open(source, 'rb') as src, open(source + '.tar', 'wb') as out:
    zstandard.ZstdDecompressor().copy_stream(src, out)
PY
        tar -xf "$ARCHIVE.tar" -C "$ROOT"
        rm -f "$ARCHIVE.tar"
    fi
    [[ -x "$ROOT/bin/ollama" ]] || { echo 'ERROR: Ollama binary not found after extracting.' >&2; exit 1; }
fi

ENV_FILE="$STORE/ollama_env.sh"
cat > "$ENV_FILE" <<EOF
# Source this file in each new terminal / GPU allocation before starting Ollama.
export PATH="$ROOT/bin:\$PATH"
export OLLAMA_MODELS="$MODEL_STORE"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MAX_LOADED_MODELS=1
EOF

# Activate now, too.
export PATH="$ROOT/bin:$PATH"
export OLLAMA_MODELS="$MODEL_STORE"
export OLLAMA_HOST="127.0.0.1:11434"

printf '\nInstalled in: %s\nModel cache: %s\nEnvironment: %s\n' "$ROOT" "$MODEL_STORE" "$ENV_FILE"
"$ROOT/bin/ollama" --version
printf '\nNext, on an allocated GPU node, run:\n'
printf '  source "%s"\n' "$ENV_FILE"
printf '  nvidia-smi\n'
printf '  cd /scratch/sahil/projects/ACM_Trust/pda_audit\n'
printf '  mkdir -p logs\n'
printf '  ollama serve > logs/ollama-server.log 2>&1 &\n'
printf '  curl -fsS http://127.0.0.1:11434/api/version\n'
printf '  bash setup_detour_cluster_models.sh "$PWD"\n'
