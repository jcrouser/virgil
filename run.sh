#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
STAMP_FILE="$VENV_DIR/.requirements.stamp"

usage() {
  cat <<'USAGE'
Usage:
  ./run.sh
  ./run.sh Werner
  ./run.sh McKay output/mckay-run
  ./run.sh corpus/Werner /tmp/virgil-output

Defaults:
  corpus: corpus/Werner
  output: output
USAGE
}

resolve_corpus_dir() {
  local value="${1:-corpus/Werner}"

  if [[ -d "$ROOT_DIR/$value" ]]; then
    printf '%s\n' "$ROOT_DIR/$value"
    return
  fi

  if [[ -d "$ROOT_DIR/corpus/$value" ]]; then
    printf '%s\n' "$ROOT_DIR/corpus/$value"
    return
  fi

  printf 'Could not find corpus directory: %s\n' "$value" >&2
  printf 'Try one of: Werner, McKay, Enumerated, or corpus/Werner\n' >&2
  exit 1
}

ensure_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
  fi

  if [[ ! -f "$STAMP_FILE" || "$REQUIREMENTS_FILE" -nt "$STAMP_FILE" ]]; then
    "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"
    date > "$STAMP_FILE"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CORPUS_DIR="$(resolve_corpus_dir "${1:-}")"
OUTPUT_DIR="${2:-output}"

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

ensure_venv

"$VENV_DIR/bin/python" "$ROOT_DIR/pipeline.py" "$CORPUS_DIR" "$OUTPUT_DIR"
