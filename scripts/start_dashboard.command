#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
if [ ! -x .venv/bin/python ]; then
  echo 'Run the setup instructions in README.md first.'
  exit 1
fi
echo 'Open http://127.0.0.1:8050 in your browser. Press Control-C here to stop.'
exec .venv/bin/python -m mobility dashboard
