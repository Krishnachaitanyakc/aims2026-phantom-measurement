#!/bin/sh
# Offline reproduction. By default, all generated files go to a temporary
# directory; neither source evidence nor checked-in results are modified.
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
if [ "$#" -gt 1 ]; then
  echo "Usage: ./repro.sh [output-directory]" >&2
  exit 2
fi
if [ "$#" -eq 1 ]; then
  artifact_output=$1
else
  artifact_output=$(mktemp -d "${TMPDIR:-/tmp}/aims2026-reproduction.XXXXXX")
fi
python3 verify_inputs.py
python3 -m unittest discover -s tests
python3 reproduce.py --out-dir "$artifact_output"
if [ -f ../scripts/audit_evidence.py ]; then
  python3 ../scripts/audit_evidence.py --artifact-root . \
    --summary "$artifact_output/summary.json" \
    --out "$artifact_output/camera_ready_evidence.json"
fi
echo "Reproduction outputs: $artifact_output"
