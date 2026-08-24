#!/usr/bin/env bash

set -e

PROJECT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
OUTPUT="../${PROJECT_NAME}_review.zip"

echo "============================================================"
echo "CREATING REVIEW ZIP"
echo "============================================================"
echo "Project : $PROJECT_DIR"
echo "Output  : $OUTPUT"
echo

rm -f "$OUTPUT"

zip -r "$OUTPUT" . \
  -x ".git/*" \
  -x ".venv/*" \
  -x ".venv_*/*" \
  -x "venv/*" \
  -x "*/__pycache__/*" \
  -x ".pytest_cache/*" \
  -x ".DS_Store" \
  -x "*/.DS_Store" \
  -x ".idea/*" \
  -x ".vscode/*" \
  \
  -x "*.png" \
  -x "*.jpg" \
  -x "*.jpeg" \
  -x "*.gif" \
  -x "*.webp" \
  -x "*.svg" \
  \
  -x "*.tif" \
  -x "*.tiff" \
  -x "*.graphml" \
  -x "*.pkl" \
  -x "*.pickle" \
  -x "*.npy" \
  -x "*.npz" \
  -x "*.dat" \
  -x "*.part" \
  \
  -x "data/*/raw/*" \
  -x "data/*/cache/*" \
  -x "data/raw/*" \
  -x "data/cache/*" \
  \
  -x "outputs/*/urban_fidelity/diagnostics_n*/*" \
  \
  -x "cache/*" \
  \
  -x "*.zip" \
  -x "*.log"

echo
echo "============================================================"
echo "ZIP CREATED"
echo "============================================================"

ls -lh "$OUTPUT"

SIZE_BYTES=$(stat -f%z "$OUTPUT")
MAX_BYTES=$((500 * 1024 * 1024))

echo

if [ "$SIZE_BYTES" -lt "$MAX_BYTES" ]; then
    echo "[OK] ZIP is below 500 MB."
else
    echo "[WARNING] ZIP is larger than 500 MB."
fi

echo
echo "Contents summary:"
unzip -l "$OUTPUT" | tail -1
