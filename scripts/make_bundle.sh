#!/usr/bin/env bash
# Build bundle/slm_bundle.tar: the data and config files the GPU machine needs.
# Uncompressed (the Parquet files are already compressed). MANIFEST.sha256 inside the tar
# lists the SHA-256 of every other file in it; bundle/slm_bundle.tar.sha256 covers the tar.
# Run from the repo root: bash scripts/make_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/.."

FILES=(
  data/processed/features.parquet
  data/processed/labels.parquet
  data/processed/splits.parquet
  data/processed/test_ids.csv
  docs/serialisation.md
)
FILES+=(configs/*.yaml)

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
for f in "${FILES[@]}"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  cp -p "$f" "$STAGE/$f"
done
( cd "$STAGE" && shasum -a 256 "${FILES[@]}" > MANIFEST.sha256 )

mkdir -p bundle
# COPYFILE_DISABLE stops macOS tar adding ._ metadata files.
( cd "$STAGE" && COPYFILE_DISABLE=1 tar -cf - MANIFEST.sha256 "${FILES[@]}" ) > bundle/slm_bundle.tar
( cd bundle && shasum -a 256 slm_bundle.tar > slm_bundle.tar.sha256 )

echo "files: $(( ${#FILES[@]} + 1 )) (incl. MANIFEST.sha256)"
echo "bundle/slm_bundle.tar: $(wc -c < bundle/slm_bundle.tar | tr -d ' ') bytes ($(du -h bundle/slm_bundle.tar | cut -f1))"
cat bundle/slm_bundle.tar.sha256
