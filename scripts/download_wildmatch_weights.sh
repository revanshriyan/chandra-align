#!/bin/bash
# Download the WildMatch-trained LoFTR weights (45 MB) and verify SHA-256.
# Usage: bash scripts/download_wildmatch_weights.sh [dest_dir]
# Default dest: ~/workspace/wildmatch_trained/best_reconstructed.pt
#
# Canonical source: the training-result share (Google Drive). If the direct
# download fails (Drive sign-in gate), download "best.pt" manually from the
# share, place the zip at the dest path with a .zip suffix, and re-run --
# this script will unzip, reconstruct, and verify it.
set -euo pipefail

DEST_DIR="${1:-$HOME/workspace/wildmatch_trained}"
DEST="$DEST_DIR/best_reconstructed.pt"
FILE_ID="10QSRThuhQzKh3WUgKtaXukupFMnh4ZCw"
EXPECT_SHA="4c2ef215630761e146e090f1cf4e7269f1afb5da9a9742b5187476a8e4665ca2"

mkdir -p "$DEST_DIR"

if [ -f "$DEST" ]; then
  if echo "$EXPECT_SHA  $DEST" | sha256sum -c - >/dev/null 2>&1; then
    echo "weights already present and verified: $DEST"
    exit 0
  fi
  echo "existing file failed SHA check, re-downloading..."
fi

echo "downloading WildMatch weights (~45 MB)..."
if ! curl -sL --max-time 300 \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
  -o "$DEST_DIR/best.pt.zip"; then
  echo "ERROR: download failed. Manual fallback:"
  echo "  1. download 'best.pt' from the shared Drive link,"
  echo "  2. save it as $DEST_DIR/best.pt.zip"
  echo "  3. re-run this script."
  exit 1
fi

# The share holds a zip of torch checkpoint internals; reconstruct best.pt.
if [ "$(head -c 2 "$DEST_DIR/best.pt.zip")" = "PK" ]; then
  rm -rf "$DEST_DIR/best_dl" && mkdir -p "$DEST_DIR/best_dl"
  unzip -q -o "$DEST_DIR/best.pt.zip" -d "$DEST_DIR/best_dl"
  if [ -d "$DEST_DIR/best_dl/best" ]; then
    (cd "$DEST_DIR/best_dl" && zip -q -r "$DEST" best -x "best/.*")
  else
    cp "$DEST_DIR/best.pt.zip" "$DEST"
  fi
fi

if echo "$EXPECT_SHA  $DEST" | sha256sum -c -; then
  echo "OK: $DEST"
else
  echo "ERROR: SHA-256 mismatch -- file may be a Drive sign-in page."
  echo "Manual fallback: download 'best.pt' from the share into $DEST_DIR/best.pt.zip and re-run."
  exit 1
fi
