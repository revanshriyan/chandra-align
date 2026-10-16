#!/bin/bash
# Download the WildMatch-trained LoFTR weights (45 MB) and verify SHA-256.
# Usage: bash scripts/download_wildmatch_weights.sh [dest_dir]
# Default dest: ~/workspace/wildmatch_trained/best_reconstructed.pt
#
# The share holds a zip of torch checkpoint internals ("best.pt"); this
# script reconstructs the loadable checkpoint and verifies it against the
# published SHA-256. Handles Drive's virus-scan confirm interstitial.
set -euo pipefail

DEST_DIR="${1:-$HOME/workspace/wildmatch_trained}"
DEST="$DEST_DIR/best_reconstructed.pt"
ZIP="$DEST_DIR/best.pt.zip"
FILE_ID="10QSRThuhQzKh3WUgKtaXukupFMnh4ZCw"
# SHA-256 of the Drive zip itself (byte-stable). The reconstructed .pt is
# functionally identical across rebuilds (verified tensor-equal) but its
# zip metadata timestamps vary, so the zip hash is the stable check.
EXPECT_ZIP_SHA="cbdd12cc0ac192975ca181e74ce036f216b4ccb733cdffd76fe1dea7748b0786"

mkdir -p "$DEST_DIR"

zip_ok() {
  [ -f "$ZIP" ] && [ "$(head -c 2 "$ZIP")" = "PK" ] && \
    echo "$EXPECT_ZIP_SHA  $ZIP" | sha256sum -c - >/dev/null 2>&1
}

if [ -f "$DEST" ] && zip_ok; then
  echo "weights already present and verified: $DEST"
  exit 0
fi

echo "downloading WildMatch weights (~45 MB)..."
curl -sL --max-time 300 \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
  -o "$ZIP"

# Drive serves a virus-scan interstitial for large files; follow its
# "Download anyway" form automatically.
if [ "$(head -c 2 "$ZIP")" != "PK" ]; then
  CONFIRM_URL=$(python3 - "$ZIP" << 'PYEOF'
import re, sys
s = open(sys.argv[1], errors="ignore").read()
i = s.find('id="download-form"')
if i < 0:
    sys.exit(1)
form = s[i:i+1500]
params = re.findall(r'name="([^"]*)"[^>]*?value="([^"]*)"', form)
from urllib.parse import urlencode
print("https://drive.usercontent.google.com/download?" + urlencode(params))
PYEOF
)
  if [ -z "$CONFIRM_URL" ]; then
    echo "ERROR: Drive did not serve the file (sign-in page?)."
    echo "Manual fallback: download 'best.pt' from the shared Drive link,"
    echo "save it as $ZIP, and re-run this script."
    exit 1
  fi
  curl -sL --max-time 300 "$CONFIRM_URL" -o "$ZIP"
fi

if [ "$(head -c 2 "$ZIP")" != "PK" ]; then
  echo "ERROR: download did not produce a zip file."
  exit 1
fi

if ! echo "$EXPECT_ZIP_SHA  $ZIP" | sha256sum -c -; then
  echo "ERROR: zip SHA-256 mismatch -- the download may be incomplete."
  exit 1
fi

# Reconstruct the torch checkpoint from the zipped internals.
rm -rf "$DEST_DIR/best_dl" && mkdir -p "$DEST_DIR/best_dl"
unzip -q -o "$ZIP" -d "$DEST_DIR/best_dl"
if [ -d "$DEST_DIR/best_dl/best" ]; then
  (cd "$DEST_DIR/best_dl" && zip -q -r "$DEST" best -x "best/.*")
  rm -rf "$DEST_DIR/best_dl"
else
  cp "$ZIP" "$DEST"
fi

# Structural sanity check: the archive must hold the torch payload.
if ! unzip -l "$DEST" | grep -q "best/data.pkl"; then
  echo "ERROR: reconstructed archive is missing the torch payload."
  exit 1
fi
echo "OK: $DEST (zip SHA-256 verified, torch payload present)"
