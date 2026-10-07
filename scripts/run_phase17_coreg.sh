#!/bin/bash
# Phase 17: ISIS3 coreg comparison driver (reproducible).
#
# Compares USGS ISIS3 `coreg` against CHANDRA-ALIGN on identical crops.
# Scope: OHRC/TMC-2 only (no IIRS camera model in ISIS3).
#
# Prerequisites:
#   - ISIS3 installed (tested: 10.0.0 from usgs-astrogeology channel),
#     ISISROOT exported, $ISISROOT/bin on PATH.
#   - Crops in ../crops/ as <name>_ref.raw / <name>_src.raw (1024x1024):
#       ohrc_w0000  : uint8  (OHRC 12:09 -> 14:06)
#       tmc2_w0000  : uint16 (TMC-2 fore -> nadir; our COARSE case)
#       tmc2_w0001  : uint16 (TMC-2 fore -> nadir; our DEGENERATE case)
#     Cut with the measured inter-product affines from
#     scripts/make_benchmark_crops.py (same mapping Phase 13 used).
#   - Wide-search template ../wide_search.def (pattern 20px, search 140px),
#     derived from $ISISROOT/appdata/templates/autoreg/coreg.maxcor.p2020.s5050.def.
#
# Outputs (this directory): per-pair cubes, .net networks, *_flat.txt chip
# tables, and the summary parsed into results/phase17_coreg.csv by hand.
# Nothing is pushed; all results stay local.
set -u
export LC_ALL=C.UTF-8

: "${ISISROOT:?set ISISROOT to your ISIS3 install}"
export PATH="$ISISROOT/bin:$PATH"

WORK="$(cd "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd)/phase17_run"
CROPS="$(cd "$(dirname "$0")/../../phase17" >/dev/null 2>&1 && pwd)/crops"
mkdir -p "$WORK"
cd "$WORK"

import_pair() {
  # $1 = name, $2 = u8|u16
  local name="$1" dt="$2" bittype="UnsignedByte"
  [ "$dt" = "u16" ] && bittype="UnsignedWord"
  raw2isis "from=$CROPS/${name}_ref.raw" "to=${name}_base.cub" \
    samples=1024 lines=1024 bands=1 "bittype=$bittype" byteorder=Lsb skip=0
  raw2isis "from=$CROPS/${name}_src.raw" "to=${name}_match.cub" \
    samples=1024 lines=1024 bands=1 "bittype=$bittype" byteorder=Lsb skip=0
}

run_coreg() {
  # $1=name $2=TRANSFORM $3=DEGREE(or -) $4=deffile(or stock) $5=tag
  local name="$1" xf="$2" deg="$3" def="$4" tag="$5"
  local args=(FROM="${name}_match.cub" MATCH="${name}_base.cub"
              TO="${name}_reg_${tag}.cub" TRANSFORM="$xf"
              ONET="${name}_${tag}.net" FLATFILE="${name}_${tag}_flat.txt"
              ROWS=9 COLUMNS=9)
  [ "$deg" != "-" ] && args+=(DEGREE="$deg")
  [ "$def" != "stock" ] && args+=(DEFFILE="$def")
  coreg "${args[@]}"
}

STOCK="$ISISROOT/appdata/templates/autoreg/coreg.maxcor.p2020.s5050.def"
WIDE="$CROPS/../wide_search.def"

import_pair ohrc_w0000 u8
import_pair tmc2_w0000 u16
import_pair tmc2_w0001 u16

# Synthetic control: shift base by known (dy=+12, dx=-7); coreg must recover it.
python3 - "$CROPS/tmc2_w0000_ref.raw" <<'EOF'
import numpy as np, sys
r = np.fromfile(sys.argv[1], dtype='<u2').reshape(1024, 1024)
s = np.roll(r, shift=(12, -7), axis=(0, 1))
s[:12, :] = r[0:1, :]; s[:, -7:] = r[:, -8:-7]
s.astype('<u2').tofile('synth_shifted.raw')
EOF
raw2isis from=synth_shifted.raw to=synth_match.cub \
  samples=1024 lines=1024 bands=1 bittype=UnsignedWord byteorder=Lsb skip=0
run_coreg synth TRANSLATE - stock synth

# Real pairs.
run_coreg tmc2_w0000 TRANSLATE - stock  t_default
run_coreg tmc2_w0000 TRANSLATE - "$WIDE" t_wide
run_coreg tmc2_w0000 WARP     1 "$WIDE" w_wide
run_coreg tmc2_w0001 TRANSLATE - "$WIDE" t_wide
run_coreg ohrc_w0000 TRANSLATE - "$WIDE" t_wide

echo "DONE — inspect *_flat.txt chip tables and summarize into results/phase17_coreg.csv"
