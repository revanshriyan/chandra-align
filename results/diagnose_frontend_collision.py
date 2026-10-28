#!/usr/bin/env python3
"""Diagnose the xoftr/matchanything 0.0s failures in the full run.

Hypothesis: the three LoFTR-variant frontends (anymatch, matchanything,
xoftr) each vendor a `src/` package and do sys.path.insert(0, <root>).
Whichever root is imported first wins `sys.modules['src']`; later
frontends then resolve `src.*` against the WRONG root and fail instantly
(exception swallowed -> (None, None) in 0.0s). The full run called
anymatch first per pair, so anymatch's src won and xoftr/matchanything
failed on every pair; the smoke process called xoftr first, so it worked.

Test: (1) xoftr alone on N4 -> expect success; (2) anymatch then xoftr
on N4 in one process -> expect xoftr (None, None); (3) matchanything
alone on N5 -> expect success.
"""
import os
import sys
import time

import cv2

REPO = os.path.expanduser("~/workspace/chandra-align")
sys.path.insert(0, REPO)
BASE = os.path.expanduser("~/workspace/crossfeat_data")


def load(pair, side):
    return cv2.imread(os.path.join(BASE, "train", f"modality_{side}", f"{pair}.png"),
                      cv2.IMREAD_GRAYSCALE)


def timed(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return out, round(time.perf_counter() - t0, 1)


print("== test 1: xoftr alone on N4 ==", flush=True)
from chandra_align.xoftr_frontend import match_pair_xoftr
iirs, tmc = load("N4", "iirs"), load("N4", "tmc")
(out, dt) = timed(match_pair_xoftr, iirs, tmc)
print(f"xoftr alone: dt={dt}s n={None if out[0] is None else len(out[0])}", flush=True)

print("== test 2: anymatch then xoftr on N4, same process ==", flush=True)
from chandra_align.anymatch_frontend import match_pair_anymatch
(out_a, dt_a) = timed(match_pair_anymatch, iirs, tmc)
print(f"anymatch: dt={dt_a}s n={None if out_a[0] is None else len(out_a[0])}", flush=True)
(out_x, dt_x) = timed(match_pair_xoftr, iirs, tmc)
print(f"xoftr after anymatch: dt={dt_x}s n={None if out_x[0] is None else len(out_x[0])}", flush=True)
print(f"sys.modules['src'] file: {sys.modules.get('src')}", flush=True)

print("== test 3: matchanything alone on N5 (fresh import order note) ==", flush=True)
print("(matchanything imports src.* too; in THIS process src is already",
      "xoftr/anymatch's, so a failure here is also consistent with the",
      "collision hypothesis, not a clean test — see run_arm_cells.py for",
      "the isolated re-runs.)", flush=True)
from chandra_align.matchanything_frontend import match_pair_matchanything
iirs5, tmc5 = load("N5", "iirs"), load("N5", "tmc")
(out_m, dt_m) = timed(match_pair_matchanything, iirs5, tmc5)
print(f"matchanything after anymatch/xoftr imports: dt={dt_m}s "
      f"n={None if out_m[0] is None else len(out_m[0])}", flush=True)
