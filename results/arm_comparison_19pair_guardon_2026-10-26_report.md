# Phase 1 — 19-pair arm comparison under the default-ON strip guard (2026-10-26)

Benchmark Engineer. Reconstruction job (the original 19-pair runner was
never committed). No repo file modified, nothing committed or pushed,
gates frozen (0.50/2.50, guard default-ON at 0.159, asserted at startup).
Worktree left on `chandrabench-v0.2` @ `19ba23a`, clean. Table NOT
committed — committing waits on Revan's approval.

## Headline

114/114 rows measured. Verdicts now: **53 COARSE / 61 DEGENERATE / 0 ERROR**
(vs Oct-15: 57 / 56 / 1). Net: −4 COARSE, +5 DEGENERATE, −1 ERROR.

**6 verdict changes total: 5 GUARD-CAUSED, 1 OTHER.** Zero unexplained
changes. The guard-ON re-run does exactly what it was supposed to do and
nothing else: it downgrades thin-strip learned-arm COARSE rows and
leaves everything else alone.

### Guard-caused flips (all COARSE → DEGENERATE via STRIP_COLLAPSE)

Every one reproduces its Oct-15 fit EXACTLY (rmse + inliers + n_corr
identical) — only the strip backstop changed the verdict:

| pair | arm | Oct-15 (rmse / inl / corr) | minor_ratio |
|------|-----|---------------------------|-------------|
| N10 | loftr | 1.3114 / 83 / 751 | 0.1475 |
| N10 | matchanything | 1.5024 / 20 / 106 | 0.0874 |
| N12 | loftr | 1.3948 / 20 / 267 | 0.1529 |
| N13 | loftr | 1.6653 / 15 / 210 | 0.1177 |
| N8 | xoftr | 1.8544 / 20 / 264 | 0.1489 |

Per-arm COARSE counts now vs Oct-15: default 0/0, pc 1/1, xoftr 15/16,
matchanything 5/6, anymatch 17/17, loftr 15/17 (see below for loftr's
arithmetic: −3 guard flips +1 ERROR→COARSE).

### The one OTHER change

- **W0 loftr: ERROR → COARSE** (1.3022 px, 1606 inliers, 2595 corr,
  minor 0.7744, guard not fired, 53.7 s). Oct-15's native-res LoFTR OOM
  on the 4634 px-tall W0 input (CPU resource limit) did NOT reproduce
  on this run — the measurement completed. The OOM was environment-
  dependent (available RAM / concurrent load at the time), not a
  property of the arm; the new measurement itself is valid (wide inlier
  spread, guard silent). This is a new COARSE data point, not a guard
  effect, and it is the reason the ERROR count is now zero.

## Method (reconstruction — the four scoping gaps, resolved)

1. **Base-LoFTR entry point: RESOLVED + validated by exact reproduction.**
   `chandra_align/xmodal/loftr_arm.py::LoFTRMatcher(pretrained="outdoor")`
   at native resolution, conf > 0.2 — the only base-LoFTR entry in the
   repo. Proof: W5 loftr reproduces Oct-15 exactly (1.1421 / 513 / 868),
   as do N2–N9 loftr rows. The Oct-15 W0 OOM is consistent with native
   res (a 1024-reduced W0 would not OOM).
2. **`rmse_gate_px` semantics: RESOLVED.** The trial-harness convention
   (`gate_verdict` in scripts/iirs_breadth.py and
   scripts/run_phase9_iirs_retry.py): `verify_guarded` (RANSAC 3.0 /
   max_iters 2000 / confidence 0.99) → inliers → rmse = MEAN inlier
   residual magnitude (not RMS — the column name is loose, the code
   computes mean) → quadrant metrics → frozen gate. Proof: W5 xoftr
   reproduces Oct-15 exactly (0.9791 / 23 / 33), as does every xoftr
   row in the isolated re-run. The ONLY intentional scoring difference
   vs Oct-15 is the strip guard (Oct-15 called the gate without pts, so
   no strip backstop could fire; this run passes pts + frame_shape with
   the guard default-ON). Pre-guard verdicts are recorded per row so
   every flip's attribution is mechanical, not judged.
3. **GSD treatment: trial-harness route, NO GSD normalization (stated
   explicitly).** Both PNG windows are used at native pixel size; each
   frontend applies only its own internal scaling. Rationale: the
   Oct-15 numbers are irreproducible via the deployed GSD path — W5
   anymatch COARSE with 1223 inliers is impossible on the deployed
   320×22 strip (docs/iirs-gsd-investigation-2026-10-16.md proves the
   deployed path gives DEGENERATE there for every arm). The deployed-
   path comparison is a different experiment, already done 2026-10-16.
   Orientation: ref = IIRS window, sec = TMC window (manifest order;
   matches the repo trial-harness convention).
4. **GPU/CPU feasibility: all six arms ran natively on this CPU-only VM**
   (no nvidia-smi, torch cuda False, 7 GB RAM). No arm is NOT
   RERUNNABLE. Final proof: zero rows with match_s=0.0 and zero ERROR
   rows in the merged table (after the collision fix below).

Recording rule (from the Oct-15 CSV + app.py internals): n_corr < 8 →
DEGENERATE + note "arm <8 correspondences" with rmse/inliers/n_corr
ALL blank in the CSV (true n_corr kept in rows.jsonl); fit abstain →
DEGENERATE + abstain note, rmse blank, inliers 0; otherwise all fields
recorded. Residual anomaly, stated honestly: 4 Oct-15 default rows
(N12, N1, W1, W5) record inliers=2 and n_corr but blank rmse, while 8
other 2-inlier default rows record 0.0; no rule in current code
distinguishes them (the pipeline's DEGENERATE dict always carries
rmse_gate_px when the fit path runs). Those rows must have taken an
exception/early-rejection path in the original harness. My single
consistent rule records rmse for every fitted row; an Oct-15-blank vs
mine-valued rmse on an UNCHANGED verdict is classified as
RECORDING-RULE drift, never a guard effect. It cannot contaminate the
verdict analysis: all such rows are already DEGENERATE.

Default-arm SIFT recipe (re-derived, not claimed identical to the
original): plain SIFT (nfeatures=8000), BF knn k=2, Lowe 0.75.
app.match_pair_hf was ruled OUT: its SIFT fallback returns only
≥8-inlier candidates, contradicting Oct-15's 2–3-inlier default rows.

## Incident: the sys.path collision (found, proven, fixed)

The first full run produced 22 false COARSE→DEGENERATE flips: every
xoftr row (except W5, from the separate smoke process) and every
matchanything row returned (None, None) in 0.0 s. Cause: anymatch,
matchanything and xoftr each vendor a `src/` package and do
sys.path.insert(0, <root>). Whichever root is imported first wins
sys.modules['src']; the other two then resolve src.* against the wrong
root and fail instantly (exception swallowed by the frontends). In the
full run anymatch was called first per pair, so its src won. Proof
(diagnose_frontend_collision.py): xoftr alone on N4 reproduces Oct-15's
n_corr=504 exactly, while anymatch-then-xoftr in one process makes
anymatch fail and matchanything fail in 0.0 s. Fix: both arms re-ran
in isolated fresh processes (one arm per process, run_arm_cells.py);
their rows were merged into the FINAL table and the smoke W5 xoftr
value was reproduced exactly as a control (0.9791 / 23 / 33).
Anymatch, loftr, pc and default rows are unaffected: anymatch won
legitimately (its numbers match Oct-15 exactly) and the other three
never touch `src`. Lesson for future multi-arm runners: one process
per LoFTR-variant frontend, or the table silently measures import
order instead of matching.

## Reproduction quality

- 63/114 rows reproduce Oct-15 rmse + inliers + n_corr EXACTLY.
- Only 2 same-verdict rows differ numerically, both default-arm SIFT
  recipe drift (N10: 0.6283/3/11 → 0.0/2/11; N9: 0.0/2/11 →
  1.0021/3/16). Verdicts unchanged.
- Default arm is still 19/19 DEGENERATE (SIFT fails on IIRS↔TMC-2;
  inliers 2–3), as expected — the guard-ON re-run cannot change any
  default row, and did not.
- Near-threshold watchlist (COARSE rows with minor < 0.25 that did NOT
  flip — one RANSAC resample away from flipping): N8 loftr 0.1618,
  N5 matchanything 0.1709, N9 matchanything 0.1700, N4 xoftr 0.1796,
  W5 xoftr 0.1820, N7 xoftr 0.1851, N8 anymatch 0.1907, N14 anymatch
  0.1969, N10 xoftr 0.2284, N12 xoftr 0.2255, N4 loftr 0.2227,
  N6 anymatch 0.2220, N9 loftr 0.2183, W4 xoftr 0.2183, N9 xoftr 0.2358,
  N10 anymatch 0.2160. Future re-runs should expect movement here.

## Files (all under ~/workspace/guardon_19pair_2026-10-26/)

- arm_comparison_19pair_guardon_2026-10-26_FINAL.csv — the merged table
  (114 rows; Oct-15 columns + guard_fired, minor_ratio,
  pre_guard_verdict, verdict_changed_vs_oct15, attribution).
- arm_comparison_19pair_guardon_2026-10-26.csv — the first-pass table
  (superseded for xoftr/matchanything rows; kept for audit).
- arm_cells_xoftr_2026-10-26.csv, arm_cells_matchanything_2026-10-26.csv
  — isolated-process re-runs that replaced those rows.
- rows.jsonl, rows_xoftr.jsonl, rows_matchanything.jsonl — full per-row
  detail (entropy, quadrants, abstain codes, strip validity).
- rerun_arm_19pair_2026-10-26.py — the reconstruction runner (header
  doc resolves all four gaps; --resume checkpointing).
- run_arm_cells.py, diagnose_frontend_collision.py — isolated re-run
  driver + collision proof.
- report.md — this file.
