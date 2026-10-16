# IIRS↔TMC-2 GSD handling in `_align_core` — investigation

Date: 2026-10-16. Reviewer item: "Fix the IIRS GSD handling in `_align_core`."
Status: investigated; default path intentionally unchanged (see recommendation).

## The mechanism (current code)

`_align_core` matches both images at a common ground sample distance
(`app.py`, in `_align_core`):

- `target_gsd_m = max(ref_gsd_m, sec_gsd_m)` — always the **coarser** of the two.
- Sensor scales: IIRS 80.0 m/px, TMC-2 5.0 m/px
  (`chandra_align/metrics/enhanced.py`, `SENSOR_PIXEL_SCALES`).
- `resize_to_common_ground_sample` downsamples only the finer image toward
  the target GSD, but floors the downsample so the **long side never drops
  below `minimum_long_side`** — 320 px for IIRS pairs
  (`IIRS_MIN_MATCH_LONG_SIDE`, `app.py:63`), 128 otherwise
  (`chandra_align/preprocessing/multimodal.py:63`).

## Worked example with the real trial windows

CrossFeat trial windows are pre-cropped to equal pixel sizes
(`~/workspace/crossfeat_data/test|val`):

| window | px (both sides) | in-pipeline TMC-2 match window | effective TMC-2 GSD | residual scale vs IIRS 80 m/px |
|---|---|---|---|---|
| W5 (test) | 2000×140 | 320×22 (scale 0.16, floored) | 31.25 m/px | 2.56× |
| N13 (test) | 1125×218 | 320×62 (scale 0.284, floored) | 17.6 m/px | 4.5× |

(The nominal 16× downsample to true 80 m/px would give W5 a 125×9 px
sliver — the figure quoted in review. The 320 floor trades GSD fidelity
for raster support; the docstring in `multimodal.py` acknowledges the
tradeoff explicitly: "Sensor metadata can imply aggressive scale factors
(e.g. OHRC/IIRS).")

Three facts that matter:

1. **The thin window is geographically honest.** At 80 m/px the TMC-2 strip
   genuinely contains 16× fewer px than stored. No resampling scheme
   recovers information that was never in the IIRS side.
2. **Geometry math stays consistent.** Points are mapped back by the actual
   scale (`actual_scale` in `resize_to_common_ground_sample`), so the
   residual scale gap costs raster quality, not coordinate correctness.
3. **The trial harness bypasses all of this.** `match_pair_anymatch`
   (`chandra_align/anymatch_frontend.py:127-141`) downsamples *both* images
   by one shared factor (positional-encoding limit), preserving relative
   scale — no GSD normalization at all. The CrossFeat/WildMatch numbers
   (W5 1.2568 px, N13 0.7556 px, fit RMSE) were measured there, with equal-
   pixel windows, leaning on LoFTR's learned scale robustness. That is why
   the trial succeeds while the pipeline path would feed the matcher a
   320×22 strip against a 2000×140 IIRS window.

## Options considered

- **A. Match at the finer GSD (upsample IIRS to 5 m/px).** Rejected on
  principle: upsampling invents no information (blur vs sharp asymmetry),
  256× memory for full windows. It would look better in a table and mean
  nothing.
- **B. Drop the 320 floor for cross-modal pairs (true 80 m/px match).**
  Geographically pure, but leaves matchers a 125×9-px sliver — almost
  certainly worse matching for the same verdict class. Experiment-only,
  behind a flag, no default-path change.
- **C. Downsample the IIRS side to the floor-limited effective GSD**
  (both sides at ~31 m/px). Destroys already-coarse information to buy a
  cosmetic "equal GSD". Rejected.
- **D. Keep current behavior, document it, declare effective match GSDs
  in telemetry.** The floor is already an acknowledged heuristic; making
  the *effective* per-side GSDs visible (derivable from `actual_scale`)
  turns a silent distortion into a measured one. Additive, gates frozen,
  zero default-path behavior change.
- **E. Full IIRS-path redesign (coarse-to-fine with IIRS as the
  resolution cap).** Real work; belongs with the parked 3D/sensor-model
  phase, not rushed under review pressure.

## Recommendation

**D.** The correct story for a judge is: IIRS↔TMC-2 is COARSE-by-design —
IIRS is 80 m/px native, so sub-pixel-vs-human-landmark was never on the
table for these pairs, and the separate-harness trial path (equal-pixel
windows, shared scale) is the legitimate cross-modal evaluation route,
already documented as honest in `docs/wildmatch-trial-2026-10-16.md`.
Ship the telemetry fields + this doc; leave pipeline behavior untouched.
B stays available as a flagged experiment if a reviewer demands the
comparison.

Open question for the project owner: approve D (telemetry + doc, one
small additive commit), or park the whole item until the landmark
calibration is done.

## Addendum 2026-10-16: deployed-path side-by-side (reviewer request)

Both trial arms run through the real `_align_core` IIRS path
(`sensor_pair_mode='Optical <-> Infrared'`, 320-px floor), next to the
trial-harness numbers. Evidence:
`results/table_iirs_deployed_path_2026-10-16.csv`.

| window | arm | trial harness (fit RMSE) | deployed IIRS path |
|---|---|---|---|
| W5 | WildMatch | 1.2568 px, 3064 inl | DEGENERATE — arm fail-closes on the 320×22 strip |
| W5 | base AnyMatch | 1.3895 px, 1097 inl | DEGENERATE — same |
| N13 | WildMatch | 0.7556 px, 2206 inl | **COARSE_ADVISORY, 49 inl, held-out 1.70 px** |
| N13 | base AnyMatch | 1.2888 px, 20 inl | DEGENERATE (9 inl) |

N13 is a genuine deployed-pipeline win: the only arm that converts
DEGENERATE→COARSE through `_align_core` on an IIRS window, with a 1.70 px
pipeline held-out RMSE (not just a trial-harness fit residual). W5's
320×22 strip remains too thin for every arm — fail-closed, honest
DEGENERATE.

## Addendum: bit-reproducibility (reviewer question)

ohrc_04 run twice on current code (seed 7): n_raw = 7562 both times,
bit-identical. The 7,629 → 7,562 shift the reviewer noticed is code drift
between the batch-A run and the current tree, not run-to-run
non-determinism. Determinism holds per code version; cross-batch
comparisons should note the code version.
