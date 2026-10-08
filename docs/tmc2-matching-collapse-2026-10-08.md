# TMC-2 Many-to-One Matching Collapse: Root Cause — 2026-10-08

Worker A26. **The tmc2 matching collapse is a benchmark-harness
misconfiguration, not a pipeline defect and not inherent to TMC-2
texture.** Every tmc2 DEGENERATE verdict in the 12-pair benchmark
(A15/A17/A19/A22/A25) was measured with the wrong `pixel_scale_m`.

## The misconfiguration

The benchmark scripts call:

```python
app._align_core(ref, sec, reference_sensor_name="TMC-2")
```

`pixel_scale_m` defaults to **0.25**. Inside `_align_core`:

- `ref_gsd_m = 0.25` (wrong — TMC-2 is ~5 m/px)
- `sec_gsd_m = get_sensor_pixel_scale("TMC-2")` = **5.0**
- `target_gsd_m = max(0.25, 5.0)` = 5.0
- reference match scale = `max(min(1.0, 0.25/5.0), 128/2048)` = **0.0625**

The reference crop is downsampled **16× to 128×128** while the secondary
stays 2048×2048. SIFT then matches a thumbnail's handful of indistinctive
descriptors against a full-resolution crater field: hundreds of secondary
keypoints all nearest-neighbor to the same few reference points.

The deployed UI does not do this: it passes
`pixel_scale_m=get_sensor_pixel_scale(reference_sensor)` = 5.0
(`app.py` line ~1938), so both images match at native resolution.

## Collapse quantification (tmc2_01)

| Config | `pixel_scale_m` | Match ref | Match sec | Matches | Unique ref targets | Max hits, one target | Frac in top-5 | Verdict |
|---|---|---|---|---|---|---|---|---|
| A: harness (as benchmarked) | 0.25 | 128×128 | 2048×2048 | 174 | 38 | **101** | 0.75 | DEGENERATE_FAILURE |
| B: UI-correct | 5.0 | 2048×2048 | 2048×2048 | 384 | 315 | 3 | 0.04 | **COARSE_ADVISORY** |
| Control: ohrc_01 (harness) | 0.25 | 2048×2048 | 2048×2048 | 544 | 487 | 3 | 0.02 | — |

Config A's top collapse target is (92.2, 56.4) px in match coordinates;
÷ 0.0625 → (1474.6, 902.1) px in source coordinates — **exactly the
(1474.6, 902.1) coordinate A25's nested-clique analysis found**. The
"nested cliques all the way down" are all the same phenomenon: one
thumbnail keypoint's descriptor basin.

Full numbers in `results/table_tmc2_collapse.csv`
(script: `scripts/diagnose_tmc2_collapse.py`).

## Mechanism verdicts

- **(b) GSD rescale — CONFIRMED** (precisely: 16× *asymmetric* rescale from
  the wrong `pixel_scale_m`). With the correct scale no rescale occurs and
  the collapse vanishes: 315/384 unique targets, verdict flips to COARSE.
- **(a) Lowe ratio too permissive — REJECTED.** On the collapsed inputs,
  tightening 0.75→0.50 cuts matches 37→15 while the top-5 concentration
  stays 0.33–0.57. The descriptors are genuinely ambiguous at 16×
  asymmetry; no ratio yields a usable consensus.
- **(c) Per-cell quota flooding — REJECTED.** Collapse present at quota 64
  (flag-off default): 37 matches, max 13 hits, frac-top-5 0.57. Raising the
  quota to 1e9 adds matches (127) but the structure persists (0.43).
- **(d) Ultra-high-response reference keypoints — proximate, not root.**
  The 128 px reference yields few, indistinctive descriptors; that is the
  *how*, but the *why* is (b).

## ohrc comparison

ohrc pairs are unaffected: `reference_sensor_name="OHRC"` with the 0.25
default happens to be the correct OHRC scale, and the secondary resolves
to 0.25 too — no rescale, no collapse (487/544 unique targets). The bug
only bites when the default 0.25 disagrees with the sensor's true scale.

## Ranked fix candidates

1. **Re-run the tmc2 benchmark arms with `pixel_scale_m=5.0`** (harness
   correction; zero `app.py`/`chandra_align/` changes). This is not a
   pipeline fix — it corrects the measurement to what the deployed UI
   does. Falsification test: all 6 tmc2 pairs through the true
   `_align_core`, flag off then on, seed 7 — expect the collapse gone,
   verdicts off DEGENERATE on at least some pairs, zero ohrc changes
   (re-run ohrc_01 with explicit 0.25 to confirm bit-identical), and the
   deform stage evaluated honestly on real point sets.
2. **Pipeline hardening (optional, not recommended now):** refuse or warn
   on extreme asymmetric rescale (e.g. min match scale < 0.25) in
   `_align_core`. The pipeline already failed closed correctly
   (DEGENERATE with reason), so this is defense-in-depth with its own
   validation cost. Deferred unless a real caller hits it.

## Clean negatives

- The collapse is **not** inherent to TMC-2 fore→nadir texture: config B
  matches 384 points with no collapse on the same pair.
- A22's standalone probe consensus (5,921 matches, no rescale) and the
  pipeline's collapsed set are reconciled: different inputs, different
  structure. Both were measured correctly; the pipeline input was wrong.
- A25's conclusion ("the tmc2 DEGENERATEs are a MATCHING problem, not a
  hook problem") stands but is refined: it is a *harness-input* problem,
  not a matcher-design problem.

## Files

- `scripts/diagnose_tmc2_collapse.py` — config A/B reproduction + collapse
  quantification through the true `_align_core` (spy on `match_pair_hf`
  inputs; /tmp stubs for gradio/spaces)
- `results/table_tmc2_collapse.csv` — per-config collapse metrics with
  terrain descriptions + product IDs
- `docs/tmc2-matching-collapse-2026-10-08.md` — this document

No `app.py` / `chandra_align/` changes. Working tree otherwise untouched.
Deterministic (seed 7), watermark-scanned clean.
