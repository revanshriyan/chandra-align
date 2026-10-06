# Phase 9 Report — IIRS retry (cross-modal)

## Data

Verified 2023-12-25 crop pair, rebuilt locally from retained products:
- IIRS `ch2_iir_nci_20231225T1904122779_d_img_d18`, bands 712.3 / 2852.6 / 5009.7 nm,
  crop lines 0–4634, samples 0–218 (from the 12842×250 BSQ array).
- TMC-2 fore `ch2_tmc_ncf_20231101T0125121344_d_img_d18`, crop lines 111844–189999,
  samples 1721–3999, read as raw uint16 BSQ (190000×4000) via numpy memmap.
- Common-GSD working pair: TMC-2 downsampled to the IIRS grid with
  `cv2.INTER_AREA` → both 4634×218. Scored in IIRS working pixels.

Key preprocessing find: the IIRS band is dominated by vertical detector
striping (up to 1605 DN column offsets). A per-column-median destripe was
applied first — without it, matchers key on stripes, not terrain.

## Toolkit (`chandra_align/xmodal/`)

| Module | Role |
| --- | --- |
| `wallis` | Wallis local-contrast normalization (A/B branch) |
| `gsd` | Common-GSD resampling; unknown GSD raises instead of silent 1:1 |
| `tps` | Smoothing thin-plate-spline warp (λ≈0.05, numpy solver) |
| `goa_nmi` | Polarity-invariant detection (gradient-magnitude Harris) + NMI rerank |
| `funnel` | Stage-by-stage diagnosis (detection → descriptor → RANSAC) |
| `sift_rescue` | SIFT rescue + TPS, gated by ≥13 inliers and Gate 3 |
| `loftr_arm` | LoFTR dense arm (torch/kornia); reports "unavailable", never fakes |

## Funnel diagnosis (the honest story)

| Band | Detected (IIRS/TMC-2) | Lowe-good | Outcome |
| --- | --- | --- | --- |
| 712 nm | 4767 / 10000 | 1 | ABSTAIN (INSUFFICIENT_UNIQUE) |
| 712 nm + Wallis | 10000 / 10000 | 2 | ABSTAIN (INSUFFICIENT_UNIQUE) |
| 2852 nm | 4424 / 10000 | 25 | ABSTAIN (NO_VALID_MODEL) |
| 5009 nm | 5000 / 10000 | 13 | DEGENERATE_FAILURE (2 inliers < 8) |

Detection works; **the funnel collapses at descriptor matching**. SIFT
descriptors do not bridge the IIRS↔TMC-2 modality gap. Wallis helps detection
(4767→10000) but not matching (1→2) — kept as a measured negative, not a default.

GOA→NMI: 21×21-patch NMI sits at chance level (0.10–0.25) at all offsets on
all bands — no measurable mutual information at that scale, even after
destriping and common-GSD. Phase correlation shows a sharp-looking peak at
(−8, −148) px, but NMI does not improve there: spurious (horizontal banding),
not signal. Reported, not trusted.

TPS rescue: correctly reports RESCUE_FAILED (insufficient correspondences).
LoFTR: arm implemented; live evaluation blocked on torch/kornia install
(pending) — scored as skipped, not as a result.

## Verdict

**No successful IIRS↔TMC-2 registration.** Every arm ends in ABSTAIN,
DEGENERATE_FAILURE, or honest skip. The 2852 nm band is the most promising
(25 Lowe matches) and is the recommended starting point for the LoFTR
evaluation once torch is available. Nothing was tuned to improve optics:
thresholds, gates, and the Lowe ratio are untouched.

## Artifacts

- `results/phase9_iirs_retry.json`, `results/table_phase09_iirs_retry.csv`
- `scripts/run_phase9_iirs_retry.py` (re-runnable)
- 17 `tests/test_xmodal.py` tests pass; no regressions in neighboring suites
