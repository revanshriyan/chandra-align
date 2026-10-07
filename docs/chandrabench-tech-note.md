# ChandraBench v0.1 — tech note

## What it is

ChandraBench v0.1 is a fixed-task, fixed-metric benchmark for lunar image
registration: 40 human-verified landmark correspondences across two
Chandrayaan-2 pairs (20 OHRC 12:09-vs-14:06, 20 TMC-2 fore-vs-nadir), scored
by a one-command evaluator. To our knowledge, no public human-verified lunar
correspondence benchmark existed at release time. The package is one download
(the `chandrabench/` directory), one command (`python eval.py`), versioned
(v0.1, frozen sha256).

## How the landmarks were collected

Candidate landmarks were proposed by an independent generator
(Harris corners + normalized cross-correlation chip search + homography
RANSAC triage; `scripts/propose_ground_truth_candidates.py`). The generator
never calls the matchers under evaluation. Each candidate was then confirmed
or rejected by independent visual review of chip pairs at full resolution
(2026-10-05): 40/40 accepted, no duplicates, quadrant spread ≥3 in every
quadrant of both pairs (OHRC 5/3/4/8, TMC-2 5/3/6/6). Per-point reviewer
notes (detector striping, stable shadows) are carried in the `caveat`
column; nothing was hidden.

Caveats, stated plainly: review is at **feature-correspondence level, not
sub-pixel**. Points carry ~0.5–1 px NCC localization noise. The two views of
each pair have genuine parallax, so the 20 points are not consistent with a
single global affine to better than ~3–4 px (a least-squares fit of the
landmarks to themselves leaves 4.27 px / 3.49 px residual). There are no
absolute LROC/SLDEM controls yet — that is the v0.2 roadmap.

## The metric and the circularity guard

Participants submit a 2×3 affine transform (source→reference,
`cv2.warpAffine` convention). The evaluator reports **forward RMSE** over all
landmarks (headline score) and **inverse RMSE** via M⁻¹, plus per-point
residuals. The guard is fail-closed:

1. Landmark CSVs are frozen; `eval.py` pins their sha256 and aborts on any
   modification.
2. The evaluator requires an explicit declaration that the landmarks were
   not used to fit the submitted transform; the declaration is recorded in
   the score report.
3. GT quadrant coverage is checked against the protocol floor (≥3/quadrant)
   and reported.

This exists because split-held-out RMSE — the usual internal diagnostic,
computed on check points drawn from the fit's own inliers — is not
independent ground truth. The benchmark's value is independence, not
flattering numbers.

## The honest baseline (release-time reference scores)

Our own pipeline, scored at release against these landmarks:

| Pair | Matcher | Inlier held-out RMSE | ChandraBench forward RMSE |
|---|---|---|---|
| OHRC_pair | LightGlue/ALIKED | 1.54 px | **5.65 px** |
| OHRC_pair | SIFT/RANSAC | 1.98 px | **5.50 px** |
| TMC-2 fore/nadir | LightGlue/ALIKED | 1.80 px | **27.35 px** |
| TMC-2 fore/nadir | SIFT/RANSAC | 1.80 px | **27.35 px** |

The inlier-based numbers look respectable; the independent numbers do not.
That disagreement — self-consistency is not correctness — is precisely what
an independent benchmark is for. We publish it because a benchmark that only
confirms its authors' numbers is advertising, not metrology.

## How to cite

> CHANDRA-ALIGN team. ChandraBench v0.1 — human-verified correspondences for
> lunar image registration (Chandrayaan-2 OHRC / TMC-2). Zenodo. DOI:
> [reserved at upload]. CC-BY-4.0.

## Roadmap

- **v0.2**: absolute controls from LROC NAC / SLDEM-derived products;
  per-cell trust-map task variant.
- **Later**: IIRS cross-modal pair once an independent check set exists;
  multi-team leaderboard page.
