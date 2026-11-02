# Phase 2 calibration / test split — FROZEN

Committed before any Phase 2 tuning. The calibration set is frozen as-is;
the test set is fixed here so that later gate or matcher decisions cannot
cherry-pick pairs after seeing results. Any change to this split needs a
new dated amendment, not a silent edit.

## Calibration (4 pairs, FROZEN — do not re-tune or re-derive)

| Pair | Landmarks | Status |
| ---- | --------- | ------ |
| `ohrc_01` | 21, human-verified 2026-10-24 (clicker) | Labeled: landmark RMSE 1.8869 px; strip-guard calibration anchor (minor 0.2428) |
| `ohrc_04` | 18, human-verified 2026-10-24 (clicker) | UNLABELED for guard calibration (fails the <2.0 px bar at 2.2591 px; 18 < 20 target) |
| `ohrc_pair` | 20 (ChandraBench v0.1) | Frozen v0.1 pair |
| `tmc2_fore_nadir` | 20 (ChandraBench v0.1) | Frozen v0.1 pair |

Coordinate note: `ohrc_01` / `ohrc_04` landmarks live in v0.1 space
(loaded-4096) after the Phase 2 re-derivation; `ohrc_pair` /
`tmc2_fore_nadir` keep their audited v0.1 windows. No coordinate
re-derivation is permitted under this freeze.

## Test (6 pairs, fixed order)

| ID | Pair | Clicking order note |
| -- | ---- | ------------------ |
| T1 | OHRC independent-acquisition pair (2021-10-23 vs 2024-04-25, sec=2024 / ref=2021) | FIRST clicking target. Different orbits, 2.5 yr apart, different processing levels. Coordinate space must be recorded in the clicker PAIRS entry when its image set is built. |
| T2 | IIRS→TMC-2 CrossFeat window W5 (test split) | Chosen from the CrossFeat TEST split (never trained on). Trial history: XoFTR 0.9791 px fit, AnyMatch 1.4531 px fit. |
| T3 | IIRS→TMC-2 CrossFeat window N13 (test split) | Chosen from the CrossFeat TEST split (never trained on). Trial history: WildMatch 0.7556 px fit, deployed held-out 1.7009 px. |
| T4 | NAC pair (LROC NAC ortho) | Only if a NAC product passes the footprint screen in `docs/phase2-nac-selene-inventory.md`. |
| T5 | SELENE (Kaguya) pair | Only if a SELENE product passes the footprint screen in `docs/phase2-nac-selene-inventory.md`. |
| T6 | NAC or SELENE pair (second site) | Only if a second screened product exists; else the fallback rule below. |

## Fallback rule (explicit, applies to T4–T6)

If, at the time clicking starts, a T4/T5/T6 slot has no
footprint-screened NAC/SELENE product, that slot is filled by a
region-diverse fallback pair, in this priority order:

1. An OHRC or TMC-2 pair from a lunar region NOT overlapping any
   calibration-pair footprint, built from products already on disk
   (PRADAN holdings first — no new download).
2. Only if no on-disk product qualifies, a newly ordered PRADAN
   product for a non-overlapping region.

Fallback pairs must still satisfy the per-pair bar below, must span at
least two distinct regions across the filled slots, and must prefer
independent acquisitions (different dates) over same-day pairs. A
fallback substitution is recorded in this file as an amendment with the
date and the reason (which screen failed); the calibration set is
never touched by a fallback.

## Per-pair bar (applies to every test pair)

- ≥ 25 landmarks per pair.
- ≥ 4 landmarks per quadrant (quadrants on the reference frame).
- ≥ 50 px apart (nearest-neighbour distance between landmarks on the
  reference frame, in clicker coordinates).
- Non-collinear: landmarks must not collapse to a line or thin strip
  (the strip guard's spatial-support check applies to any fitted
  transform using these points).
- Only rows exported as VERIFIED under the clicker protocol count
  toward the bar (coordinates + feature type + confidence present).
  CANDIDATE rows do not count.

## What this split does not decide

- Gate thresholds, matcher arms, and evaluation metrics are decided
  elsewhere (Phase 3, gated on this split plus the Phase 2 table).
- Landmark CLICKING stays human-only. This document, the clicker
  tooling, and any candidate pre-placement are preparation only and
  create no verified landmarks.

## Amendments

### Amendment 2026-11-02 — Sun-angle source; T4–T6 sourcing and screening outcome

(a) **Sun angles are computed** from each product's label UTC time
coordinates plus its footprint geometry, for all Phase 2 pairs. The
delivered PDS4 labels and geometry products were verified on disk to
carry no illumination angles (the geometry CSV columns are Longitude,
Latitude, Pixel, Scan only), so reading angles from the labels is not
available and is closed as the primary source. Visible shadow
direction in the imagery is the recorded cross-check; a shadow
estimator is the per-product fallback only where time or footprint is
unusable, flagged in the pair record. SPICE remains parked with the
deferred 3D work.

(b) **T4–T6 are sourced via archive fetches** (the team's explicit
choice, superseding the region-diverse fallback as the current plan;
the fallback remains the contingency if the fetches fail). Pairs must
be the archives' own published files, not pairs manufactured from the
team's data. Screening outcome, 2026-11-02:

- **Q3 — SELENE TC via JAXA DARTS open listings: one slot filled.**
  Morning/evening MAP pair `TCO_MAPm04_N12E339N09E342SC` +
  `TCO_MAPe04_N12E339N09E342SC` (Copernicus region, lat 9–12°N,
  lon 339–342°E, 7.4 m/px, identical grid), each 301,989,888 B,
  downloaded and hash-verified. A real cross-illumination pair.
- **Q1 — LROC NAC EDR via ODE: blocked.** Candidate pairs were
  screened (e.g. Hadley `M157934151L` × `M113941548L`, incidence
  35° vs 59°), but the ODE REST validator returns "Invalid IIPT" for
  EDRNAC product queries, so no files could be fetched. Retry path is
  the ODE web search UI; it has not been approved or attempted yet.
- **Q2 — SELENE TC via JAXA SOAC: blocked.** The route works but is
  account-gated; no account was created. Creating one is a human
  decision, not a workaround item.

Two of the three T4–T6 slots remain open pending Q1/Q2 unblocking or
a later dated amendment substituting screened alternates (Tycho and
Kepler TC morning/evening tiles via Q3 are already screened).
