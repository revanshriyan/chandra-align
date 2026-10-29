# Phase 2 NAC / SELENE inventory + footprint screening

Purpose: decide what NAC (LROC) and SELENE (Kaguya) coverage can supply
Phase 2 test pairs T4–T6, using metadata BEFORE any large download.
No multi-GB product was downloaded for this inventory, and no product
was downloaded at all: nothing on the screened list met all three of
small + footprint-matched + useful as a pair.

## Our pair regions (ground truth for screening)

All coordinates planetocentric, east longitude, degrees.

| Region | Lat range | Lon range | Source |
| ------ | --------- | --------- | ------ |
| T1 OHRC independent pair, overlapping window | [-69.94, -68.85] | [32.18, 32.43] | `ohrc_independent_gt_2026-10-15/ohrc_independent_gt.json` corners |
| OHRC 2024-04-25 ref strip (`...1209509264`) | [-69.94, -69.10] | [32.06, 32.47] | PRADAN PDS4 label corners |
| OHRC 2024-04-25 src strip (`...1406019344`) | [-69.93, -69.10] | [32.08, 32.46] | PRADAN PDS4 label corners |
| TMC-2 fore strip 2023-11-01 (`...ncf...`) | [-89.11, -58.88] | [18.90, 139.24] | PRADAN PDS4 label corners (envelope; swath is diagonal) |
| TMC-2 nadir strip 2023-11-01 (`...ncn...`) | [-87.92, -60.26] | [19.04, 175.38] | PRADAN PDS4 label corners (envelope; swath is diagonal) |
| IIRS cube (CrossFeat source) | [-81.91, -58.03] | [91.78, 101.34] | `docs/iirs_provenance.json` |

## On disk: LROC NAC, Vikram (Chandrayaan-3 landing) site — 2.9 GB

`~/workspace/lroc_data/` (Vikram-site only — no other NAC site on disk):

| File | Size | Notes |
| ---- | ---- | ----- |
| `NAC_DTM_VIKRAMSITE1.TIF` (+ `.LBL`) | 487,659,274 B | DTM, 3 m/px, polar stereographic, center lat -69.3 / lon 32.3 |
| `NAC_ORTHO_VIKRAMSITE1_100CM.IMG` | 2,193,750,104 B | Ortho, 1 m/px |
| `NAC_ORTHO_VIKRAMSITE1_3M.IMG` | 243,781,056 B | Ortho, 3 m/px |
| `LROLRC_2001_INDEX.TAB` | 105,200,962 B | RDR volume index (this volume only, not a global EDR index) |

Footprint from the DTM label: lat [-70.0742, -68.5014], lon
[31.2285, 33.4600]. Both orthos are the SAME observation,
`M1443025251` (2023-07-03T12:06:23–12:06:53) — the 1 m / 3 m pair is
one observation at two resolutions, not two acquisitions.

### Screening results (bounding-box level)

- **T1 region: CONTAINED.** T1 lat [-69.94, -68.85] ⊂ NAC
  [-70.07, -68.50]; T1 lon [32.18, 32.43] ⊂ NAC [31.23, 33.46].
  The on-disk NAC ortho therefore covers the T1 site: a NAC↔OHRC
  cross-sensor pair at T1 needs ZERO download. Pixel-level overlap
  still to be verified at crop time (the OHRC strips are diagonal
  swaths; box containment is necessary, not sufficient).
- **OHRC 2024 strips: CONTAINED (box level).** Both strip boxes sit
  inside the NAC box. Same pixel-level caveat as above.
- **TMC-2 strips: NO OVERLAP at the Vikram site.** Linear
  interpolation along each strip's UL→LL edge puts the fore strip
  near lon ~57° and the nadir strip near lon ~67° at lat -69.3°,
  vs the site's lon 32.3° — a ~25–35° gap (≈700+ km), far beyond
  any plausible swath width. The interpolation is approximate
  (orbital strips are curved, not straight) but the gap rules out
  overlap. The TMC-2/IIRS CrossFeat region (lat ~-80, lon ~95)
  sits on the TMC-2 strips far from the Vikram site, so no
  NAC↔TMC-2 pairing exists on disk.

## On disk: LROC NAC example (ASP stereo example) — NOT a candidate

`~/workspace/qa_lunar/real/LRONAC_example/` (195 MB): two NAC
observations `M181058717LE` / `M181073012LE` with CSM camera-model
JSONs and ~100 MB `.cub` crops each. The site footprint is NOT
stated in the sidecar files read, and a web search did not resolve
the example's site from these product IDs. Status: UNSCREENED.
It must not be used as a T4–T6 pair until its footprint is
recovered (full ISIS `.cub` labels or the ASP example docs) and
screened against a target region.

## On disk: SELENE (Kaguya) — NONE

No SELENE data products exist on disk (filename search across the
workspace data directories). Any SELENE pair requires a new fetch,
screened first by the queued queries below.

## Archive offerings (screening routes, no downloads made)

### LROC NAC EDR / CDR / RDR

- **ODE product search** (PDS Geosciences Node, Moon):
  `https://ode.rsl.wustl.edu/moon/productsearch` — box search over
  NAC EDR/CDR footprints with illumination filters. The LROC index
  carries per-observation incidence and sub-solar azimuth, so the
  query can filter on illumination as well as footprint.
- **LROC QuickMap / PDS archive search** (`wms.lroc.asu.edu/lroc/search`):
  point/polygon search with lighting and viewing-geometry filters.
- **lunar-toolkit** (`pip install lunar-toolkit`): `build-db` needs a
  ~3.8 GB local EDR index download — disproportionate for a
  single-site screen; NOT recommended for this task. Prefer ODE.
- **EDR vs CDR rule** (from the LROC processing guides): download
  EDR, not CDR, for any further processing (CDRs are ~2x larger and
  may not use the latest calibration). NAC observations normally
  come in left/right camera pairs from separate pages; one frame
  suffices for a site-overlap pair.

### SELENE / Kaguya Terrain Camera (TC)

- **TC MAP products (morning / evening maps)** via the JAXA SELENE
  Data Archive (DARTS, free account required):
  `https://darts.isas.jaxa.jp/planet/pdap/selene/index.html.en`.
  Per-observation map-projected images — the actual pairing
  candidates. A morning/evening pair of the same site is also a
  ready-made sun-angle pair.
- **TC global orthomosaic, 474 m/px (64 ppd)**: global TIF
  506 MB (`.../Lunar_Kaguya_TC_Ortho_Global_64ppd_v02.tif`);
  evening mosaic 530,935,963 B (`..._Evening_Global_64ppd_v04.tif`).
  REJECTED for pairing: GSD ratios vs our frames are ~1900x (OHRC
  0.25 m), ~95x (TMC-2 5 m), ~474x (NAC 1 m). Usable only as a
  coarse context layer, never as a T-pair.
- **TC full-resolution segments (4096 px/deg, ~7.4 m/px)**:
  `pdsimage.wr.usgs.gov/downloads/Kaguya_TC/` (population in
  progress; per-segment sizes TBD at listing). A segment vs TMC-2
  (5 m) is a ~1.5x scale ratio — the plausible SELENE pairing
  shape. A segment vs OHRC (0.25 m) is ~30x — hard, same class
  as the IIRS↔TMC-2 16x case.
- **AWS Open Data (no AWS account required)**: S3 bucket
  `astrogeo-ard`, prefix `moon/kaguya/terrain_camera/monoscopic/uncontrolled/`
  (TC monoscopic COGs, map-projected, LOLA-controlled).
  Listing command: `aws s3 ls --no-sign-request
  s3://astrogeo-ard/moon/kaguya/terrain_camera/monoscopic/uncontrolled/`.
  No `aws` CLI exists on this VM, so this route is queued, not run.
  COG filenames carry observation IDs, not coordinates — region
  filtering needs the accompanying scene metadata, not the listing
  alone.

## Screening verdicts

| Candidate | Verdict | Reason / next step |
| --------- | ------- | ------------------ |
| V1: on-disk NAC ortho ↔ OHRC at T1 site | SCREENED (box level) | Zero-download cross-sensor pair candidate. Next: crop + pixel-level overlap check, then clicker image set. |
| V2: NAC 1 m ↔ NAC 3 m (same observation) | SCREENED as scale pair only | Same observation M1443025251 — tests scale, NOT independent acquisition. Cannot fill T4 as an independent pair. |
| V3: second NAC observation, Vikram site (true NAC–NAC pair) | QUEUED (query Q1) | Needs an ODE box search excluding M1443025251. Size TBD by query. |
| V4: SELENE TC morning ↔ evening MAP, Vikram site | QUEUED (query Q2) | DARTS search on the site box; also a sun-angle pair. Size TBD by query. |
| V5: SELENE TC global mosaic 474 m ↔ our frames | REJECTED for pairing | GSD ratios 95x–1900x. Context only. |
| V6: SELENE TC full-res segment (~7.4 m) ↔ TMC-2 (5 m) | QUEUED (listing) | ~1.5x scale ratio, plausible. Segment file for the Vikram lon range; size TBD at listing. |
| V7: LRONAC ASP example pair | UNSCREENED | Site footprint not recovered from sidecars or web search. Not a candidate until screened. |

## Queued queries (exact parameters — run before any fetch)

- **Q1 (NAC second observation):** ODE product search, Moon, NAC
  EDR, box lat [-70.0742, -68.5014], lon [31.2285, 33.4600],
  exclude PRODUCT_ID `M1443025251`. Prefer an observation with
  incidence within ±20° of M1443025251's for a matchable pair;
  record any-illumination hits separately as sun-angle candidates.
  Fetch only the EDR frame(s) that overlap the box; size TBD.
- **Q2 (SELENE TC MAP pair):** DARTS SELENE archive search, LISM
  TC MAP (morning + evening), same site box as Q1. Fetch only
  MAP products whose footprints contain the box; size TBD.
- **Q3 (SELENE segment listing):** list
  `pdsimage.wr.usgs.gov/downloads/Kaguya_TC/` for the segment
  covering lon 31–34°; record its filename + size before fetching.
- **Q4 (S3 inventory, if aws CLI installed):** run the
  no-sign-request listing above and cross-check scene metadata
  for Vikram-box footprints.

## Caveats that travel with this inventory

- Bounding-box containment is NOT swath overlap. Every SCREENED
  verdict above still needs a pixel-level overlap check at crop
  time (diagonal swaths, map-projection differences).
- Sun angles are NOT in Chandrayaan PDS4 labels; for NAC/SELENE
  fetches, illumination metadata must be captured from the archive
  record at download time (the LROC index carries it; DARTS MAP
  records carry morning/evening illumination class).
- GSD ladder for pairing sense: NAC 1 m, OHRC 0.25 m, TMC-2 5 m,
  TC segment ~7.4 m, TC mosaic 474 m, IIRS 80 m. Pairs wider than
  ~30x have no on-project precedent of success; pairs beyond
  ~100x are rejected, not attempted.
- This inventory creates no verified landmarks and no pair
  assignments beyond the split doc. T4–T6 slots stay open until a
  queued query returns a screened product; otherwise the split
  doc's fallback rule applies.
