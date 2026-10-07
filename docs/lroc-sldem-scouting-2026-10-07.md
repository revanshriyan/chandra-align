# LROC NAC DTM / SLDEM Absolute-Control Scouting — 2026-10-07

Read-only research. No repo changes, no downloads of multi-GB files, nobody contacted.

## Verdict

**Feasible for the OHRC pair (medium effort). Partial/coarse-only for the TMC-2 pair (hard).**
Fastest path: start with OHRC + `NAC_DTM_VIKRAMSITE1` — a direct-hit 3 m/px LROC NAC DTM
with 1 m orthophotos covering the Shiv Shakti Point region. For TMC-2, no high-res
reference product was identified for the lon 19–35 / lat −60..−85 band; the fallback is
Barker's 20 m/px LOLA south-polar DEM (4× coarser than TMC-2).

**SLDEM2015 is NOT usable for either pair**: it covers only ±60° latitude (Barker et al.
2016; USGS product page). The OHRC site (−69.37°S) and essentially all of the TMC-2 band
(−60..−85°S) lie outside its coverage. (It is ~59 m/px anyway — far too coarse for
0.25 m/px OHRC.)

## 1. Candidate products

### 1A. OHRC pair — NAC_DTM_VIKRAMSITE1 (direct hit)

- **Product ID:** `NAC_DTM_VIKRAMSITE1` — "Chandrayaan-3 Landing Site (Pre-Landing) DTM"
- **Source:** LROC NAC stereo (M1442997156 L/R + M1443025251 L/R), ASU LROC team
- **Resolution:** 3 m/px DTM; derived orthophotos at **1 m** and 3 m
- **Coverage:** lat −70.07..−68.51, lon 31.31..33.37 (center −69.29, 32.34) — brackets
  the Shiv Shakti Point / Vikram site our OHRC pair covers
- **Geodetic quality:** relative linear error 1.92 m; registered to 63 LOLA profiles,
  RMS 1.85 m
- **Derived products:** DTM GeoTIFF, 1 m + 3 m orthophotos, shaded relief, color slope,
  confidence map
- **Catalog page:** https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_DTM_VIKRAMSITE1
- **PDS archive path pattern:** `LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/VIKRAMSITE1/`
  (host: `pds.lroc.im-ldi.com/data/…`; the `pds.mcp.nasa.gov` mirror returned 403 on
  directory listing — use the ASU host or the view_rdr page links)
- **Size:** exact VIKRAMSITE1 size not verified (listing blocked); typical NAC DTM
  GeoTIFFs run ~120–370 MB (e.g. FECUNPIT 121,761,751 B; FRESHMELT 371,180,099 B per a
  third-party fetch manifest). Expect a few hundred MB for DTM + orthophoto.
- **License:** PDS public domain; LROC team (Arizona State University) requests
  attribution. Standard NASA PDS terms — citable and redistributable with credit.
- **Corroboration:** a 2026 arXiv paper built a 0.30 m OHRC-stereo DEM of this exact
  site and geodetically anchored it *to this NAC DTM* — confirming the product is
  fit for this purpose. (Note: that paper's DEM is same-sensor-derived, so it is
  not *independent* control — the NAC DTM itself is.)

### 1B. TMC-2 pair — Barker LOLA south-polar DEMs (coarse fallback)

- **Product:** `LDEM_60S_20MPP_ADJ.TIF` (and siblings) — LOLA digital elevation model,
  60–90°S coverage, available at 10/20/30/40/60 m/px
- **Source:** Michael Barker / NASA GSFC PGDA (Planetary Geodynamics Data Archive)
- **Projection:** south polar stereographic, X/Y meters, MOON_ME frame, DE421 ephemeris;
  cloud-optimized GeoTIFFs
- **Page:** https://pgda.gsfc.nasa.gov/products/90
- **Citation:** Barker et al. 2023, Planet. Sci. J. 4, 183 (doi:10.3847/PSJ/acf3e1);
  data DOI 10.60903/gsfcpgda-lola-spole
- **Fit:** 20 m/px = 4× coarser than TMC-2 (5 m/px). Usable for coarse absolute
  control only (crater-rim-scale features), not landmark-level validation.
- **Size:** single COG tile, likely tens of MB — exact size not verified.
- **Also noted:** 5 m/px LOLA site DEMs exist for Artemis-candidate sites
  (Shackleton, Nobile, de Gerlache, Haworth, …) — **none** fall in our lon 19–35 band
  per the published site list.

### 1C. TMC-2 pair — NAC DTM coverage: UNVERIFIED

No named NAC DTM was found for lon 19–35°E, lat −60..−85°S in searchable sources.
The ASU archive holds 600+ site-specific NAC DTMs (~2–5 m posting) with a footprint
shapefile (`SHAPEFILE_NAC_DTMS`), but checking coverage of our specific band needs
the interactive LROC RDR search / QuickMap
(https://quickmap.lroc.asu.edu/ ; RDR search via https://lroc.im-ldi.com/data).
This is the open item before committing to the coarse LOLA fallback.

## 2. Registration path sketch

For each pair, the procedure is the same — transfer absolute coordinates through
features visible in both our imagery and the reference product:

1. **Overlap check.** Read the refined corner coordinates from our products' PDS4 XML
   labels; confirm they fall inside the reference product bounds (VIKRAMSITE1:
   lat −70.07..−68.51, lon 31.31..33.37; LDEM_60S: 60–90°S in south polar stereo).
2. **Feature transfer.** Identify ~10–15 common features (crater rims, boulders,
   ridge crests) in our image and in the reference orthophoto (1 m for VIKRAMSITE1;
   20 m hillshade for LOLA). Read each feature's absolute (lat, lon) from the
   reference georeferencing; record the corresponding pixel in our frame.
   These are the absolute GCPs.
3. **Adjudication.** For each of our 40 human landmarks, compare the pipeline's
   estimated position against the GCP-derived absolute position. Large systematic
   offsets implicate the pipeline; agreement with human picks but not GCPs
   implicates the human GT.
4. **Error budget (honest):** reference resolution dominates — 3 m DTM ≈ 12 OHRC px;
   20 m LOLA ≈ 4 TMC-2 px. Feature-identification adds ~1–2 reference px. So absolute
   control resolves disputes at the ~3–6 m (≈12–24 OHRC px) / ~20–40 m (≈4–8 TMC-2 px)
   level — enough to adjudicate the 5.65 px / 27 px GT disagreements, **not** enough
   to validate sub-pixel claims. Absolute control answers "are we right to ~10 px",
   not "are we right to 0.1 px".

Error sources, in order: reference-product posting (dominant) → feature
identification error → orthophoto/DEM co-registration (1.85 m RMS to LOLA for
VIKRAMSITE1) → projection/datum handling (polar stereo vs our frames) →
illumination mismatch between acquisitions.

## 3. Difficulty verdicts

| Pair | Verdict | Single biggest blocker |
|---|---|---|
| OHRC (Shiv Shakti) | **MEDIUM** | Confirming our exact pair footprint lies inside the VIKRAMSITE1 bounds (PDS4 corner check), and accepting the 12× resolution gap — control is coarse, not sub-pixel |
| TMC-2 fore/nadir | **HARD** | No high-res reference identified for lon 19–35, lat −60..−85; interactive catalog check still needed, likely ending at 20 m LOLA coarse control |

## 4. Recommended sequence

1. OHRC first: download VIKRAMSITE1 1 m orthophoto + DTM GeoTIFF (few hundred MB),
   verify footprint overlap from our PDS4 labels, pick 10–15 transfer features,
   compute absolute residuals for the 20 OHRC landmarks.
2. TMC-2: run the interactive LROC RDR/QuickMap coverage check for the band; if no
   NAC DTM, download `LDEM_60S_20MPP_ADJ.TIF` and do crater-rim-scale coarse control.
3. Write up as a methods note with the error budget stated upfront — the value is
   adjudicating pipeline-vs-human-GT, not claiming sub-pixel absolute accuracy.
