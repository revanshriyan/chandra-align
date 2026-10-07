# LROC Absolute-Control Adjudication — Status 2026-10-07

## Goal
Determine whether the 5.65px (OHRC) / 27.35px (TMC-2) pipeline-vs-human-GT
disagreements are pipeline error or human-GT error, using LROC NAC DTM
VIKRAMSITE1 as absolute reference.

## Confirmed
- VIKRAMSITE1 DTM downloaded and verified: 7668x15895 px, float32, 3m/px,
  polar stereographic, bounds lat -70.07..-68.51 / lon 31.31..33.37.
- OHRC product ch2_ohr_nrp_20240425T1209509264_d_img_d18 corners from PDS4
  label: UL(-69.099842, 32.179683) UR(-69.104632, 32.473448)
  LL(-69.930319, 32.062357) LR(-69.935282, 32.366871).
- **Overlap: CONFIRMED.** OHRC footprint lies fully inside VIKRAMSITE1 bounds.
- OHRC 6000x6000 crop extracted around lat -69.3, lon 32.3 (line 22447,
  sample 4914 via linear corner interpolation). Valid data (range 7-255).
- DTM central 500x500 window read (99.8% valid); hillshade rendered
  (az 315, el 45).

## Attempted
Direct SIFT cross-modal registration: OHRC downsampled 12x to 3m/px vs DTM
hillshade. Result: 675 vs 54 keypoints, 7 Lowe matches, 3 RANSAC inliers,
degenerate transform. **Cross-modal SIFT is insufficient** for optical vs
shaded-relief at this scale — expected; hillshade lacks distinctive texture.

## Next steps
1. Try phase-congruency front-end (chandra_align/xmodal/phase_congruency.py,
   Phase 14) which was built for modality gaps — wire it for OHRC vs hillshade.
2. Vary hillshade sun angle to match OHRC acquisition illumination (check PDS4
   solar incidence angle) — shadow alignment is critical for keypoint matching.
3. If cross-modal still fails: fall back to manual feature transfer (~10-15
   crater-rim tie points) for coarse (~12-24px) adjudication — sufficient to
   resolve the 5.65px question, per the scouting error budget.
4. The 1m orthophoto (not found as GeoTIFF) would make this far easier;
   revisit via interactive LROC RDR catalog if needed.

## Files
- /home/hatch/workspace/lroc_data/NAC_DTM_VIKRAMSITE1.TIF (466MB, verified)
- /tmp/ohrc_crop.npy (6000x6000 OHRC, lat -69.3/lon 32.3 area)
- /tmp/dtm_hillshade.npy (500x500 hillshade)
