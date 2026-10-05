# IIRS first contact — overlap check stopped before matching

## Input and provenance

The supplied PRADAN product is `ch2_iir_nci_20260622T1324243428_d_img_d18`, a Chandrayaan-2 IIRS calibrated (`nci`) PDS4 observation from 2026-06-22 13:24:24Z to 13:32:16Z. Its reported footprint is latitude −81.907° to −58.032° and longitude 91.783° to 101.344°. The source archive MD5 is `abce53af664ae0278dec1f545a9625ac`; the supplied provenance record is [iirs_provenance.json](iirs_provenance.json).

The three supplied arrays were verified before analysis: each is float32, shape `(8904, 250)`, with 2,226,000 finite samples. Their measured min/max/mean are b001 `0 / 416673.34375 / 825.5132`, b128 `0 / 951.0065 / 63.4240`, and b256 `0 / 53512.4766 / 216.9211`, matching the supplied reference statistics within float precision.

These are three pre-extracted candidate bands (direct BSQ read, no resampling) supplied because the full 2.28 GB cube could not be transferred to this machine. The repo's real .qub reader was NOT engaged on the full product. The full cube is retained by the QA engineer for any future full-cube rerun.

## Band choice and planned normalization

If a valid overlap is established, band 1 at 712.3 nm is the planned first candidate because it is nearest TMC-2's visible panchromatic response and therefore has the strongest structural-similarity prior. The 2852.6 nm and 5009.7 nm candidates are retained for a later controlled comparison. No band was actually matched in this run.

The intended image normalization is a 2nd–98th percentile stretch followed by the repository's IIRS CLAHE preprocessing. No normalization was applied because no valid overlapping TMC-2 crop was located. No IIRS ground scale was assigned or inferred.

## Overlap check and stop reason

The IIRS footprint bounds were compared with the actual fore and nadir product footprints, not only their broad coordinate bounds. The TMC-2 PDS4 labels identify a south-polar stereographic projection and give the four product corner latitude/longitude values. Those corners were projected with a common spherical lunar polar-stereographic transform (lunar radius 1737.4 km) and used to form each strip's footprint polygon. The IIRS footprint bounds were projected in the same plane and treated as a polygon for this coarse overlap check.

The projected polygons have zero intersection area with both TMC-2 products:

- Fore (`ch2_tmc_ncf_20231101T0125121344_d_img_d18`): projected bounds approximately x 17.56–333.43 km, y −26.73–914.57 km.
- Nadir (`ch2_tmc_ncn_20231101T0125121377_d_img_d18`): projected bounds approximately x 5.06–318.96 km, y −68.47–871.28 km.
- IIRS reported footprint bounds: projected bounds approximately x 241.01–994.85 km, y −195.78 to −7.65 km.

The geographic bounding boxes alone overlap, but the reconstructed strip polygons do not. The PDS labels expose corner geometry rather than a per-pixel georeferencing model, so a precise crop cannot be assigned from these files. Following the protocol, no crop was guessed and no non-overlapping IIRS→TMC-2 pair was sent to `match_pair_hf`.

## Attempt status

No matcher attempt was made. Therefore inliers, quadrant coverage, in-sample or held-out RMSE/MAE, verdicts, and rejection-reason counts are not measured. [table_issue12_iirs.csv](../results/table_issue12_iirs.csv) contains the requested result schema and no fabricated attempt rows. Issue #12's acceptance criteria are not met; obtain per-pixel georeferencing or a verified overlapping crop before rerunning this phase.
