# L1 Rescue Telemetry/Export Consistency Audit — 2026-10-08

## Summary

**Finding: CRITICAL inconsistency** — When the L1 rescue verdict is adopted,
the exported raster uses the NORMAL (worse) deformation field, not the rescue
refit's field. The verdict claims e.g. 0.291px RMSE, but the exported image
is warped with the field that scored worse.

**Status:** Fixed (additive, fail-closed, no gate changes).

## The Problem

When `maybe_rescue_and_refit` triggers and the rescue refit reaches
SUCCESS_SUBPIXEL while the normal path does not, `app.py` adopts the rescue
verdict (lines 1096-1118):

- `status_code` → "SUCCESS_SUBPIXEL" ✓
- `rmse_px` → rescue RMSE (e.g., 0.291px) ✓
- `inlier_cnt` → rescue_n (e.g., 1445) ✓
- `quadrant_metrics` → rescue quadrants ✓
- `deform_field_info["verdict_source"]` → "l1_rescue_second_opinion" ✓

**BUT** the exported artifacts were NOT updated:

1. **Dense remap export** (app.py:1202-1206): Uses
   `deform_field_info.get("M_hook")` and `deform_field_info.get("field")`,
   which are from the NORMAL stage (app.py:804: `deform_field_info =
   dict(stage_info)`). The rescue refit's field (`_refit["field"]`, fit on
   the 1445 admitted points) was never propagated. The exported raster
   claims 0.291px but is warped with the worse normal field.

2. **Exported affine** (app.py:1189, 1292): `affine_matrix` and `H` are from
   the normal path. (The M_hook is identical in normal and rescue — the
   rescue refit reuses the same Ma — so this is consistent, but the FIELD
   differs.)

3. **Point sets** (app.py:1009, 1235): `inlier_ref`/`inlier_sec` contain
   normal-path points (e.g., 17 points), but `inlier_cnt` reports 1445.
   The `deformation_vectors` and GCP CSV export are computed from the
   17 normal points, not the 1445 rescue-admitted points.

## The Fix

**In the rescue block** (app.py, after the gate check at line ~868):
- Store the refit field and M_hook in the rescue info:
  `_rescue["refit_field"] = _refit["field"]`
  `_rescue["refit_M_hook"] = _refit["M_hook"]`
- These are EXCLUDED from JSON telemetry by `_jsonable_deform_field_info`
  (added to the exclusion list), but available for the export path.

**In the adoption block** (app.py, lines 1096-1118):
- After updating the verdict fields, also update:
  `deform_field_info["field"] = _ri["refit_field"]`
  `deform_field_info["M_hook"] = _ri["refit_M_hook"]`
- The dense remap export (line 1202) now uses the rescue field automatically.
- Fail-closed: if refit_field is missing/invalid, keep the normal field.

**Point sets:** The `inlier_ref`/`inlier_sec` are NOT updated to the rescue
admitted points. Rationale: the rescue admitted points are in RAW
coordinates, while the pipeline's `pts_ref`/`pts_sec` were overwritten with
field-corrected points at line 803. Mixing coordinate frames would be
worse than the current inconsistency. The GCP export documents the normal
path; the verdict documents the rescue path. This is noted in telemetry
via `verdict_source`.

## Verification

- [x] Rescue adoption still triggers correctly (verdict_source set) — logic
      unchanged, only adds field swap after existing verdict updates
- [x] Exported raster uses rescue field when adopted — adoption block now
      copies `_refit["field"]`/`_refit["M_hook"]` into `deform_field_info`,
      which the dense remap export (line 1202) reads; fail-closed on any
      error (keeps normal field)
- [x] Flag-off path bit-identical — rescue block only runs when
      `CHANDRA_DEFORM_FIELD=1` and stage applied; adoption only when
      rescue_gate_code is SUCCESS
- [x] JSON telemetry excludes raw field arrays — `_jsonable_deform_field_info`
      strips `refit_field`/`refit_M_hook` at top level and inside nested
      `rescue_info`; verified with isolated test (no gradio needed)
- [x] No gate threshold changes — validate_registration_gate untouched
- [x] Existing deform-field tests pass (17 passed, 7 skipped)
- [ ] Full 12-pair live verification pending (parent agent's audit subagent)

## Constraints Honored

- No gate threshold modifications
- Default pipeline path untouched (flag-off bit-identical)
- Additive only: new keys in rescue_info, new lines in adoption block
- Fail-closed: any error keeps the normal field
- No AI watermarks
