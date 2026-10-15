"""LROC absolute-truth adjudication THROUGH the deployed _align_core pipeline.

This is the pipeline-driven counterpart to scripts/lroc_field_adjudication.py
(worker A21, standalone). It closes the audit critique that "the LROC result
is one site and a standalone script; it did not go through the deployed
pipeline."

Method:
- SAME inputs as the standalone: LROC 1m/3m ortho windows (raw binary reads),
  OHRC template resized to 532/178 px. Raw float32 is passed to _align_core;
  the pipeline does its own percentile-normalize + CLAHE (the deployed path).
- SAME strict no-leakage spatial-split protocol: correspondences come from
  app._align_core (not the standalone SIFT recipe); they are split into
  fit (coord<=lo+40%) / check (coord>=lo+60%) on the SEC (window) coords,
  x- and y-axes separately.
- Baseline bound: LMEDS fit on fit-set only -> RMSE on check-set -> meters.
- Field bound: RANSAC M_hook on fit-set only -> apply_deform_field_stage on
  fit-set only -> field evaluated on the unseen check-set -> meters.
  The pipeline's INTERNAL field is deliberately NOT used for scoring: it was
  fit on all hook inliers (including check-region points), which would leak.
  Re-fitting the field on the fit-set only is the honest, comparable arm.
- CHANDRA_DEFORM_FIELD=1 is set so this is the field-enabled deployed path.

What is being tested: whether the deployed pipeline's matching produces
correspondences that support absolute-accuracy bounds comparable to the
standalone's published 1.7-2.3m (1m grid) / 2.0-2.5m (3m grid).

Read-only evaluation: nothing here modifies the pipeline, gates, or defaults.
Deterministic: cv2.setRNGSeed(7) before pipeline call and every estimator.
Fail-closed: pipeline ValueError -> DEGENERATE row, no crash.
"""
import numpy as np, cv2, csv, sys, os, time

SEED = 7
REPO = '/home/hatch/workspace/chandra-align'
LROC1M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_100CM.IMG'
LROC3M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_3M.IMG'
CROP = REPO + '/data/benchmark_crops/ohrc_01_reference.png'
OUT_CSV = REPO + '/results/table_lroc_pipeline.csv'

os.environ['CHANDRA_DEFORM_FIELD'] = '1'
sys.path.insert(0, REPO)
# app.py imports gradio for its UI; mock it since we only need _align_core
# (gradio is never touched outside UI-building functions).
from unittest.mock import MagicMock
# app.py pulls gradio/spaces only for its UI, and chandra_align.visualization
# imports matplotlib only for diagnostic plots _align_core never calls.
# Mock all three so the import works on this matplotlib-less env.
for _mod in ('gradio', 'spaces', 'matplotlib', 'matplotlib.pyplot',
             'matplotlib.gridspec', 'matplotlib.colors', 'matplotlib.ticker',
             'matplotlib.transforms', 'matplotlib._path', 'matplotlib.cbook'):
    sys.modules[_mod] = MagicMock()
import app as app_module
from chandra_align.deform_field import (
    apply_deform_field_stage, _eval_residual_field, _apply_affine,
    _jacobian_min_det)

# Standalone published bounds (from lroc_field_adjudication.py A21 runs)
STANDALONE = {
    '1m': dict(n_corr=164, n_inl=96, hox=2.58, hoy=3.46,
               field_hox=1.7, field_hoy=2.3),   # field tightened bounds
    '3m': dict(n_corr=46, n_inl=46, hox=7.06, hoy=6.39,
               field_hox=2.0, field_hoy=2.5),
}


def read_window(path, off, samples, cx, cy, hw):
    win = np.empty((2 * hw, 2 * hw), dtype='<u2')
    with open(path, 'rb') as fh:
        for i, r in enumerate(range(cy - hw, cy + hw)):
            fh.seek(off + (r * samples + (cx - hw)) * 2)
            win[i] = np.fromfile(fh, dtype='<u2', count=2 * hw)
    return win.astype(np.float32)


def decomp(M):
    a11, a21 = M[0, 0], M[1, 0]
    return (float(np.sqrt(a11 ** 2 + a21 ** 2)),
            float(np.degrees(np.arctan2(a21, a11))))


def spatial_split(p2):
    out = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        v = p2[:, axis]
        lo, hi = v.min(), v.max()
        fi = np.where(v <= lo + 0.40 * (hi - lo))[0]
        ci = np.where(v >= lo + 0.60 * (hi - lo))[0]
        out[name] = (fi, ci)
    return out


def heldout_lmeds_m(p1, p2, fi, ci, m_per_px):
    """LMEDS on fit idx, RMSE on check idx, in meters."""
    if len(fi) < 6 or len(ci) < 1:
        return float('nan'), len(fi), len(ci)
    cv2.setRNGSeed(SEED)
    Mf, _ = cv2.estimateAffinePartial2D(p1[fi], p2[fi], method=cv2.LMEDS)
    if Mf is None:
        return float('nan'), len(fi), len(ci)
    pred = (Mf @ np.hstack([p1[ci], np.ones((len(ci), 1))]).T).T
    chk = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
    return chk * m_per_px, len(fi), len(ci)


def run_grid(name, tsz, m_per_px, cx, cy, hw, off, lines, samples):
    print('=== PIPELINE GRID %s: template %dx%d, window %dpx at (%d,%d), %.1f m/px ==='
          % (name, tsz, tsz, 2 * hw, cx, cy, m_per_px))
    ohrc = cv2.imread(CROP, cv2.IMREAD_GRAYSCALE)
    assert ohrc is not None and ohrc.shape == (2048, 2048)
    # Raw float32: the pipeline does its own normalize + CLAHE (deployed path).
    tmpl = cv2.resize(ohrc, (tsz, tsz), interpolation=cv2.INTER_AREA).astype(np.float32)
    win = read_window(LROC1M if name == '1m' else LROC3M, off, samples, cx, cy, hw)

    cv2.setRNGSeed(SEED)
    t0 = time.time()
    # NOTE on pixel_scale_m: the inputs are PRE-REGISTERED at a common GSD
    # (template resized to m_per_px, window at m_per_px -- same as standalone).
    # The pipeline's GSD-aware resampling would otherwise downsample the
    # window based on sensor-database metadata ('LROC' defaults to 0.25 m/px,
    # but these are ortho products at 1.0/3.0 m/px). Passing 0.25 declares
    # matched GSDs so matching runs at full resolution; meters are computed
    # in the protocol below using the TRUE m_per_px, not this value.
    try:
        out = app_module._align_core(
            tmpl, win, pixel_scale_m=0.25,
            reference_sensor_name='OHRC', secondary_sensor_name='LROC')
    except ValueError as exc:
        print('%s PIPELINE DEGENERATE: %s' % (name, exc))
        return dict(grid=name, pipeline_status='DEGENERATE_FAILURE',
                    pipeline_error=str(exc)[:120])
    dt = time.time() - t0

    p1 = np.asarray(out['pts_ref_inliers'], dtype=np.float64)  # template coords
    p2 = np.asarray(out['pts_sec_inliers'], dtype=np.float64)  # window coords
    n_inl = len(p1)
    status = out.get('judge_metrics', {}).get('status_code', '?')
    rmse_in = float(out.get('rmse_px', float('nan')))
    df = out.get('judge_metrics', {}).get('deform_field_stage', {})
    print('%s PIPELINE: status=%s inliers=%d rmse=%.4fpx field_applied=%s (%.1fs)'
          % (name, status, n_inl, rmse_in, df.get('applied'), dt))

    row = dict(grid=name, pipeline_status=status,
               n_corr=int(out.get('total_matches', -1)), n_inliers=n_inl,
               pipeline_rmse_px=rmse_in,
               pipeline_field_applied=bool(df.get('applied')),
               pipeline_field_reason=df.get('reason'),
               pipeline_field_heldout_px=df.get('heldout_rmse_px'))
    if n_inl < 8:
        row['protocol'] = 'insufficient_inliers'
        return row

    # ---- identical strict protocol on pipeline correspondences ----
    splits = spatial_split(p2)
    hox, nfx, ncx = heldout_lmeds_m(p1, p2, *splits['x'], m_per_px)
    hoy, nfy, ncy = heldout_lmeds_m(p1, p2, *splits['y'], m_per_px)
    row.update(base_hox_m=hox, base_hoy_m=hoy,
               base_hox_fit=nfx, base_hox_check=ncx,
               base_hoy_fit=nfy, base_hoy_check=ncy)
    print('%s pipeline-correspondence BASELINE bound: x=%.2fm(%d/%d) y=%.2fm(%d/%d)'
          % (name, hox, nfx, ncx, hoy, nfy, ncy))

    for sname in ('x', 'y'):
        fi, ci = splits[sname]
        key = 'field_' + sname
        if len(fi) < 6 or len(ci) < 1:
            row[key + '_applied'] = False
            row[key + '_reason'] = 'split_too_small'
            continue
        cv2.setRNGSeed(SEED)
        Mh, _ = cv2.estimateAffinePartial2D(
            p1[fi], p2[fi], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if Mh is None:
            row[key + '_applied'] = False
            row[key + '_reason'] = 'hook_ransac_none'
            continue
        info = apply_deform_field_stage(p1[fi], p2[fi], Mh, seed=SEED)
        row[key + '_applied'] = bool(info['applied'])
        row[key + '_reason'] = info['reason']
        row[key + '_lambda'] = info.get('lambda_chosen')
        row[key + '_n_fit'] = len(fi)
        row[key + '_n_check'] = len(ci)
        row[key + '_min_jac_det'] = info.get('min_jacobian_det')
        if not info['applied']:
            print('%s pipeline field-%s: DECLINED (%s)'
                  % (name, sname, info['reason']))
            continue
        field = info['field']
        Mh_arr = np.asarray(Mh, dtype=np.float64)
        d_check = _eval_residual_field(field, p1[ci])
        pred = _apply_affine(Mh_arr, p1[ci]) + d_check
        chk_px = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
        chk_m = chk_px * m_per_px
        row[key + '_check_rmse_px'] = chk_px
        row[key + '_check_rmse_m'] = chk_m
        row[key + '_min_jac_det_check'] = float(
            _jacobian_min_det(Mh_arr, field, p1[ci]))
        base_ho = hox if sname == 'x' else hoy
        row[key + '_bound_delta_m'] = float(base_ho - chk_m)
        print('%s pipeline field-%s: APPLIED λ=%s check %.3fpx=%.3fm '
              'vs base %.2fm | Δ=%+.3fm | minJac %.3f (check %.3f)'
              % (name, sname, info.get('lambda_chosen'), chk_px, chk_m,
                 base_ho, base_ho - chk_m, info.get('min_jacobian_det'),
                 row[key + '_min_jac_det_check']))
    return row


def main():
    t0 = time.time()
    rows = []
    rows.append(run_grid('1m', 532, 1.0, 10818, 16081, 450, 46006, 47683, 23003))
    rows.append(run_grid('3m', 178, 3.0, 3608, 5360, 180, 15336, 15895, 7668))

    with open(OUT_CSV, 'w', newline='') as fh:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT_CSV, 'in %.1fs' % (time.time() - t0))

    # ---- comparison table ----
    print()
    print('=== STANDALONE vs PIPELINE (held-out absolute bounds, meters) ===')
    print('%-4s %-8s %12s %12s %12s %12s' % (
        'grid', 'metric', 'standalone', 'pipeline', 'pipe_field', 'delta'))
    for r in rows:
        g = r['grid']
        s = STANDALONE[g]
        for ax in ('x', 'y'):
            base_p = r.get('base_ho%s_m' % ax, float('nan'))
            fld_p = r.get('field_%s_check_rmse_m' % ax, float('nan'))
            s_base = s['ho' + ax]
            s_fld = s['field_ho' + ax]
            d = (fld_p - s_fld) if fld_p == fld_p else float('nan')
            print('%-4s %-8s %12.2f %12.2f %12.2f %12.2f   (standalone field %.1f-%.1f)' % (
                g, 'bound_' + ax, s_base, base_p, fld_p, d,
                s['field_hox'], s['field_hoy']))


if __name__ == '__main__':
    main()
