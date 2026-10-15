"""LROC pipeline adjudication: split-robustness repeats (audit item #6).

The published bounds rest on single tiny spatial splits (fit/check =
13/10, 12/13, 21/34, 25/22). This script re-uses the pipeline
correspondences from scripts/lroc_pipeline_adjudication.py but evaluates
the strict no-leakage protocol under 5 different spatial partitions per
grid, then reports the SPREAD (min/max/mean) of the held-out bounds.

Splits (all spatial, all with a no-man's-land gap to preserve no-leakage):
  seed 7:  fit <= lo+40%, check >= lo+60%   (the published partition)
  seed 11: fit <= lo+30%, check >= lo+70%
  seed 13: fit <= lo+50%, check >= lo+70%
  seed 17: fit >= hi-40%, check <= hi-60%    (mirrored: fit on far side)
  seed 23: fit <= lo+35%, check >= lo+65%

For each (grid, seed, axis): LMEDS baseline on fit -> check RMSE in meters;
then RANSAC hook + apply_deform_field_stage on fit only -> field check RMSE.
Deterministic per seed (cv2.setRNGSeed(seed)).

Output: results/table_lroc_pipeline_splits.csv
"""
import numpy as np, cv2, csv, sys, os

SEED_BASE = 7
REPO = '/home/hatch/workspace/chandra-align'
LROC1M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_100CM.IMG'
LROC3M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_3M.IMG'
CROP = REPO + '/data/benchmark_crops/ohrc_01_reference.png'
OUT_CSV = REPO + '/results/table_lroc_pipeline_splits.csv'

os.environ['CHANDRA_DEFORM_FIELD'] = '1'
sys.path.insert(0, REPO)
from unittest.mock import MagicMock
for _mod in ('gradio', 'spaces', 'matplotlib', 'matplotlib.pyplot',
             'matplotlib.gridspec', 'matplotlib.colors', 'matplotlib.ticker',
             'matplotlib.transforms', 'matplotlib._path', 'matplotlib.cbook'):
    sys.modules[_mod] = MagicMock()
import app as app_module
from chandra_align.deform_field import (
    apply_deform_field_stage, _eval_residual_field, _apply_affine,
    _jacobian_min_det)

# (seed, fit_lo_frac, fit_hi_frac, check_lo_frac, check_hi_frac, mirrored)
# fit  = coords in [lo + fit_lo_frac*span, lo + fit_hi_frac*span]
# check = coords in [lo + check_lo_frac*span, lo + check_hi_frac*span]
SPLITS = [
    (7,  0.00, 0.40, 0.60, 1.00, False),   # published partition
    (11, 0.00, 0.30, 0.70, 1.00, False),
    (13, 0.00, 0.50, 0.70, 1.00, False),
    (17, 0.60, 1.00, 0.00, 0.40, True),    # mirrored
    (23, 0.00, 0.35, 0.65, 1.00, False),
]


def read_window(path, off, samples, cx, cy, hw):
    win = np.empty((2 * hw, 2 * hw), dtype='<u2')
    with open(path, 'rb') as fh:
        for i, r in enumerate(range(cy - hw, cy + hw)):
            fh.seek(off + (r * samples + (cx - hw)) * 2)
            win[i] = np.fromfile(fh, dtype='<u2', count=2 * hw)
    return win.astype(np.float32)


def get_correspondences(name, tsz, m_per_px, cx, cy, hw, off, samples):
    """Run the deployed pipeline once; return (p1, p2) inlier correspondences."""
    ohrc = cv2.imread(CROP, cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.resize(ohrc, (tsz, tsz), interpolation=cv2.INTER_AREA).astype(np.float32)
    win = read_window(LROC1M if name == '1m' else LROC3M, off, samples, cx, cy, hw)
    sec_sensor = 'LROC_ORTHO_1M' if name == '1m' else 'LROC_ORTHO_3M'
    cv2.setRNGSeed(SEED_BASE)
    out = app_module._align_core(
        tmpl, win, pixel_scale_m=m_per_px,
        reference_sensor_name='OHRC', secondary_sensor_name=sec_sensor)
    p1 = np.asarray(out['pts_ref_inliers'], dtype=np.float64)
    p2 = np.asarray(out['pts_sec_inliers'], dtype=np.float64)
    status = out.get('judge_metrics', {}).get('status_code', '?')
    print('%s: pipeline status=%s inliers=%d' % (name, status, len(p1)), flush=True)
    return p1, p2


def spatial_partition(v, flo, fhi, clo, chi):
    lo, hi = v.min(), v.max()
    span = hi - lo
    fi = np.where((v >= lo + flo * span) & (v <= lo + fhi * span))[0]
    ci = np.where((v >= lo + clo * span) & (v <= lo + chi * span))[0]
    return fi, ci


def run_split(p1, p2, seed, flo, fhi, clo, chi, mirrored, m_per_px, axis):
    """One (seed, axis) evaluation. Returns dict with base + field bounds in meters."""
    cv2.setRNGSeed(seed)
    v = p2[:, axis]
    fi, ci = spatial_partition(v, flo, fhi, clo, chi)
    row = dict(seed=seed, axis='xy'[axis], mirrored=mirrored,
               n_fit=len(fi), n_check=len(ci))
    if len(fi) < 6 or len(ci) < 1:
        row['base_m'] = float('nan')
        row['field_m'] = float('nan')
        row['note'] = 'split_too_small'
        return row
    # baseline: LMEDS on fit, score on check
    Mf, _ = cv2.estimateAffinePartial2D(p1[fi], p2[fi], method=cv2.LMEDS)
    if Mf is None:
        row['base_m'] = float('nan')
    else:
        pred = (Mf @ np.hstack([p1[ci], np.ones((len(ci), 1))]).T).T
        chk = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
        row['base_m'] = chk * m_per_px
    row['field_applied'] = False
    row['field_m'] = float('nan')
    # field: RANSAC hook + deform field on fit only, score on check
    Mh, _ = cv2.estimateAffinePartial2D(
        p1[fi], p2[fi], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Mh is None:
        row['note'] = 'hook_ransac_none'
        return row
    info = apply_deform_field_stage(p1[fi], p2[fi], Mh, seed=seed)
    row['field_applied'] = bool(info['applied'])
    row['field_reason'] = info['reason']
    if not info['applied']:
        return row
    field = info['field']
    Mh_arr = np.asarray(Mh, dtype=np.float64)
    d_check = _eval_residual_field(field, p1[ci])
    pred = _apply_affine(Mh_arr, p1[ci]) + d_check
    chk_px = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
    row['field_m'] = chk_px * m_per_px
    row['min_jac'] = float(_jacobian_min_det(Mh_arr, field, p1[ci]))
    return row


def main():
    grids = [
        ('1m', 532, 1.0, 10818, 16081, 450, 46006, 23003),
        ('3m', 178, 3.0, 3608, 5360, 180, 15336, 7668),
    ]
    rows = []
    for name, tsz, m_per_px, cx, cy, hw, off, samples in grids:
        p1, p2 = get_correspondences(name, tsz, m_per_px, cx, cy, hw, off, samples)
        if len(p1) < 8:
            print('%s: insufficient inliers, skipping' % name)
            continue
        for seed, flo, fhi, clo, chi, mirrored in SPLITS:
            for axis in (0, 1):
                r = run_split(p1, p2, seed, flo, fhi, clo, chi,
                              mirrored, m_per_px, axis)
                r['grid'] = name
                rows.append(r)
                print('  %s seed=%d axis=%s fit/check=%d/%d base=%s field=%s' % (
                    name, seed, r['axis'], r['n_fit'], r['n_check'],
                    ('%.2f' % r['base_m']) if r['base_m'] == r['base_m'] else 'nan',
                    ('%.2f' % r['field_m']) if r['field_m'] == r['field_m'] else 'nan'),
                    flush=True)

    with open(OUT_CSV, 'w', newline='') as fh:
        keys = ['grid', 'seed', 'axis', 'mirrored', 'n_fit', 'n_check',
                'base_m', 'field_applied', 'field_reason', 'field_m', 'min_jac', 'note']
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT_CSV)

    # ---- spread summary ----
    print()
    print('=== BOUND SPREAD ACROSS 5 SPLITS (meters) ===')
    for name in ('1m', '3m'):
        for axis in ('x', 'y'):
            vals = [r['field_m'] for r in rows
                    if r['grid'] == name and r['axis'] == axis
                    and r['field_m'] == r['field_m']]
            if vals:
                print('  %s-%s field: n=%d min=%.2f max=%.2f mean=%.2f' % (
                    name, axis, len(vals), min(vals), max(vals),
                    sum(vals) / len(vals)))
            bvals = [r['base_m'] for r in rows
                     if r['grid'] == name and r['axis'] == axis
                     and r['base_m'] == r['base_m']]
            if bvals:
                print('  %s-%s base:  n=%d min=%.2f max=%.2f mean=%.2f' % (
                    name, axis, len(bvals), min(bvals), max(bvals),
                    sum(bvals) / len(bvals)))


if __name__ == '__main__':
    main()
