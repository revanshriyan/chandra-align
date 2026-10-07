"""LROC absolute-truth adjudication WITH the deform-field stage (worker A21).

Reproduces the 1m and 3m LROC adjudication baselines exactly (A10's recipe:
SIFT nfeatures=8000, CLAHE, Lowe 0.75, partial-affine RANSAC 3.0px, seed 7),
then applies the deform-field stage as a library to the adjudication's
correspondences.

NOTE on method: the LROC adjudication is a standalone SIFT+RANSAC script, NOT
app.py::_align_core. Running _align_core on the LROC windows would be a
different experiment (different matching/balancing/gating) and would not be
comparable to the published baseline. The stage is therefore applied as a
library -- the identical mathematical operation the pipeline hook performs
(hook RANSAC inliers -> TPS residual field -> corrected points).

Pre-registered SUCCESS: the independent held-out ABSOLUTE bound (meters, from
the published spatial x/y split -- NOT in-sample, NOT the stage's internal
held-out) tightens on a grid, with no folding (min Jacobian det >= 0.5), no
transform inconsistency (dscale <= 0.002, drot <= 0.1 deg vs baseline), and
zero verdict downgrades. Inlier-RMSE improvement with a WORSENED absolute
bound is a NEGATIVE (overfit), per the A10 trap.

Honest protocol (no leakage): the field is fit on the FIT portion of the
published spatial split ONLY; the CHECK portion is never seen by the field.
The stage's internal lambda selection runs its own stride split WITHIN the
fit set. M_hook is fit on all inliers (same information the baseline's M
uses) for the agreement check.

Deterministic: cv2.setRNGSeed(7) before every randomized estimator call.
Fail-closed: RANSAC returning None -> DEGENERATE row, no crash.
"""
import numpy as np, cv2, csv, sys, os, time

SEED = 7
REPO = '/home/hatch/workspace/chandra-align'
LROC1M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_100CM.IMG'
LROC3M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_3M.IMG'
CROP = REPO + '/data/benchmark_crops/ohrc_01_reference.png'
sys.path.insert(0, REPO)
from chandra_align.metrics.quadrant import compute_quadrant_metrics, validate_registration_gate
from chandra_align.deform_field import (
    apply_deform_field_stage, _eval_residual_field, _apply_affine,
    _jacobian_min_det)

OUT_CSV = REPO + '/results/table_lroc_field.csv'


def read_window(path, off, samples, cx, cy, hw):
    win = np.empty((2 * hw, 2 * hw), dtype='<u2')
    with open(path, 'rb') as fh:
        for i, r in enumerate(range(cy - hw, cy + hw)):
            fh.seek(off + (r * samples + (cx - hw)) * 2)
            win[i] = np.fromfile(fh, dtype='<u2', count=2 * hw)
    return win.astype(np.float32)


def prep(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    u = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u)


def decomp(M):
    a11, a21 = M[0, 0], M[1, 0]
    return (float(np.sqrt(a11 ** 2 + a21 ** 2)),
            float(np.degrees(np.arctan2(a21, a11))),
            float(M[0, 2]), float(M[1, 2]))


def ransac_fit(p1, p2):
    cv2.setRNGSeed(SEED)
    M, inl = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M, (None if inl is None else inl.ravel().astype(bool))


def fit_rmse(M, p1, p2, inl):
    r = np.linalg.norm((M @ np.hstack([p1, np.ones((len(p1), 1))]).T).T - p2, axis=1)
    return float(np.sqrt((r[inl] ** 2).mean())), r


def sift_pairs(t8, o8):
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(t8, None)
    k2, d2 = sift.detectAndCompute(o8, None)
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, nn in raw if m.distance < 0.75 * nn.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2, len(k1), len(k2)


def spatial_split(p2inl):
    """Published x/y split: fit on coord<=lo+40%, check on coord>=lo+60%.

    Returns dict axis -> (fit_local_idx, check_local_idx) into the inlier arrays.
    """
    out = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        v = p2inl[:, axis]
        lo, hi = v.min(), v.max()
        fi = np.where(v <= lo + 0.40 * (hi - lo))[0]
        ci = np.where(v >= lo + 0.60 * (hi - lo))[0]
        out[name] = (fi, ci)
    return out


def heldout_lmeds_split(p1i, p2i, fi, ci, m_per_px):
    """LMEDS on fit idx, evaluate on check idx. Returns (rmse_m, n_fit, n_check)."""
    if len(fi) < 6 or len(ci) < 1:
        return float('nan'), len(fi), len(ci)
    cv2.setRNGSeed(SEED)
    Mf, _ = cv2.estimateAffinePartial2D(p1i[fi], p2i[fi], method=cv2.LMEDS)
    if Mf is None:
        return float('nan'), len(fi), len(ci)
    pred = (Mf @ np.hstack([p1i[ci], np.ones((len(ci), 1))]).T).T
    chk = float(np.sqrt((np.linalg.norm(pred - p2i[ci], axis=1) ** 2).mean()))
    return chk * m_per_px, len(fi), len(ci)


def gate_tier(rmse, n_inl, ent, counts):
    _, tier = validate_registration_gate(
        rmse, n_inl, 8, ent,
        {'Q1': counts[0], 'Q2': counts[1], 'Q3': counts[2], 'Q4': counts[3]})
    return tier


def quad_stats(p2inl, rinl, hw):
    quads, ent = compute_quadrant_metrics(p2inl, rinl, (2 * hw, 2 * hw))
    counts = [int(quads[k]['inlier_count']) for k in ('Q1', 'Q2', 'Q3', 'Q4')]
    return counts, float(ent)


def run_grid(name, tsz, m_per_px, cx, cy, hw, off, lines, samples,
             do_tmpl_match, base_expect):
    print('=== GRID %s: template %dx%d @ %.1fm/px, window %dpx at (%d,%d) ==='
          % (name, tsz, tsz, m_per_px, 2 * hw, cx, cy))
    ohrc = cv2.imread(CROP, cv2.IMREAD_GRAYSCALE)
    assert ohrc is not None and ohrc.shape == (2048, 2048)
    tmpl = cv2.resize(ohrc, (tsz, tsz), interpolation=cv2.INTER_AREA).astype(np.float32)
    t8 = prep(tmpl)

    if do_tmpl_match:
        sx, sy, sr = 3521, 5276, 300
        search = read_window(LROC3M, off, samples, sx, sy, sr)
        s8 = prep(search)
        res = cv2.matchTemplate(s8, t8, cv2.TM_CCOEFF_NORMED)
        _, mx, _, mloc = cv2.minMaxLoc(res)
        cx, cy = sx - sr + mloc[0] + tsz // 2, sy - sr + mloc[1] + tsz // 2
        print('NCC peak: %.4f at ortho (%d,%d)' % (mx, cx, cy))
        assert abs(cx - 3608) <= 1 and abs(cy - 5360) <= 1 and mx > 0.40, \
            'template location does not reproduce baseline'

    win = read_window(LROC1M if name == '1m' else LROC3M, off, samples, cx, cy, hw)
    o8 = prep(win)
    p1, p2, nk1, nk2 = sift_pairs(t8, o8)
    print('sift keypoints: tmpl=%d win=%d lowe-matches=%d' % (nk1, nk2, len(p1)))

    # ---- baseline: reproduce exactly, else STOP ----
    Mb, inlb = ransac_fit(p1, p2)
    assert Mb is not None, 'baseline RANSAC failed'
    n_b = int(inlb.sum())
    rmse_b, r_b = fit_rmse(Mb, p1, p2, inlb)
    sc_b, ro_b, tx_b, ty_b = decomp(Mb)
    p1i, p2i = p1[inlb], p2[inlb]
    counts_b, ent_b = quad_stats(p2i, r_b[inlb], hw)
    tier_b = gate_tier(rmse_b, n_b, ent_b, counts_b)
    splits = spatial_split(p2i)
    hox_b, nfx, ncx = heldout_lmeds_split(p1i, p2i, *splits['x'], m_per_px)
    hoy_b, nfy, ncy = heldout_lmeds_split(p1i, p2i, *splits['y'], m_per_px)
    print('%s BASELINE: inliers=%d/%d rmse=%.4fpx=%.3fm scale=%.4f rot=%.3f '
          'tier=%s heldout-x=%.2fm(%d/%d) heldout-y=%.2fm(%d/%d)'
          % (name, n_b, len(p1), rmse_b, rmse_b * m_per_px, sc_b, ro_b,
             tier_b, hox_b, nfx, ncx, hoy_b, nfy, ncy))
    e = base_expect
    assert n_b == e['n_inl'], (n_b, e['n_inl'])
    assert len(p1) == e['n_corr'], (len(p1), e['n_corr'])
    assert abs(rmse_b - e['rmse']) < 0.005, (rmse_b, e['rmse'])
    assert abs(sc_b - e['scale']) < 0.002, (sc_b, e['scale'])
    assert abs(ro_b - e['rot']) < 0.02, (ro_b, e['rot'])
    assert abs(hox_b - e['hox']) < 0.15, (hox_b, e['hox'])
    assert abs(hoy_b - e['hoy']) < 0.15, (hoy_b, e['hoy'])
    print('%s BASELINE REPRODUCED within rounding' % name)

    # ---- field arm: strict no-leakage protocol ----
    # M_hook AND the field are fit on the fit-set ONLY; the check-set is
    # never seen by any fitted parameter. (An earlier draft fit M_hook on
    # all inliers; that leaked check-region information into the 4-DOF
    # global fit and inflated the apparent gain. This version is strict.)
    row = dict(grid=name, n_corr=len(p1), n_inliers=n_b,
               base_rmse_px=rmse_b, base_rmse_m=rmse_b * m_per_px,
               base_scale=sc_b, base_rot=ro_b, base_tier=tier_b,
               base_hox_m=hox_b, base_hoy_m=hoy_b,
               base_hox_fit=nfx, base_hox_check=ncx,
               base_hoy_fit=nfy, base_hoy_check=ncy)

    arm = {}
    for sname in ('x', 'y'):
        fi, ci = splits[sname]
        if len(fi) < 6 or len(ci) < 1:
            arm[sname] = dict(applied=False, reason='split_too_small')
            continue
        # M_hook on the fit-set only (pipeline hook uses RANSAC, seed 7).
        cv2.setRNGSeed(SEED)
        Mh_fi, _ = cv2.estimateAffinePartial2D(
            p1i[fi], p2i[fi], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if Mh_fi is None:
            arm[sname] = dict(applied=False, reason='hook_ransac_none')
            print('%s field-%s: hook RANSAC None on fit-set -> declined'
                  % (name, sname))
            continue
        sc_h, ro_h, _, _ = decomp(Mh_fi)
        if sname == 'x':
            Mh_fi_x = Mh_fi.copy()
        info = apply_deform_field_stage(p1i[fi], p2i[fi], Mh_fi, seed=SEED)
        rec = dict(applied=bool(info['applied']), reason=info['reason'],
                   n_fit=len(fi), n_check=len(ci),
                   hook_dscale_fit=abs(sc_h - sc_b),
                   hook_drot_fit_deg=abs(ro_h - ro_b),
                   lambda_chosen=info.get('lambda_chosen'),
                   rmse_before_px=info.get('rmse_before_px'),
                   rmse_after_px=info.get('rmse_after_px'),
                   internal_holdout_px=info.get('heldout_rmse_px'),
                   internal_holdout_affine_px=info.get('heldout_rmse_affine_px'),
                   max_disp_px=info.get('max_displacement_px'),
                   min_jac_det=info.get('min_jacobian_det'))
        if not info['applied']:
            print('%s field-%s: DECLINED (%s) on %d fit pts'
                  % (name, sname, info['reason'], len(fi)))
            arm[sname] = rec
            continue
        field = info['field']
        Mh_arr = np.asarray(Mh_fi, dtype=np.float64)
        # folding check on the check set too (field never saw these points)
        min_det_check = _jacobian_min_det(Mh_arr, field, p1i[ci])
        rec['min_jac_det_check'] = float(min_det_check)
        # evaluate on the unseen check set
        d_check = _eval_residual_field(field, p1i[ci])
        pred = _apply_affine(Mh_arr, p1i[ci]) + d_check
        chk_px = float(np.sqrt((np.linalg.norm(pred - p2i[ci], axis=1) ** 2).mean()))
        rec['check_rmse_px'] = chk_px
        rec['check_rmse_m'] = chk_px * m_per_px
        # inlier RMSE after field correction (same inlier set) for tier check
        d_all = _eval_residual_field(field, p1i)
        r_corr = np.linalg.norm(
            (_apply_affine(Mh_arr, p1i) + d_all) - p2i, axis=1)
        rmse_corr = float(np.sqrt((r_corr ** 2).mean()))
        rec['inlier_rmse_corr_px'] = rmse_corr
        rec['tier_corr'] = gate_tier(rmse_corr, n_b, ent_b, counts_b)
        base_ho = hox_b if sname == 'x' else hoy_b
        rec['bound_tightened'] = bool(rec['check_rmse_m'] < base_ho)
        rec['bound_delta_m'] = float(base_ho - rec['check_rmse_m'])
        print('%s field-%s: APPLIED λ=%s fit-rmse %.3f->%.3fpx | '
              'check %.3fpx=%.3fm vs baseline %.2fm | Δ=%+.3fm | '
              'minJac %.3f (check %.3f) | hookΔ(dscale=%.5f,drot=%.3f°) | tier %s->%s'
              % (name, sname, rec['lambda_chosen'], rec['rmse_before_px'],
                 rec['rmse_after_px'], chk_px, rec['check_rmse_m'], base_ho,
                 rec['bound_delta_m'], rec['min_jac_det'],
                 rec['min_jac_det_check'], rec['hook_dscale_fit'],
                 rec['hook_drot_fit_deg'], tier_b, rec['tier_corr']))
        arm[sname] = rec

    # determinism re-run of the field arm (x-split)
    fi, ci = splits['x']
    if len(fi) >= 6:
        cv2.setRNGSeed(SEED)
        Mh2, _ = cv2.estimateAffinePartial2D(
            p1i[fi], p2i[fi], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        info2 = apply_deform_field_stage(p1i[fi], p2i[fi], Mh2, seed=SEED)
        det_same = (bool(info2['applied']) == bool(arm['x']['applied'])
                    and info2['reason'] == arm['x']['reason']
                    and info2.get('lambda_chosen') == arm['x'].get('lambda_chosen')
                    and (Mh2 is not None) and np.allclose(Mh2, Mh_fi_x))
        print('%s determinism re-run: %s' % (name, 'IDENTICAL' if det_same else 'MISMATCH'))
        row['determinism'] = 'IDENTICAL' if det_same else 'MISMATCH'
        assert det_same, 'field arm not deterministic'
    else:
        row['determinism'] = 'n/a'

    for sname in ('x', 'y'):
        for k, v in arm[sname].items():
            row['field_%s_%s' % (sname, k)] = v
    return row


t0 = time.time()
res1m = run_grid('1m', 532, 1.0, 10818, 16081, 450, 46006, 47683, 23003,
                 do_tmpl_match=False,
                 base_expect=dict(n_corr=164, n_inl=96, rmse=1.4120,
                                  scale=1.0156, rot=2.12, hox=2.58, hoy=3.46))
res3m = run_grid('3m', 178, 3.0, 3608, 5360, 180, 15336, 15895, 7668,
                 do_tmpl_match=True,
                 base_expect=dict(n_corr=46, n_inl=46, rmse=0.8611,
                                  scale=1.0129, rot=2.679, hox=7.06, hoy=6.39))

hdr = ['grid', 'n_corr', 'n_inliers', 'base_rmse_px', 'base_rmse_m',
       'base_scale', 'base_rot', 'base_tier', 'base_hox_m', 'base_hoy_m',
       'base_hox_fit', 'base_hox_check', 'base_hoy_fit', 'base_hoy_check',
       'determinism',
       'field_x_applied', 'field_x_reason', 'field_x_n_fit', 'field_x_n_check',
       'field_x_hook_dscale_fit', 'field_x_hook_drot_fit_deg',
       'field_x_lambda_chosen', 'field_x_rmse_before_px', 'field_x_rmse_after_px',
       'field_x_internal_holdout_px', 'field_x_internal_holdout_affine_px',
       'field_x_max_disp_px', 'field_x_min_jac_det', 'field_x_min_jac_det_check',
       'field_x_check_rmse_px', 'field_x_check_rmse_m',
       'field_x_inlier_rmse_corr_px', 'field_x_tier_corr',
       'field_x_bound_tightened', 'field_x_bound_delta_m',
       'field_y_applied', 'field_y_reason', 'field_y_n_fit', 'field_y_n_check',
       'field_y_hook_dscale_fit', 'field_y_hook_drot_fit_deg',
       'field_y_lambda_chosen', 'field_y_rmse_before_px', 'field_y_rmse_after_px',
       'field_y_internal_holdout_px', 'field_y_internal_holdout_affine_px',
       'field_y_max_disp_px', 'field_y_min_jac_det', 'field_y_min_jac_det_check',
       'field_y_check_rmse_px', 'field_y_check_rmse_m',
       'field_y_inlier_rmse_corr_px', 'field_y_tier_corr',
       'field_y_bound_tightened', 'field_y_bound_delta_m']
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=hdr, extrasaction='ignore')
    w.writeheader()
    w.writerow(res1m)
    w.writerow(res3m)
print('wrote', OUT_CSV, 'in %.1fs' % (time.time() - t0))
