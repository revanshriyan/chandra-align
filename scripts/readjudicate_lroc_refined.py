"""LROC re-adjudication with ECC-refined correspondences (worker A10).

Reproduces the 1m and 3m absolute-truth adjudications exactly (SIFT 8000,
CLAHE, Lowe 0.75, partial-affine RANSAC 3.0px, seed 7), then re-runs each
with patch-constrained ECC refinement of the SIFT inlier correspondences
(A9 arm-c logic: 31x31 patches, MOTION_TRANSLATION, per-image bounds,
6px max-shift guard, fallback to original on failure).

Pre-registered success criterion: the held-out ABSOLUTE bound (meters)
improves vs baseline on either grid with the transform agreeing with
baseline (rot within 0.05deg, scale within 0.001). Inlier-RMSE improvement
alone is NOT a claim.

Deterministic: cv2.setRNGSeed(7) before every randomized estimator call.
Fail-closed: RANSAC returning None -> DEGENERATE row, no crash.
"""
import numpy as np, cv2, csv, sys, time

SEED = 7
REPO = '/home/hatch/workspace/chandra-align'
LROC1M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_100CM.IMG'
LROC3M = '/home/hatch/workspace/lroc_data/NAC_ORTHO_VIKRAMSITE1_3M.IMG'
CROP = REPO + '/data/benchmark_crops/ohrc_01_reference.png'
sys.path.insert(0, REPO)
from chandra_align.metrics.quadrant import compute_quadrant_metrics, validate_registration_gate

OUT_CSV = REPO + '/results/table_lroc_refined.csv'


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


def heldout_lmeds(p1, p2, inl, m_per_px):
    """Spatial x/y split: fit LMEDS on coord<=lo+40%, check on coord>=lo+60%."""
    out = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        v = p2[inl, axis]
        lo, hi = v.min(), v.max()
        fi = np.where(inl & (p2[:, axis] <= lo + 0.40 * (hi - lo)))[0]
        ci = np.where(inl & (p2[:, axis] >= lo + 0.60 * (hi - lo)))[0]
        if len(fi) < 6 or len(ci) < 1:
            out[name] = (float('nan'), float('nan'), len(fi), len(ci), ci, fi)
            continue
        cv2.setRNGSeed(SEED)
        Mf, _ = cv2.estimateAffinePartial2D(p1[fi], p2[fi], method=cv2.LMEDS)
        if Mf is None:
            out[name] = (float('nan'), float('nan'), len(fi), len(ci), ci, fi)
            continue
        pred = (Mf @ np.hstack([p1[ci], np.ones((len(ci), 1))]).T).T
        chk = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
        out[name] = (chk, chk * m_per_px, len(fi), len(ci), ci, fi)
    return out


def gate_tier(rmse, n_inl, ent, counts):
    _, tier = validate_registration_gate(
        rmse, n_inl, 8, ent,
        {'Q1': counts[0], 'Q2': counts[1], 'Q3': counts[2], 'Q4': counts[3]})
    return tier


def quad_stats(p2inl, rinl, hw):
    quads, ent = compute_quadrant_metrics(p2inl, rinl, (2 * hw, 2 * hw))
    counts = [int(quads[k]['inlier_count']) for k in ('Q1', 'Q2', 'Q3', 'Q4')]
    return counts, float(ent)


def refine_ecc(t_img, o_img, p1, p2, patch=31, max_shift=6.0):
    """Patch-constrained ECC refinement of p2 (ortho-side) positions.

    t_img: prepped template (uint8), o_img: prepped ortho window (uint8).
    For each pair: 31x31 patches around integer keypoint coords,
    findTransformECC(template_patch, input_patch, MOTION_TRANSLATION);
    p2 <- p2 + warp[:,2]. Shift > max_shift or cv2.error -> keep original.
    """
    out = p2.copy()
    n_ok = n_fail = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    h = patch // 2
    H1, W1 = t_img.shape
    H2, W2 = o_img.shape
    for i in range(len(p1)):
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        if not (h <= x1 < W1 - h and h <= y1 < H1 - h
                and h <= x2 < W2 - h and h <= y2 < H2 - h):
            n_fail += 1
            continue
        tp = t_img[int(y1) - h:int(y1) + h + 1, int(x1) - h:int(x1) + h + 1].astype(np.float32)
        ip = o_img[int(y2) - h:int(y2) + h + 1, int(x2) - h:int(x2) + h + 1].astype(np.float32)
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            cv2.findTransformECC(tp, ip, warp, cv2.MOTION_TRANSLATION, criteria)
            sh = warp[:, 2]
            if float(np.hypot(sh[0], sh[1])) > max_shift:
                n_fail += 1
                continue
            out[i] = p2[i] + sh
            n_ok += 1
        except cv2.error:
            n_fail += 1
    return out, n_ok, n_fail


def sift_pairs(t8, o8):
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(t8, None)
    k2, d2 = sift.detectAndCompute(o8, None)
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, nn in raw if m.distance < 0.75 * nn.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2, len(k1), len(k2)


# ------------------------------------------------- 0. ECC sign self-test
# Shift a real patch by a KNOWN amount; check that p2 + warp[:,2] recovers it.
print('=== 0. ECC SIGN SELF-TEST ===')
rng = np.random.RandomState(3)
probe = (rng.rand(64, 64) * 255).astype(np.float32)
dx_true, dy_true = 2.7, -1.9
probe_shifted = cv2.warpAffine(
    probe, np.float32([[1, 0, dx_true], [0, 1, dy_true]]), (64, 64),
    flags=cv2.INTER_LINEAR)
w = np.eye(2, 3, dtype=np.float32)
cv2.findTransformECC(probe, probe_shifted, w, cv2.MOTION_TRANSLATION,
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
p2_probe = np.float32([[32.0, 32.0]])
p2_plus = p2_probe + w[:, 2]
p2_minus = p2_probe - w[:, 2]
err_plus = float(np.hypot(*(p2_plus[0] - (32 + dx_true, 32 + dy_true))))
err_minus = float(np.hypot(*(p2_minus[0] - (32 + dx_true, 32 + dy_true))))
print('warp[:,2]=%s err(p2+warp)=%.3f err(p2-warp)=%.3f'
      % (np.round(w[:, 2], 3), err_plus, err_minus))
assert err_plus < 0.5 and err_plus < 0.2 * err_minus, 'ECC sign convention broken'
print('sign convention confirmed: p2_refined = p2 + warp[:,2]')


def run_grid(name, tsz, m_per_px, cx, cy, hw, off, lines, samples,
            do_tmpl_match, base_expect):
    global ROWS
    print('=== GRID %s: template %dx%d @ %.1fm/px, window %dpx at (%d,%d) ==='
          % (name, tsz, tsz, m_per_px, 2 * hw, cx, cy))
    ohrc = cv2.imread(CROP, cv2.IMREAD_GRAYSCALE)
    assert ohrc is not None and ohrc.shape == (2048, 2048)
    tmpl = cv2.resize(ohrc, (tsz, tsz), interpolation=cv2.INTER_AREA).astype(np.float32)
    t8 = prep(tmpl)

    if do_tmpl_match:
        # 3m: re-run the NCC template search exactly as the baseline did
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

    results = {}

    def process(arm, p1a, p2a, n_corr, base=None, ecc_stat=None):
        M, inl = ransac_fit(p1a, p2a)
        if M is None:
            print('%s %s: RANSAC returned None -> DEGENERATE (fail closed)'
                  % (name, arm))
            ROWS.append((name, arm, n_corr, 0, float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'),
                         'DEGENERATE_FAILURE', float('nan'), float('nan'),
                         float('nan'), 0, 0, 0, 0, float('nan'),
                         float('nan'), float('nan'), 0, 0, 0, 0, SEED,
                         *(ecc_stat or (0, 0)), 'ransac-none'))
            results[arm] = None
            return None, None, None
        n = int(inl.sum())
        rmse, r = fit_rmse(M, p1a, p2a, inl)
        sc, ro, tx, ty = decomp(M)
        counts, ent = quad_stats(p2a[inl], r[inl], hw)
        tier = gate_tier(rmse, n, ent, counts)
        ho = heldout_lmeds(p1a, p2a, inl, m_per_px)
        agree = (float('nan'), float('nan'), float('nan'))
        if base is not None:
            bsc, bro, btx, bty = base
            agree = (abs(ro - bro), abs(sc - bsc), float(np.hypot(tx - btx, ty - bty)))
        ROWS.append((name, arm, n_corr, n, rmse, rmse * m_per_px, sc, ro, tx, ty,
                     tier, agree[0], agree[1], agree[2],
                     counts[0], counts[1], counts[2], counts[3], ent,
                     ho['x'][1], ho['y'][1],
                     ho['x'][2], ho['x'][3], ho['y'][2], ho['y'][3], SEED,
                     *(ecc_stat or (0, 0)),
                     'x-fit=%d/x-check=%d y-fit=%d/y-check=%d'
                     % (ho['x'][2], ho['x'][3], ho['y'][2], ho['y'][3])))
        print('%s %s: inliers=%d/%d rmse=%.4fpx=%.3fm scale=%.4f rot=%.3f '
              'quads=%s ent=%.3f tier=%s heldout-x=%.2fm(%d/%d) '
              'heldout-y=%.2fm(%d/%d)%s'
              % (name, arm, n, n_corr, rmse, rmse * m_per_px, sc, ro, counts,
                 ent, tier, ho['x'][1], ho['x'][2], ho['x'][3],
                 ho['y'][1], ho['y'][2], ho['y'][3],
                 ('' if ecc_stat is None else
                  ' ecc_ok=%d/fb=%d agree(drot=%.4f,dscale=%.5f,dtrans=%.3f)'
                  % (ecc_stat[0], ecc_stat[1], agree[0], agree[1], agree[2]))))
        results[arm] = dict(rmse=rmse, sc=sc, ro=ro, tx=tx, ty=ty, n=n,
                            hox=ho['x'][1], hoy=ho['y'][1])
        return M, inl, ho

    # ---- baseline: reproduce exactly, else STOP ----
    Mb, inlb, hob = process('baseline', p1, p2, len(p1))
    assert Mb is not None, 'baseline RANSAC failed'
    rb = results['baseline']
    e = base_expect
    assert rb['n'] == e['n_inl'], (rb['n'], e['n_inl'])
    assert len(p1) == e['n_corr'], (len(p1), e['n_corr'])
    assert abs(rb['rmse'] - e['rmse']) < 0.005, (rb['rmse'], e['rmse'])
    assert abs(rb['sc'] - e['scale']) < 0.002, (rb['sc'], e['scale'])
    assert abs(rb['ro'] - e['rot']) < 0.02, (rb['ro'], e['rot'])
    assert abs(rb['hox'] - e['hox']) < 0.15, (rb['hox'], e['hox'])
    assert abs(rb['hoy'] - e['hoy']) < 0.15, (rb['hoy'], e['hoy'])
    print('%s BASELINE REPRODUCED within rounding' % name)

    # ---- refined arm: ECC on the baseline inlier pairs ----
    p1i, p2i = p1[inlb], p2[inlb]
    p2r, n_ok, n_fail = refine_ecc(t8, o8, p1i, p2i)
    Mr, inlr, hor = process('ecc_refined', p1i, p2r, len(p1i),
                            base=(rb['sc'], rb['ro'], rb['tx'], rb['ty']),
                            ecc_stat=(n_ok, n_fail))

    # ---- supplementary: refined transform evaluated on the BASELINE check
    # sets (same points, original unrefined positions) — isolates transform
    # quality from the inlier-set selection effect.
    cross = {}
    if Mr is not None:
        for sname in ('x', 'y'):
            ci_b = hob[sname][4]
            if len(ci_b) == 0:
                cross[sname] = float('nan')
                continue
            pred = (Mr @ np.hstack([p1[ci_b], np.ones((len(ci_b), 1))]).T).T
            chk = float(np.sqrt((np.linalg.norm(pred - p2[ci_b], axis=1) ** 2).mean()))
            cross[sname] = chk * m_per_px
            print('%s cross-check: refined-M on baseline %s-check (%d pts): '
                  '%.2fm (baseline held-out was %.2fm)'
                  % (name, sname, len(ci_b), cross[sname], hob[sname][1]))
    # ---- causal test: IDENTICAL fit/check splits, only the fit positions
    # are refined. Fit LMEDS on baseline fit indices with refined positions,
    # evaluate on baseline check indices with original positions. This
    # isolates the refinement effect from fit-set size/selection effects.
    causal = {}
    if Mr is not None:
        inlb_idx = np.where(inlb)[0]
        lut = {g: j for j, g in enumerate(inlb_idx.tolist())}
        for sname, ax in (('x', 0), ('y', 1)):
            fi_b = hob[sname][5]
            ci_b = hob[sname][4]
            fi_sub = np.array([lut[g] for g in fi_b.tolist() if g in lut],
                              dtype=int)
            if len(fi_sub) < 6 or len(ci_b) < 1:
                causal[sname] = float('nan')
                continue
            cv2.setRNGSeed(SEED)
            Mf_ref, _ = cv2.estimateAffinePartial2D(
                p1i[fi_sub], p2r[fi_sub], method=cv2.LMEDS)
            if Mf_ref is None:
                causal[sname] = float('nan')
                continue
            pred = (Mf_ref @ np.hstack([p1[ci_b], np.ones((len(ci_b), 1))]).T).T
            chk = float(np.sqrt((np.linalg.norm(pred - p2[ci_b], axis=1) ** 2).mean()))
            causal[sname] = chk * m_per_px
            print('%s causal: LMEDS on baseline %s-fit idx w/ refined pos '
                  '(%d pts) -> baseline %s-check (%d pts): %.2fm '
                  '(baseline: %.2fm)'
                  % (name, sname, len(fi_sub), sname, len(ci_b),
                     causal[sname], hob[sname][1]))
    results['cross'] = cross
    results['causal'] = causal
    CAUSAL[(name, 'ecc_refined')] = (causal.get('x', float('nan')),
                                     causal.get('y', float('nan')))
    return results


t0 = time.time()
ROWS = []
CAUSAL = {}
res1m = run_grid('1m', 532, 1.0, 10818, 16081, 450, 46006, 47683, 23003,
                 do_tmpl_match=False,
                 base_expect=dict(n_corr=164, n_inl=96, rmse=1.4120,
                                  scale=1.0156, rot=2.12, hox=2.58, hoy=3.46))
res3m = run_grid('3m', 178, 3.0, 3608, 5360, 180, 15336, 15895, 7668,
                 do_tmpl_match=True,
                 base_expect=dict(n_corr=46, n_inl=46, rmse=0.8611,
                                  scale=1.0129, rot=2.679, hox=7.06, hoy=6.39))

# ------------------------------------------------- CSV
# Insert causal columns (causal_x_m, causal_y_m) after heldout_y_check.
hdr = ['grid', 'arm', 'n_corr', 'n_inliers', 'inlier_rmse_px',
       'inlier_rmse_m', 'scale', 'rotation_deg', 'tx_px', 'ty_px',
       'gate_tier', 'agree_drot_deg', 'agree_dscale',
       'agree_dtrans_px', 'q1', 'q2', 'q3', 'q4', 'entropy_4q',
       'heldout_x_m', 'heldout_y_m', 'heldout_x_fit',
       'heldout_x_check', 'heldout_y_fit', 'heldout_y_check',
       'causal_x_m', 'causal_y_m',
       'seed', 'n_ecc_ok', 'n_ecc_fallback', 'notes']
fixed = []
for row in ROWS:
    row = list(row)
    cxv, cyv = CAUSAL.get((row[0], row[1]), (float('nan'), float('nan')))
    row[25:25] = [cxv, cyv]
    assert len(row) == len(hdr), (len(row), len(hdr))
    fixed.append(tuple(row))
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(fixed)
print('wrote', OUT_CSV, '(%d rows)' % len(fixed))

# ------------------------------------------------- pre-registered success criterion
print('=== PRE-REGISTERED SUCCESS CRITERION ===')
print('held-out ABSOLUTE bound (m) improves vs baseline on either grid,')
print('with transform agreement (drot<=0.05deg, dscale<=0.001).')
overall = False
for name, res in (('1m', res1m), ('3m', res3m)):
    b, r = res['baseline'], res['ecc_refined']
    if b is None or r is None:
        print('%s: missing arm -> NO CLAIM' % name)
        continue
    agree = abs(r['ro'] - b['ro']) <= 0.05 and abs(r['sc'] - b['sc']) <= 0.001
    better = (r['hox'] <= b['hox']) and (r['hoy'] <= b['hoy']) and \
             ((r['hox'] < b['hox']) or (r['hoy'] < b['hoy']))
    ok = agree and better
    overall = overall or ok
    cx_ = res.get('cross', {})
    print('%s: baseline ho=(%.2f, %.2f)m refined ho=(%.2f, %.2f)m '
          'agree=%s better=%s => %s'
          % (name, b['hox'], b['hoy'], r['hox'], r['hoy'], agree, better,
             'PASS' if ok else 'FAIL'))
    if cx_:
        print('%s: supplementary cross-check (refined-M on baseline check '
              'sets): x=%.2fm y=%.2fm' % (name, cx_['x'], cx_['y']))
    caz = res.get('causal', {})
    if caz:
        print('%s: causal test (identical splits, refined fit positions): '
              'x=%.2fm (baseline %.2fm) y=%.2fm (baseline %.2fm)'
              % (name, caz['x'], b['hox'], caz['y'], b['hoy']))
print('OVERALL:', 'SUCCESS CRITERION MET' if overall else 'NOT MET')
print('elapsed %.0fs' % (time.time() - t0))
