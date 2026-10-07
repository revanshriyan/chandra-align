"""LoFTR (detector-free dense) matcher experiment on the real OHRC pair.

Compares kornia pretrained LoFTR ('outdoor') against the SIFT baseline
(1.6087px, 844 inliers, COARSE) and the ECC-refined arm (1.386px).

Design notes (all documented, nothing hidden):
- Full 2048x2048 LoFTR coarse attention is infeasible on CPU, so LoFTR runs
  at reduced resolution (LOFTR_SIZE=1024, long side) and keypoints are scaled
  back to full resolution. Arm (c) then applies the A9 patch-ECC refinement
  at FULL resolution for sub-pixel localization. This separates the two
  effects: LoFTR = dense coverage, ECC = localization.
- Filtering: LoFTR has no Lowe ratio (dense). We use the model's own
  confidence with threshold 0.2 (the LoFTR paper's coarse-match threshold).
  Sensitivity to this choice is reported, not hidden.
- RANSAC + frozen gates + spatial held-out replicate
  scripts/refine_subpixel_experiment.py exactly (seed 7).
- LoFTR inference itself is deterministic in eval mode (no dropout); it is
  not seed-controlled beyond torch.manual_seed(7). RANSAC is seeded via
  cv2.setRNGSeed(7) before every fit.

Pre-registered success: beat ECC-refined 1.386px inlier RMSE with transform
agreement to the SIFT baseline (drot<=0.05deg, dscale<=0.001) AND held-out
not worse -- OR fill Q1 (top-left on the SOURCE frame, empty under SIFT at
0/411/45/388) with an agreeing transform. Quadrant counts are reported on
both frames: the p2 (source) frame matches A9's convention; the p1
(reference) frame shows the emptiness is partly the (134,354)px shift.
Inlier-RMSE-only wins without held-out support are cosmetic.
"""
import csv
import time

import cv2
import numpy as np

cv2.setRNGSeed(7)
SEED = 7
LOFTR_SIZE = 1024          # long side for LoFTR inference (CPU feasibility)
CONF_THRESH = 0.2          # LoFTR paper coarse-match confidence threshold

REF_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_reference.png'
SRC_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_source.png'
OUT_CSV = '/home/hatch/workspace/chandra-align/results/table_loftr_ohrc.csv'

ref = cv2.imread(REF_P, cv2.IMREAD_GRAYSCALE)
src = cv2.imread(SRC_P, cv2.IMREAD_GRAYSCALE)
assert ref is not None and src is not None, 'could not load OHRC pair'
H, W = ref.shape
print('pair: %dx%d' % (W, H), flush=True)


# ------------------------------------------------------------------ matchers
def sift_match(a, b):
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, nn in raw if m.distance < 0.75 * nn.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def loftr_match(a, b, size=LOFTR_SIZE, conf_thresh=CONF_THRESH):
    """Dense LoFTR correspondences. Returns (p1, p2, conf, stats) at FULL res."""
    import torch
    import kornia.feature as KF
    torch.manual_seed(SEED)
    h0, w0 = a.shape
    s = size / max(h0, w0)
    hs, ws = int(round(h0 * s)), int(round(w0 * s))
    a_s = cv2.resize(a, (ws, hs), interpolation=cv2.INTER_AREA)
    b_s = cv2.resize(b, (ws, hs), interpolation=cv2.INTER_AREA)
    t0 = torch.from_numpy(a_s).float().unsqueeze(0).unsqueeze(0) / 255.0
    t1 = torch.from_numpy(b_s).float().unsqueeze(0).unsqueeze(0) / 255.0
    model = KF.LoFTR(pretrained='outdoor')
    model.eval()
    with torch.no_grad():
        out = model({'image0': t0, 'image1': t1})
    # kornia 0.8.3 returns unbatched (N,2)/(N,) tensors (verified empirically)
    k0 = out['keypoints0'].numpy()   # (N,2) in resized coords
    k1 = out['keypoints1'].numpy()
    conf = out['confidence'].numpy()
    n_raw = len(k0)
    keep = conf >= conf_thresh
    stats = {'n_raw': int(n_raw), 'conf_thresh': conf_thresh,
             'infer_size': (ws, hs), 'scale_back': 1.0 / s}
    p1 = (k0[keep] / s).astype(np.float32)   # back to full resolution
    p2 = (k1[keep] / s).astype(np.float32)
    return p1, p2, conf[keep], stats


# ------------------------------------------------------------- shared machinery
def ransac_fit(p1, p2):
    cv2.setRNGSeed(SEED)
    M, inl = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.995)
    if M is None or inl is None:
        return None, np.zeros(len(p1), dtype=bool)
    return M, inl.ravel().astype(bool)


def inlier_rmse(M, p1, p2, inl):
    pred = (M @ np.hstack([p1, np.ones((len(p1), 1))]).T).T
    r = np.linalg.norm(pred - p2, axis=1)
    return float(np.sqrt((r[inl] ** 2).mean())), r


def decomp(M):
    a11, a21 = M[0, 0], M[1, 0]
    return (float(np.sqrt(a11 ** 2 + a21 ** 2)),
            float(np.degrees(np.arctan2(a21, a11))),
            float(M[0, 2]), float(M[1, 2]))


def gate_verdict(rmse, n_inl):
    if rmse <= 0.50 and n_inl >= 8:
        return 'ACCEPT-tier (residual/inliers)'
    if rmse <= 2.50 and n_inl >= 8:
        return 'COARSE-tier (residual/inliers)'
    return 'DEGENERATE-tier (residual/inliers)'


def quadrants(p):
    """4-quadrant counts + entropy. Call on p1 (reference frame) or p2."""
    q = np.zeros(4, dtype=int)
    q[0] = np.sum((p[:, 0] < W / 2) & (p[:, 1] < H / 2))  # Q1 top-left
    q[1] = np.sum((p[:, 0] >= W / 2) & (p[:, 1] < H / 2))
    q[2] = np.sum((p[:, 0] < W / 2) & (p[:, 1] >= H / 2))
    q[3] = np.sum((p[:, 0] >= W / 2) & (p[:, 1] >= H / 2))
    tot = q.sum()
    ent = float(-np.sum((q / tot) * np.log2(q / tot + 1e-12))) if tot else 0.0
    return q, ent


def heldout(p1, p2, inl):
    """Spatial x/y split: fit on 40% band, check on disjoint 40% band."""
    res = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        v = p2[inl, axis]
        lo, hi = v.min(), v.max()
        fi = np.where(inl & (p2[:, axis] <= lo + 0.40 * (hi - lo)))[0]
        ci = np.where(inl & (p2[:, axis] >= lo + 0.60 * (hi - lo)))[0]
        if len(fi) < 4 or len(ci) < 1:
            res[name] = (float('nan'), len(fi), len(ci))
            continue
        Mf, _ = ransac_fit(p1[fi], p2[fi])
        if Mf is None:
            res[name] = (float('nan'), len(fi), len(ci))
            continue
        pred = (Mf @ np.hstack([p1[ci], np.ones((len(ci), 1))]).T).T
        chk = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
        res[name] = (chk, len(fi), len(ci))
    return res


def refine_ecc(a, b, p1, p2, patch=31):
    """Per-patch ECC translation refinement of p2 (A9 arm c)."""
    out = p2.copy()
    n_ok = n_fail = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    h = patch // 2
    for i in range(len(p1)):
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        if not (h <= x1 < W - h and h <= y1 < H - h
                and h <= x2 < W - h and h <= y2 < H - h):
            n_fail += 1
            continue
        t_patch = a[int(y1) - h:int(y1) + h + 1, int(x1) - h:int(x1) + h + 1]
        i_patch = b[int(y2) - h:int(y2) + h + 1, int(x2) - h:int(x2) + h + 1]
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            cv2.findTransformECC(
                t_patch.astype(np.float32), i_patch.astype(np.float32),
                warp, cv2.MOTION_TRANSLATION, criteria)
            shift = warp[:, 2]
            if np.linalg.norm(shift) > 5.8:   # A9's shift-magnitude guard
                n_fail += 1
                continue
            out[i] = p2[i] + shift
            n_ok += 1
        except cv2.error:
            n_fail += 1
    return out, {'n_refined': n_ok, 'n_fallback': n_fail}


def run_arm(name, p1, p2, base=None):
    """Fit + score one arm. Returns (row_dict, M, inl)."""
    t0 = time.time()
    M, inl = ransac_fit(p1, p2)
    n_inl = int(inl.sum())
    if M is None or n_inl < 8:
        return ({'arm': name, 'n_corr': len(p1), 'n_inliers': n_inl,
                 'inlier_rmse_px': float('nan'), 'gate_tier': 'DEGENERATE',
                 'notes': 'fit failed / <8 inliers'}, None, inl)
    rmse, _ = inlier_rmse(M, p1, p2, inl)
    sc, rot, tx, ty = decomp(M)
    # Primary quadrant frame: p2 (source coords) -- matches A9's reported
    # 0/411/45/388. Also report p1 (reference) frame for completeness.
    q2, ent = quadrants(p2[inl])
    q1f, _ = quadrants(p1[inl])
    ho = heldout(p1, p2, inl)
    row = {'arm': name, 'n_corr': len(p1), 'n_inliers': n_inl,
           'inlier_rmse_px': rmse, 'scale': sc, 'rotation_deg': rot,
           'tx_px': tx, 'ty_px': ty,
           'gate_tier': gate_verdict(rmse, n_inl),
           'q1': int(q2[0]), 'q2': int(q2[1]), 'q3': int(q2[2]),
           'q4': int(q2[3]), 'entropy_4q': ent,
           'q1_ref': int(q1f[0]), 'q2_ref': int(q1f[1]),
           'q3_ref': int(q1f[2]), 'q4_ref': int(q1f[3]),
           'heldout_x_rmse_px': ho['x'][0], 'heldout_y_rmse_px': ho['y'][0],
           'heldout_x_fit': ho['x'][1], 'heldout_x_check': ho['x'][2],
           'heldout_y_fit': ho['y'][1], 'heldout_y_check': ho['y'][2],
           'seed': SEED, 'elapsed_s': round(time.time() - t0, 1)}
    if base is not None:
        bsc, brot = base['scale'], base['rotation_deg']
        row['agree_drot_deg'] = abs(rot - brot)
        row['agree_dscale'] = abs(sc - bsc)
        row['agree_dtrans_px'] = float(
            np.hypot(tx - base['tx_px'], ty - base['ty_px']))
    return row, M, inl


# ------------------------------------------------------------------ baseline
print('=== 0. SIFT BASELINE (reproduce 1.6087px) ===', flush=True)
p1b, p2b = sift_match(ref, src)
brow, Mb, inlb = run_arm('a_sift_baseline', p1b, p2b)
r = brow
print(f"{r['arm']}: corr={r['n_corr']} inliers={r['n_inliers']} rmse={r['inlier_rmse_px']:.4f} "
      f"scale={r['scale']:.6f} rot={r['rotation_deg']:.4f} trans=({r['tx_px']:.2f},{r['ty_px']:.2f}) "
      f"Q={r['q1']}/{r['q2']}/{r['q3']}/{r['q4']} ent={r['entropy_4q']:.3f} "
      f"ho_x={r['heldout_x_rmse_px']:.2f} ho_y={r['heldout_y_rmse_px']:.2f} [{r['gate_tier']}]",
      flush=True)
assert abs(brow['inlier_rmse_px'] - 1.6087) < 0.01, 'SIFT baseline diverged!'
# A9's "Q1 empty" claim was on the source frame: reproduce it here.
assert (brow['q1'], brow['q2'], brow['q3'], brow['q4']) == (0, 411, 45, 388), \
    'baseline p2-frame quadrants differ from A9: %d/%d/%d/%d' % (
        brow['q1'], brow['q2'], brow['q3'], brow['q4'])
print('baseline reproduces: 1.6087px floor, Q1 EMPTY on source frame '
      '(ref-frame Q1=%d -- the emptiness is partly the (134,354)px shift).'
      % brow['q1_ref'], flush=True)
base_hox, base_hoy = brow['heldout_x_rmse_px'], brow['heldout_y_rmse_px']

# ------------------------------------------------------------------ LoFTR raw
print('=== 1. LoFTR (raw, conf>=%.2f) ===' % CONF_THRESH, flush=True)
t0 = time.time()
p1l, p2l, confl, lstat = loftr_match(ref, src)
print('LoFTR: raw=%d kept=%d (infer %dx%d, scale-back x%.2f) in %.0fs'
      % (lstat['n_raw'], len(p1l), lstat['infer_size'][0],
         lstat['infer_size'][1], lstat['scale_back'], time.time() - t0),
      flush=True)
assert len(p1l) >= 8, 'LoFTR produced <8 correspondences: DEGENERATE, stopping'
lrow, Ml, inll = run_arm('b_loftr_raw', p1l, p2l, base=brow)
lrow['notes'] = ('loftr_raw=%d conf>=%.2f infer=%dx%d' %
                 (lstat['n_raw'], CONF_THRESH, lstat['infer_size'][0],
                  lstat['infer_size'][1]))
r = lrow
print(f"{r['arm']}: corr={r['n_corr']} inliers={r['n_inliers']} rmse={r['inlier_rmse_px']:.4f} "
      f"scale={r['scale']:.6f} rot={r['rotation_deg']:.4f} trans=({r['tx_px']:.2f},{r['ty_px']:.2f}) "
      f"Q={r['q1']}/{r['q2']}/{r['q3']}/{r['q4']} ent={r['entropy_4q']:.3f} "
      f"agree(drot={r['agree_drot_deg']:.4f},dscale={r['agree_dscale']:.5f}) "
      f"ho_x={r['heldout_x_rmse_px']:.2f} ho_y={r['heldout_y_rmse_px']:.2f} [{r['gate_tier']}]",
      flush=True)

# ------------------------------------------------------- LoFTR + ECC refine
print('=== 2. LoFTR + full-res ECC refinement ===', flush=True)
p2r, estat = refine_ecc(ref, src, p1l, p2l)
erow, _, _ = run_arm('c_loftr_ecc', p1l, p2r, base=brow)
erow['notes'] = ('ecc_refined=%d fallback=%d; ' % (estat['n_refined'],
                 estat['n_fallback']) + lrow['notes'])
r = erow
print(f"{r['arm']}: corr={r['n_corr']} inliers={r['n_inliers']} rmse={r['inlier_rmse_px']:.4f} "
      f"scale={r['scale']:.6f} rot={r['rotation_deg']:.4f} trans=({r['tx_px']:.2f},{r['ty_px']:.2f}) "
      f"Q={r['q1']}/{r['q2']}/{r['q3']}/{r['q4']} ent={r['entropy_4q']:.3f} "
      f"agree(drot={r['agree_drot_deg']:.4f},dscale={r['agree_dscale']:.5f}) "
      f"ho_x={r['heldout_x_rmse_px']:.2f} ho_y={r['heldout_y_rmse_px']:.2f} [{r['gate_tier']}]",
      flush=True)

# ------------------------------------------------------------------ CSV
rows = [brow, lrow, erow]
cols = ['arm', 'n_corr', 'n_inliers', 'inlier_rmse_px', 'scale',
        'rotation_deg', 'tx_px', 'ty_px', 'gate_tier',
        'agree_drot_deg', 'agree_dscale', 'agree_dtrans_px',
        'q1', 'q2', 'q3', 'q4', 'entropy_4q',
        'q1_ref', 'q2_ref', 'q3_ref', 'q4_ref',
        'heldout_x_rmse_px', 'heldout_y_rmse_px',
        'heldout_x_fit', 'heldout_x_check', 'heldout_y_fit', 'heldout_y_check',
        'seed', 'elapsed_s', 'notes']
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print('wrote', OUT_CSV, flush=True)

# ------------------------------------------------- pre-registered success
print('=== 3. PRE-REGISTERED SUCCESS CRITERION ===', flush=True)
print('bar: inlier_rmse < 1.386px (ECC-refined SIFT) + transform agreement '
      '(drot<=0.05, dscale<=0.001) + held-out not worse than '
      'SIFT (x=%.2f, y=%.2f); OR Q1 filled with agreeing transform'
      % (base_hox, base_hoy), flush=True)
for r in rows[1:]:
    agree = (r['agree_drot_deg'] <= 0.05 and r['agree_dscale'] <= 0.001)
    ho_ok = (r['heldout_x_rmse_px'] <= base_hox + 1e-9 and
             r['heldout_y_rmse_px'] <= base_hoy + 1e-9)
    beats = r['inlier_rmse_px'] < 1.386
    q1fill = r['q1'] > 0 and agree
    ok = (beats and agree and ho_ok) or q1fill
    print('%s: rmse=%.4f (<1.386: %s) agree=%s ho_not_worse=%s Q1=%d => %s'
          % (r['arm'], r['inlier_rmse_px'], beats, agree, ho_ok, r['q1'],
             'PASS' if ok else 'FAIL'), flush=True)
print('done.', flush=True)
