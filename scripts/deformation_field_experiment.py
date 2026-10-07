"""A13: regularized dense deformation field on the real OHRC ohrc_01 pair.

Last remaining lever: four matcher families sit on one floor
(SIFT 1.61 / SIFT+ECC 1.39 / LoFTR 1.70 / LoFTR+ECC 1.68px) and A12 proved
the floor is not coverage -- Q1's 2,165 matches obey a coherent DIFFERENT
local transform (884/2165 RANSAC inliers at 1.75px alone; per-quadrant fits
mutually inconsistent). No single partial-affine represents the frame.

Model: global partial-affine + thin-plate-spline residual field
    p2_pred = M @ p1 + TPS(p1)
fit on fit-split points only. TPS implemented directly in numpy
(scipy is ABI-broken in this env); coordinates normalized by /1000,
smoothing = lambda * I added to the kernel diagonal.

Protocol:
  0. Reproduce SIFT baseline (5061/844/1.6087px) or STOP.
  1. Trusted set: ECC-refine the 844 baseline-inlier p2 positions
     (refine_ecc copied from A9; failures keep original).
  2. Smoothness sweep, lambda in pre-registered grid, selected by
     HELD-OUT error only (spatial x/y splits, A9 methodology verbatim).
  3. Killer validation: dense SIFT -> Q1-local RANSAC (expect ~884 @1.75px);
     fit affine+TPS on non-Q1 global inliers; predict the 884 Q1 inliers.
     Global affine gives ~19px there; Q1's own fit gives 1.75px.
  4. Sanity: smooth, folding-free field (max displacement, Jacobian stats).
     Overfit (inlier << held-out with wild displacements) = FAILED approach.

Pre-registered success: held-out improves on BOTH x and y splits vs the
global-affine baseline with a smooth folding-free field, OR the Q1-prediction
lands within a few px of Q1's local 1.75px. Else clean negative.
"""
import csv
import time

import cv2
import numpy as np

cv2.setRNGSeed(7)
SEED = 7

REF_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_reference.png'
SRC_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_source.png'
OUT_CSV = '/home/hatch/workspace/chandra-align/results/table_deformation.csv'

ref = cv2.imread(REF_P, cv2.IMREAD_GRAYSCALE)
src = cv2.imread(SRC_P, cv2.IMREAD_GRAYSCALE)
H, W = ref.shape
print('image %dx%d' % (W, H), flush=True)

TPS_SCALE = 1000.0  # coordinate normalization for the TPS kernel
LAMBDAS = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]  # pre-registered grid


# ---------------------------------------------------------------- utilities
def sift_match(a, b, nfeatures=8000, contrastThreshold=0.04, edgeThreshold=10):
    sift = cv2.SIFT_create(nfeatures=nfeatures,
                           contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, nn in raw if m.distance < 0.75 * nn.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def ransac_fit(p1, p2):
    cv2.setRNGSeed(SEED)
    M, inl = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.995)
    if M is None:
        return None, np.zeros(len(p1), dtype=bool)
    return M, inl.ravel().astype(bool)


def apply_affine(M, p):
    return (M @ np.hstack([p, np.ones((len(p), 1))]).T).T


def rmse(a, b):
    return float(np.sqrt((np.linalg.norm(a - b, axis=1) ** 2).mean()))


def refine_ecc(a, b, p1, p2, patch=31):
    """Per-patch ECC translation refinement of p2. Copied from A9."""
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
            out[i] = p2[i] + warp[:, 2]
            n_ok += 1
        except cv2.error:
            n_fail += 1
    return out, {'n_refined': n_ok, 'n_fallback': n_fail}


def _tps_kernel(d2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d2 > 0, d2 * np.log(d2), 0.0)


def tps_fit(px, py, v, lmbda):
    """Thin-plate spline f(px,py)=v. Coords must be pre-normalized."""
    n = len(px)
    d2 = (px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2
    K = _tps_kernel(d2)
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K + lmbda * np.eye(n)
    A[:n, n] = 1.0
    A[:n, n + 1] = px
    A[:n, n + 2] = py
    A[n, :n] = 1.0
    A[n + 1, :n] = px
    A[n + 2, :n] = py
    b = np.zeros(n + 3)
    b[:n] = v
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:n], sol[n:]  # w, a


def tps_eval(w, a, px, py, qx, qy):
    d2 = (qx[:, None] - px[None, :]) ** 2 + (qy[:, None] - py[None, :]) ** 2
    return _tps_kernel(d2) @ w + a[0] + a[1] * qx + a[2] * qy


def fit_field(p1_fit, r_fit, lmbda):
    px, py = p1_fit[:, 0] / TPS_SCALE, p1_fit[:, 1] / TPS_SCALE
    wx, ax = tps_fit(px, py, r_fit[:, 0], lmbda)
    wy, ay = tps_fit(px, py, r_fit[:, 1], lmbda)
    return (wx, ax, wy, ay, px, py)


def eval_field(M, field, p1):
    wx, ax, wy, ay, px, py = field
    qx, qy = p1[:, 0] / TPS_SCALE, p1[:, 1] / TPS_SCALE
    d = np.stack([tps_eval(wx, ax, px, py, qx, qy),
                  tps_eval(wy, ay, px, py, qx, qy)], axis=1)
    return apply_affine(M, p1) + d


def band_indices(p2, inl, axis):
    v = p2[inl, axis]
    lo, hi = v.min(), v.max()
    fi = np.where(inl & (p2[:, axis] <= lo + 0.40 * (hi - lo)))[0]
    ci = np.where(inl & (p2[:, axis] >= lo + 0.60 * (hi - lo)))[0]
    return fi, ci


def heldout_affine(p1, p2, inl):
    """A9 verbatim: RANSAC affine on 40% band, check on disjoint 40% band."""
    res = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        fi, ci = band_indices(p2, inl, axis)
        if len(fi) < 4 or len(ci) < 1:
            res[name] = (float('nan'), len(fi), len(ci))
            continue
        Mf, _ = ransac_fit(p1[fi], p2[fi])
        if Mf is None:
            res[name] = (float('nan'), len(fi), len(ci))
            continue
        res[name] = (rmse(apply_affine(Mf, p1[ci]), p2[ci]), len(fi), len(ci))
    return res


def heldout_field(p1, p2, inl, lmbda):
    """Same bands as heldout_affine; affine+TPS fit on fit band."""
    res = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        fi, ci = band_indices(p2, inl, axis)
        if len(fi) < 4 or len(ci) < 1:
            res[name] = (float('nan'), len(fi), len(ci))
            continue
        Mf, _ = ransac_fit(p1[fi], p2[fi])
        if Mf is None:
            res[name] = (float('nan'), len(fi), len(ci))
            continue
        r_fit = p2[fi] - apply_affine(Mf, p1[fi])
        field = fit_field(p1[fi].astype(np.float64), r_fit, lmbda)
        pred = eval_field(Mf, field, p1[ci].astype(np.float64))
        res[name] = (rmse(pred, p2[ci]), len(fi), len(ci))
    return res


def field_sanity(M, field, n_grid=32):
    """Smoothness/folding diagnostics on a grid over the frame."""
    gx, gy = np.meshgrid(np.linspace(0, W, n_grid), np.linspace(0, H, n_grid))
    g = np.stack([gx.ravel(), gy.ravel()], axis=1)
    wx, ax, wy, ay, px, py = field
    qx, qy = g[:, 0] / TPS_SCALE, g[:, 1] / TPS_SCALE
    dx = tps_eval(wx, ax, px, py, qx, qy)
    dy = tps_eval(wy, ay, px, py, qx, qy)
    max_disp = float(np.sqrt(dx ** 2 + dy ** 2).max())
    # numerical Jacobian of the displacement via central differences
    hpx = W / (n_grid - 1)
    Dx = dx.reshape(n_grid, n_grid)
    Dy = dy.reshape(n_grid, n_grid)
    dxx = (np.roll(Dx, -1, 1) - np.roll(Dx, 1, 1)) / (2 * hpx)
    dxy = (np.roll(Dx, -1, 0) - np.roll(Dx, 1, 0)) / (2 * hpx)
    dyx = (np.roll(Dy, -1, 1) - np.roll(Dy, 1, 1)) / (2 * hpx)
    dyy = (np.roll(Dy, -1, 0) - np.roll(Dy, 1, 0)) / (2 * hpx)
    grad_fro = np.sqrt(dxx ** 2 + dxy ** 2 + dyx ** 2 + dyy ** 2)
    jac_det = (1 + dxx) * (1 + dyy) - dxy * dyx
    return (max_disp, float(grad_fro[1:-1, 1:-1].max()),
            float(jac_det[1:-1, 1:-1].min()))


rows = []

# ------------------------------------------------------- 0. baseline
print('=== 0. BASELINE ===', flush=True)
p1b, p2b = sift_match(ref, src)
Mb, inlb = ransac_fit(p1b, p2b)
rb = rmse(apply_affine(Mb, p1b[inlb]), p2b[inlb])
print('baseline: matches=%d inliers=%d rmse=%.4f' % (len(p1b), inlb.sum(), rb),
      flush=True)
assert len(p1b) == 5061, 'match count changed: %d' % len(p1b)
assert inlb.sum() == 844, 'inlier count changed: %d' % inlb.sum()
assert abs(rb - 1.6087) < 0.01, 'rmse changed: %.4f' % rb
print('baseline reproduces 5061/844/1.6087px.', flush=True)

# ------------------------------------------------------- 1. trusted set
print('=== 1. ECC-REFINED TRUSTED SET ===', flush=True)
p1t = p1b[inlb]
p2t, ecc_stat = refine_ecc(ref, src, p1t, p2b[inlb])
print('ECC: refined=%d fallback=%d' % (ecc_stat['n_refined'],
                                      ecc_stat['n_fallback']), flush=True)
inl_all = np.ones(len(p1t), dtype=bool)

# ------------------------------------------------------- 2. sweep
print('=== 2. SMOOTHNESS SWEEP (held-out selects) ===', flush=True)
aff_ho = heldout_affine(p1t, p2t, inl_all)
print('affine-only heldout: x=%.3f (fit %d/check %d) y=%.3f (fit %d/check %d)'
      % (aff_ho['x'][0], aff_ho['x'][1], aff_ho['x'][2],
         aff_ho['y'][0], aff_ho['y'][1], aff_ho['y'][2]), flush=True)
rows.append(('affine_baseline', -1, len(p1t),
             rmse(apply_affine(Mb, p1t), p2t),
             aff_ho['x'][0], aff_ho['y'][0],
             float('nan'), float('nan'), float('nan'), float('nan'),
             'A9-methodology affine on ECC-refined trusted set'))

best, best_key = None, None
for lam in LAMBDAS:
    t0 = time.time()
    fho = heldout_field(p1t, p2t, inl_all, lam)
    # inlier fit (overfit diagnostic): fit on all, eval on all
    Mall, _ = ransac_fit(p1t, p2t)
    r_all = p2t - apply_affine(Mall, p1t)
    field_all = fit_field(p1t.astype(np.float64), r_all, lam)
    irmse = rmse(eval_field(Mall, field_all, p1t.astype(np.float64)), p2t)
    hx, hy = fho['x'][0], fho['y'][0]
    key = hx + hy
    tag = ''
    if best is None or key < best_key:
        best, best_key = lam, key
        tag = ' <- best so far'
    print('lambda=%g: inlier=%.4f heldout_x=%.3f heldout_y=%.3f (%.0fs)%s'
          % (lam, irmse, hx, hy, time.time() - t0, tag), flush=True)
    rows.append(('sweep', lam, len(p1t), irmse, hx, hy,
                 float('nan'), float('nan'), float('nan'), float('nan'),
                 'x-fit=%d/x-check=%d y-fit=%d/y-check=%d'
                 % (fho['x'][1], fho['x'][2], fho['y'][1], fho['y'][2])))

print('selected lambda* = %g by min heldout_x+heldout_y' % best, flush=True)
# sanity of the selected field (fit on all trusted points)
Mall, _ = ransac_fit(p1t, p2t)
r_all = p2t - apply_affine(Mall, p1t)
field_star = fit_field(p1t.astype(np.float64), r_all, best)
max_disp, max_grad, min_det = field_sanity(Mall, field_star)
irmse_star = rmse(eval_field(Mall, field_star, p1t.astype(np.float64)), p2t)
fho_star = heldout_field(p1t, p2t, inl_all, best)
print('lambda*: inlier_rmse=%.4f heldout_x=%.3f heldout_y=%.3f '
      'max_disp=%.2fpx max_grad=%.4f min_jacdet=%.4f'
      % (irmse_star, fho_star['x'][0], fho_star['y'][0],
         max_disp, max_grad, min_det), flush=True)
rows.append(('selected', best, len(p1t), irmse_star,
             fho_star['x'][0], fho_star['y'][0],
             float('nan'), max_disp, max_grad, min_det,
             'smoothness selected by held-out; sanity on full fit'))

# ------------------------------------------------------- 3. killer test
print('=== 3. KILLER VALIDATION (fit Q2+Q3+Q4, predict Q1) ===', flush=True)
p1d, p2d = sift_match(ref, src, nfeatures=20000,
                      contrastThreshold=0.01, edgeThreshold=20)
print('dense matches: %d' % len(p1d), flush=True)
Md, inld = ransac_fit(p1d, p2d)
print('dense global: inliers=%d rmse=%.4f'
      % (inld.sum(), rmse(apply_affine(Md, p1d[inld]), p2d[inld])), flush=True)
q1mask = (p2d[:, 0] < W / 2.0) & (p2d[:, 1] < H / 2.0)  # A12 convention: p2 coords
print('Q1 Lowe matches: %d' % q1mask.sum(), flush=True)
Mq1, inl_q1 = ransac_fit(p1d[q1mask], p2d[q1mask])
n_q1 = int(inl_q1.sum())
rmse_q1 = rmse(apply_affine(Mq1, p1d[q1mask][inl_q1]), p2d[q1mask][inl_q1])
print('Q1-local: inliers=%d rmse=%.4f (expect ~884 @ ~1.75px)'
      % (n_q1, rmse_q1), flush=True)
assert inld[q1mask].sum() == 0, 'dense global has Q1 inliers?!'
assert 800 <= n_q1 <= 970 and abs(rmse_q1 - 1.75) < 0.15, \
    'Q1-local solution changed: %d @ %.3f' % (n_q1, rmse_q1)

fitmask = inld & ~q1mask
p1f, p2f = p1d[fitmask].astype(np.float64), p2d[fitmask].astype(np.float64)
p1q = p1d[q1mask][inl_q1].astype(np.float64)
p2q = p2d[q1mask][inl_q1].astype(np.float64)
print('field fit points (non-Q1 global inliers): %d; Q1 test points: %d'
      % (fitmask.sum(), len(p1q)), flush=True)

Mf, _ = ransac_fit(p1f.astype(np.float32), p2f.astype(np.float32))
base_q1 = rmse(apply_affine(Mf, p1q), p2q)
med_q1 = float(np.median(np.linalg.norm(apply_affine(Mf, p1q) - p2q, axis=1)))
print('global-affine on 884 Q1 inliers: rmse=%.2f median=%.2f '
      '(expect ~19px median)' % (base_q1, med_q1), flush=True)
rows.append(('killer_affine', -1, int(fitmask.sum()), float('nan'),
             float('nan'), float('nan'), base_q1,
             float('nan'), float('nan'), float('nan'),
             'global affine (fit Q2+Q3+Q4) predicting 884 Q1 inliers; median=%.2f'
             % med_q1))
rows.append(('killer_local', -1, n_q1, rmse_q1,
             float('nan'), float('nan'), float('nan'),
             float('nan'), float('nan'), float('nan'),
             'Q1-local RANSAC fit: the target to approach'))

for lam in LAMBDAS:
    t0 = time.time()
    r_fit = p2f - apply_affine(Mf, p1f)
    field = fit_field(p1f, r_fit, lam)
    pred = eval_field(Mf, field, p1q)
    e = rmse(pred, p2q)
    med = float(np.median(np.linalg.norm(pred - p2q, axis=1)))
    max_disp, max_grad, min_det = field_sanity(Mf, field)
    print('killer lambda=%g: Q1 pred rmse=%.3f median=%.3f '
          'max_disp=%.1f max_grad=%.4f min_det=%.4f (%.0fs)'
          % (lam, e, med, max_disp, max_grad, min_det, time.time() - t0),
          flush=True)
    rows.append(('killer', lam, int(fitmask.sum()), float('nan'),
                 float('nan'), float('nan'), e,
                 max_disp, max_grad, min_det,
                 'median=%.3f' % med))

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['experiment', 'lambda', 'n_fit', 'inlier_rmse_px',
                'heldout_x_px', 'heldout_y_px', 'q1_pred_rmse_px',
                'max_disp_px', 'max_jac_grad', 'min_jac_det', 'notes'])
    w.writerows(rows)
print('wrote', OUT_CSV, flush=True)
