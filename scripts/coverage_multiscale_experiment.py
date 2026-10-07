"""A12: spatial-coverage experiment — attack the empty Q1 quadrant.

Every arm on the real OHRC ohrc_01 pair has ZERO inliers in Q1 (top-left;
baseline counts 0/411/45/388). A quarter of the frame is pure extrapolation
and held-out error (6.14/4.73px) dwarfs inlier RMSE (1.61px). Default SIFT
finds nothing matchable in Q1; this script forces detection:

  arm a_baseline : SIFT(nfeatures=8000) / Lowe 0.75 / RANSAC 3.0px, seed 7
                   (must reproduce 5061/844/1.6087px/Q1=0 or STOP)
  arm b_multiscale: Gaussian pyramid at 0.5x/1x/2x, SIFT per level,
                   coordinates mapped back to full-res, merged + deduped,
                   single RANSAC
  arm c_dense     : SIFT(nfeatures=20000, contrastThreshold=0.01,
                   edgeThreshold=20) — force keypoints in texture-poor areas
  arm d_combined  : pyramid 0.5x/1x/2x with dense params per level
                   (runs only if b and c both complete without error)

Pre-registered success: Q1 gains inliers AND transform agrees with baseline
(|drot|<=0.05deg, |dscale|<=0.001) AND held-out improves on >=1 split vs
baseline without worsening the other by >0.5px. Coverage without geometric
agreement is decoration: arms that fill Q1 but disagree with baseline or
worsen held-out are reported as FAILED, never wins.

Quadrants/entropy use the repo's own chandra_align.metrics.quadrant
convention (Q1=Top-Left, entropy=-sum(p*log2(p))). Frozen gates:
ACCEPT: rmse<=0.50, >=8 inliers, entropy>=0.75, >=3 quads.
COARSE: rmse<=2.50, >=8 inliers, entropy>=0.50, >=2 quads.
Else DEGENERATE_FAILURE. Gates never touched.
"""
import csv
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, '/home/hatch/workspace/chandra-align')
from chandra_align.metrics.quadrant import compute_quadrant_metrics

cv2.setRNGSeed(7)
SEED = 7

REF_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_reference.png'
SRC_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_source.png'
OUT_CSV = '/home/hatch/workspace/chandra-align/results/table_coverage.csv'

ref = cv2.imread(REF_P, cv2.IMREAD_GRAYSCALE)
src = cv2.imread(SRC_P, cv2.IMREAD_GRAYSCALE)
H, W = ref.shape
print('images:', ref.shape, src.shape, flush=True)


# ------------------------------------------------------------------ helpers
def sift_match(a, b, nfeatures=8000, contrastThreshold=0.04, edgeThreshold=10,
               lowe=0.75):
    sift = cv2.SIFT_create(nfeatures=nfeatures,
                           contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                len(k1) if k1 is not None else 0,
                len(k2) if k2 is not None else 0)
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, nn in raw if m.distance < lowe * nn.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2, len(k1), len(k2)


def multiscale_match(a, b, scales=(0.5, 1.0, 2.0), nfeatures=8000,
                     contrastThreshold=0.04, edgeThreshold=10, lowe=0.75):
    """Detect/match SIFT per pyramid level, map coords back to full-res,
    merge and dedupe (<1px) before RANSAC."""
    P1, P2 = [], []
    nkp = [0, 0]
    for s in scales:
        ws, hs = int(round(W * s)), int(round(H * s))
        a_s = cv2.resize(a, (ws, hs), interpolation=cv2.INTER_AREA
                         if s < 1.0 else cv2.INTER_LINEAR)
        b_s = cv2.resize(b, (ws, hs), interpolation=cv2.INTER_AREA
                         if s < 1.0 else cv2.INTER_LINEAR)
        p1s, p2s, n1, n2 = sift_match(a_s, b_s, nfeatures=nfeatures,
                                     contrastThreshold=contrastThreshold,
                                     edgeThreshold=edgeThreshold, lowe=lowe)
        nkp[0] += n1
        nkp[1] += n2
        if len(p1s):
            P1.append(p1s / s)
            P2.append(p2s / s)
    if not P1:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                nkp[0], nkp[1])
    P1 = np.vstack(P1)
    P2 = np.vstack(P2)
    # dedupe near-identical correspondences (<1px in both endpoints)
    key = np.round(np.hstack([P1, P2])).astype(np.int64)
    seen, keep = set(), []
    for i, k in enumerate(map(tuple, key)):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    keep = np.array(keep)
    return P1[keep], P2[keep], nkp[0], nkp[1]


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


def quadrants(p2_inl, r_inl):
    quads, entropy = compute_quadrant_metrics(p2_inl, r_inl, (H, W))
    counts = [quads['Q1']['inlier_count'], quads['Q2']['inlier_count'],
              quads['Q3']['inlier_count'], quads['Q4']['inlier_count']]
    return counts, float(entropy)


def gate_tier(rmse, n_inl, entropy, n_quads):
    if rmse <= 0.50 and n_inl >= 8 and entropy >= 0.75 and n_quads >= 3:
        return 'SUCCESS_SUBPIXEL'
    if rmse <= 2.50 and n_inl >= 8 and entropy >= 0.50 and n_quads >= 2:
        return 'COARSE_ADVISORY'
    return 'DEGENERATE_FAILURE'


def heldout(p1, p2, inl):
    """Spatial x/y split: RANSAC-fit on 40% band, check on disjoint 40% band
    of inlier p2 coordinates. Identical procedure to the A9 baseline."""
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


# ------------------------------------------------------------------ baseline
print('=== 0. BASELINE ===', flush=True)
t0 = time.time()
p1b, p2b, nk1b, nk2b = sift_match(ref, src)
Mb, inlb = ransac_fit(p1b, p2b)
rmseb, rb_all = inlier_rmse(Mb, p1b, p2b, inlb)
sb, rotb, txb, tyb = decomp(Mb)
qb, eb = quadrants(p2b[inlb], rb_all[inlb])
hob = heldout(p1b, p2b, inlb)
tierb = gate_tier(rmseb, int(inlb.sum()), eb, sum(c > 0 for c in qb))
print('baseline: matches=%d inliers=%d rmse=%.4f quads=%s entropy=%.4f '
      'heldout_x=%.3f heldout_y=%.3f [%s] (%.0fs)'
      % (len(p1b), inlb.sum(), rmseb, qb, eb, hob['x'][0], hob['y'][0],
         tierb, time.time() - t0), flush=True)
assert len(p1b) == 5061, 'baseline matches != 5061'
assert int(inlb.sum()) == 844, 'baseline inliers != 844'
assert abs(rmseb - 1.6087) < 0.01, 'baseline rmse != 1.6087'
assert qb[0] == 0, 'baseline Q1 is not empty?!'
print('baseline reproduces 5061/844/1.6087px/Q1=0.', flush=True)

# ------------------------------------------------------------------ arms
ARMS = [
    ('b_multiscale', 'multiscale',
     dict(scales=(0.5, 1.0, 2.0), nfeatures=8000,
          contrastThreshold=0.04, edgeThreshold=10)),
    ('c_dense', 'dense',
     dict(nfeatures=20000, contrastThreshold=0.01, edgeThreshold=20)),
    ('d_combined', 'multiscale',
     dict(scales=(0.5, 1.0, 2.0), nfeatures=12000,
          contrastThreshold=0.02, edgeThreshold=15)),
]

rows = []
arm_ok = {}
for arm_name, kind, kw in ARMS:
    if arm_name == 'd_combined' and not (arm_ok.get('b_multiscale')
                                        and arm_ok.get('c_dense')):
        print('SKIP d_combined: b or c did not run clean.', flush=True)
        continue
    print('=== arm %s ===' % arm_name, flush=True)
    t0 = time.time()
    try:
        if kind == 'multiscale':
            p1, p2, nk1, nk2 = multiscale_match(ref, src, **kw)
        else:
            p1, p2, nk1, nk2 = sift_match(ref, src, **kw)
        if len(p1) < 8:
            raise RuntimeError('only %d correspondences (<8)' % len(p1))
        M, inl = ransac_fit(p1, p2)
        if M is None or inl.sum() < 1:
            raise RuntimeError('RANSAC degenerate: no inliers')
        rmse, r_all = inlier_rmse(M, p1, p2, inl)
        s, rot, tx, ty = decomp(M)
        q, e = quadrants(p2[inl], r_all[inl])
        n_quads = sum(c > 0 for c in q)
        tier = gate_tier(rmse, int(inl.sum()), e, n_quads)
        dtrans = float(np.hypot(tx - txb, ty - tyb))
        drot = abs(rot - rotb)
        dscale = abs(s - sb)
        ho = heldout(p1, p2, inl)
        agree = (drot <= 0.05) and (dscale <= 0.001)
        # pre-registered coverage-win check
        q1_gain = q[0] > 0
        ho_better = ((ho['x'][0] < hob['x'][0] - 1e-9)
                     or (ho['y'][0] < hob['y'][0] - 1e-9))
        ho_not_worse = ((ho['x'][0] <= hob['x'][0] + 0.5)
                        and (ho['y'][0] <= hob['y'][0] + 0.5))
        win = q1_gain and agree and ho_better and ho_not_worse
        rows.append([arm_name, len(p1), int(inl.sum()), rmse, s, rot, tx, ty,
                     tier, q[0], q[1], q[2], q[3], e,
                     drot, dscale, dtrans,
                     ho['x'][0], ho['y'][0],
                     ho['x'][1], ho['x'][2], ho['y'][1], ho['y'][2],
                     SEED, nk1, nk2,
                     'kp_ref=%d kp_src=%d; x-fit=%d/x-check=%d y-fit=%d/y-check=%d; '
                     'Q1_gain=%s agree=%s => %s (%.0fs)'
                     % (nk1, nk2, ho['x'][1], ho['x'][2], ho['y'][1],
                        ho['y'][2], q1_gain, agree,
                        'COVERAGE_WIN' if win else 'NOT_A_WIN',
                        time.time() - t0)])
        arm_ok[arm_name] = True
        print('%s: corr=%d inliers=%d rmse=%.4f quads=%s entropy=%.4f '
              'scale=%.6f rot=%.4f agree(drot=%.4f,dscale=%.5f) '
              'heldout_x=%.3f heldout_y=%.3f [%s] => %s (%.0fs)'
              % (arm_name, len(p1), inl.sum(), rmse, q, e, s, rot,
                 drot, dscale, ho['x'][0], ho['y'][0], tier,
                 'COVERAGE_WIN' if win else 'NOT_A_WIN', time.time() - t0),
              flush=True)
    except Exception as ex:
        arm_ok[arm_name] = False
        rows.append([arm_name, 0, 0, float('nan'), float('nan'),
                     float('nan'), float('nan'), float('nan'),
                     'DEGENERATE_FAILURE', 0, 0, 0, 0, 0.0,
                     float('nan'), float('nan'), float('nan'),
                     float('nan'), float('nan'), 0, 0, 0, 0,
                     SEED, 0, 0, 'ERROR: %s: %s'
                     % (type(ex).__name__, str(ex)[:200])])
        print('%s: ERROR %s: %s' % (arm_name, type(ex).__name__, ex),
              flush=True)
        traceback.print_exc(limit=3)

# ------------------------------------------------------------------ CSV
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['arm', 'n_corr', 'n_inliers', 'inlier_rmse_px', 'scale',
                'rotation_deg', 'tx_px', 'ty_px', 'gate_tier',
                'q1', 'q2', 'q3', 'q4', 'entropy_4q',
                'agree_drot_deg', 'agree_dscale', 'agree_dtrans_px',
                'heldout_x_rmse_px', 'heldout_y_rmse_px',
                'heldout_x_fit', 'heldout_x_check',
                'heldout_y_fit', 'heldout_y_check',
                'seed', 'n_kp_ref', 'n_kp_src', 'notes'])
    w.writerows(rows)
print('wrote', OUT_CSV, flush=True)

# ------------------------------------------------------------------ verdict
print('=== PRE-REGISTERED SUCCESS CRITERION ===', flush=True)
any_win = False
for r in rows:
    win = 'COVERAGE_WIN' in r[-1]
    any_win |= win
    print('%s: %s' % (r[0], 'COVERAGE_WIN' if win else 'not a win'), flush=True)
print('OVERALL:', 'COVERAGE WIN ACHIEVED' if any_win else 'NO COVERAGE WIN',
      flush=True)
