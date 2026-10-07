"""A9: sub-pixel correspondence refinement experiment.

Question: can LK/ECC refinement of SIFT correspondences break the 1.6px
inlier-RMSE floor on the real OHRC ohrc_01 pair?

Protocol:
  0. Reproduce baseline (SIFT 8000 / Lowe 0.75 / RANSAC 3.0px, seed 7).
  1. Synthetic sanity control (exact shift truth): refinement must not
     degrade truth RMSE, else the method is invalid -> STOP.
  2. Real-pair arms: (a) baseline, (b) LK-refined inliers, (c) ECC-refined
     inliers. RANSAC + frozen gate re-run per arm.
  3. Held-out (spatial split) per arm: refinement must not worsen check RMSE.
  4. Pre-registered success: inlier RMSE < 0.50px AND transform agrees with
     baseline (~0.1px / 0.05deg / 0.001 scale) AND held-out not worse.

Refinement details (documented for the write-up):
  LK : cv2.calcOpticalFlowPyrLK(ref, src, prevPts=p1_inl, nextPts=p2_inl init,
       winSize=(21,21), maxLevel=3,
       criteria=(TERM_CRITERIA_EPS|TERM_CRITERIA_COUNT, 40, 0.03),
       flags=cv2.OPTFLOW_USE_INITIAL_FLOW). status==0 -> keep original p2.
  ECC: per-inlier 31x31 patches (ref around p1, src around p2),
       cv2.findTransformECC(ref_patch, src_patch, eye(2,3),
       MOTION_TRANSLATION, criteria=(EPS|COUNT, 50, 1e-4)).
       p2' = p2 + warp[:,2]. On failure/exception or out-of-bounds patch ->
       keep original p2.
Seeds: cv2.setRNGSeed(7) before every RANSAC.
"""
import csv
import time

import cv2
import numpy as np

cv2.setRNGSeed(7)
SEED = 7

REF_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_reference.png'
SRC_P = '/home/hatch/workspace/chandra-align/data/benchmark_crops/ohrc_01_source.png'
OUT_CSV = '/home/hatch/workspace/chandra-align/results/table_subpixel_refinement.csv'

ref = cv2.imread(REF_P, cv2.IMREAD_GRAYSCALE)
src = cv2.imread(SRC_P, cv2.IMREAD_GRAYSCALE)
H, W = ref.shape


def sift_match(a, b):
    sift = cv2.SIFT_create(nfeatures=8000)
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
    inl = inl.ravel().astype(bool)
    return M, inl


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
    # Frozen gates (thresholds only; spatial terms need quadrant counts,
    # reported separately). Returns tier name.
    if rmse <= 0.50 and n_inl >= 8:
        return 'ACCEPT-tier (residual/inliers)'
    if rmse <= 2.50 and n_inl >= 8:
        return 'COARSE-tier (residual/inliers)'
    return 'DEGENERATE-tier (residual/inliers)'


def truth_rmse(M_est, M_true):
    ys, xs = np.mgrid[0.15:0.86:9j, 0.15:0.86:9j]
    g = np.stack([xs.ravel() * W, ys.ravel() * H], axis=1).astype(np.float32)
    e = (M_est @ np.hstack([g, np.ones((len(g), 1))]).T).T
    t = (M_true @ np.hstack([g, np.ones((len(g), 1))]).T).T
    return float(np.sqrt((np.linalg.norm(e - t, axis=1) ** 2).mean()))


def refine_lk(a, b, p1, p2):
    """Sub-pixel LK refinement of p2 given p1. Returns refined p2 + stats."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.03)
    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        a, b, p1.reshape(-1, 1, 2), p2.reshape(-1, 1, 2).copy(),
        winSize=(21, 21), maxLevel=3, criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    st = st.ravel().astype(bool)
    out = p2.copy()
    out[st] = nxt[st].reshape(-1, 2)
    return out, {'n_refined': int(st.sum()), 'n_fallback': int((~st).sum())}


def refine_ecc(a, b, p1, p2, patch=31):
    """Per-patch ECC translation refinement of p2. Returns refined p2 + stats."""
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
        pred = (Mf @ np.hstack([p1[ci], np.ones((len(ci), 1))]).T).T
        chk = float(np.sqrt((np.linalg.norm(pred - p2[ci], axis=1) ** 2).mean()))
        res[name] = (chk, len(fi), len(ci))
    return res


# ---------------------------------------------------------------- baseline
print('=== 0. BASELINE (real pair) ===')
t0 = time.time()
p1b, p2b = sift_match(ref, src)
Mb, inlb = ransac_fit(p1b, p2b)
rmseb, _ = inlier_rmse(Mb, p1b, p2b, inlb)
sb, rb, txb, tyb = decomp(Mb)
print('matches=%d inliers=%d rmse=%.4f scale=%.6f rot=%.4f trans=(%.2f,%.2f) [%s] (%.0fs)'
      % (len(p1b), inlb.sum(), rmseb, sb, rb, txb, tyb,
         gate_verdict(rmseb, inlb.sum()), time.time() - t0))
assert abs(rmseb - 1.6087) < 0.01, 'baseline does not reproduce!'
print('baseline reproduces the documented 1.6087px floor.')

# ------------------------------------------------- 1. synthetic sanity control
print('=== 1. SYNTHETIC CONTROL (exact shift truth 7.3,-3.9) ===')
M_true = np.float32([[1, 0, 7.3], [0, 1, -3.9]])
synth = cv2.warpAffine(ref, M_true, (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)
p1s, p2s = sift_match(ref, synth)
Ms, inls = ransac_fit(p1s, p2s)
trmse_base = truth_rmse(Ms, M_true)
print('synth baseline: matches=%d inliers=%d truth_rmse=%.4f'
      % (len(p1s), inls.sum(), trmse_base))

p1si, p2si = p1s[inls], p2s[inls]
p2_lk, lk_stat = refine_lk(ref, synth, p1si, p2si)
Mlk, inllk = ransac_fit(p1si, p2_lk)
trmse_lk = truth_rmse(Mlk, M_true)
print('synth LK:      refined=%d fallback=%d truth_rmse=%.4f %s'
      % (lk_stat['n_refined'], lk_stat['n_fallback'], trmse_lk,
         'OK' if trmse_lk <= trmse_base + 1e-9 else 'DEGRADED'))

p2_ecc, ecc_stat = refine_ecc(ref, synth, p1si, p2si)
Mec, inlec = ransac_fit(p1si, p2_ecc)
trmse_ecc = truth_rmse(Mec, M_true)
print('synth ECC:     refined=%d fallback=%d truth_rmse=%.4f %s'
      % (ecc_stat['n_refined'], ecc_stat['n_fallback'], trmse_ecc,
         'OK' if trmse_ecc <= trmse_base + 1e-9 else 'DEGRADED'))

if trmse_lk > trmse_base + 1e-9 or trmse_ecc > trmse_base + 1e-9:
    print('SANITY CONTROL FAILED: refinement degrades truth. STOPPING.')
    raise SystemExit(1)
print('sanity control passed: refinement does not hurt on synthetic truth.')

# ------------------------------------------------- 2. real-pair arms
print('=== 2. REAL-PAIR ARMS (refine baseline inliers) ===')
p1i, p2i = p1b[inlb], p2b[inlb]
rows = []
base_ho = heldout(p1b, p2b, inlb)
rows.append(('a_baseline', len(p1b), int(inlb.sum()), rmseb, sb, rb, txb, tyb,
             gate_verdict(rmseb, inlb.sum()), 0.0, 0.0, 0.0,
             base_ho['x'][0], base_ho['y'][0], SEED,
             'unrefined; x-fit=%d/x-check=%d y-fit=%d/y-check=%d'
             % (base_ho['x'][1], base_ho['x'][2], base_ho['y'][1], base_ho['y'][2])))

for name, fn in (('b_LK', refine_lk), ('c_ECC', refine_ecc)):
    t0 = time.time()
    p2r, stat = fn(ref, src, p1i, p2i)
    Mr, inlr = ransac_fit(p1i, p2r)
    rmser, _ = inlier_rmse(Mr, p1i, p2r, inlr)
    sr, rr, txr, tyr = decomp(Mr)
    dtrans = float(np.hypot(txr - txb, tyr - tyb))
    drot = abs(rr - rb)
    dscale = abs(sr - sb)
    ho = heldout(p1i, p2r, inlr)
    rows.append((name, len(p1i), int(inlr.sum()), rmser, sr, rr, txr, tyr,
                 gate_verdict(rmser, inlr.sum()), dtrans, drot, dscale,
                 ho['x'][0], ho['y'][0], SEED,
                 'refined=%d fallback=%d; x-fit=%d/x-check=%d y-fit=%d/y-check=%d (%.0fs)'
                 % (stat['n_refined'], stat['n_fallback'],
                    ho['x'][1], ho['x'][2], ho['y'][1], ho['y'][2],
                    time.time() - t0)))
    print('%s: inliers=%d rmse=%.4f scale=%.6f rot=%.4f trans=(%.2f,%.2f) '
          'agree(dtrans=%.3f,drot=%.4f,dscale=%.5f) heldout_x=%.3f heldout_y=%.3f [%s]'
          % (name, inlr.sum(), rmser, sr, rr, txr, tyr, dtrans, drot, dscale,
             ho['x'][0], ho['y'][0], gate_verdict(rmser, inlr.sum())))

# ------------------------------------------------- 3. CSV
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['arm', 'n_corr', 'n_inliers', 'inlier_rmse_px', 'scale',
                'rotation_deg', 'tx_px', 'ty_px', 'gate_tier',
                'agree_dtrans_px', 'agree_drot_deg', 'agree_dscale',
                'heldout_x_rmse_px', 'heldout_y_rmse_px', 'seed', 'notes'])
    w.writerows(rows)
print('wrote', OUT_CSV)

# ------------------------------------------------- 4. success criterion
print('=== 3. PRE-REGISTERED SUCCESS CRITERION ===')
ok_all = True
for r in rows[1:]:
    name, _, _, rmse, _, _, _, _, _, dtrans, drot, dscale, hox, hoy = r[:14]
    agree = dtrans <= 0.1 and drot <= 0.05 and dscale <= 0.001
    ho_ok = (hox <= base_ho['x'][0] + 1e-9) and (hoy <= base_ho['y'][0] + 1e-9)
    ok = (rmse < 0.50) and agree and ho_ok
    ok_all &= ok
    print('%s: rmse<0.50 %s | transform-agrees %s | heldout-not-worse %s => %s'
          % (name, rmse < 0.50, agree, ho_ok, 'PASS' if ok else 'FAIL'))
print('OVERALL:', 'SUCCESS CRITERION MET' if ok_all else 'NOT MET')
