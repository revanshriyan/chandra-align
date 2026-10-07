#!/usr/bin/env python3
"""Worker A18: matcher keypoint-cap sweep.

Question: does raising the per-cell keypoint quota in match_pair_hf let the
deform-field stage engage fully in-pipeline?

Method: import app.py via /tmp/stubs (throwaway gradio/spaces/matplotlib stubs,
never committed), monkeypatch ONLY app.select_detector_keypoints with a
parameterized quota_per_cell. Everything else -- matcher, hook RANSAC, stage,
balance/bucket/NCC chain, frozen gate -- is the true _align_core. The ONLY
difference between runs is the quota (and the CHANDRA_DEFORM_FIELD flag).

Quota values: 64 (current), 128, 256, 10**9 (== uncapped; bounded in practice
by SIFT nfeatures=10000).

Deterministic: seed 7; the pipeline sets its own cv2 seeds internally.
"""
import sys
import os
import time
import resource
import csv

sys.path.insert(0, '/tmp/stubs')
os.environ.pop('CHANDRA_DEFORM_FIELD', None)

import numpy as np
import cv2

cv2.setRNGSeed(7)
np.random.seed(7)

import app
from chandra_align.features import distribution as distmod

ORIG_SELECT = distmod.select_detector_keypoints
CURRENT_QUOTA = {'q': 64}
REC = {'kp_counts': [], 'lowe_matches': None, 'engine': None}

ORIG_MATCH = app.match_pair_hf


def patched_select(kps, shape, quota=64):
    kps_out, idx = ORIG_SELECT(kps, shape, quota_per_cell=CURRENT_QUOTA['q'])
    REC['kp_counts'].append(len(kps_out))
    return kps_out, idx


def patched_match(img1, img2, thr=3.0):
    p1, p2, eng, diag = ORIG_MATCH(img1, img2, thr)
    REC['lowe_matches'] = int(len(p1))
    REC['engine'] = eng
    return p1, p2, eng, diag


app.select_detector_keypoints = patched_select
app.match_pair_hf = patched_match

PAIRS = ['ohrc_01', 'ohrc_02', 'ohrc_03', 'ohrc_05', 'tmc2_04']
QUOTAS = [64, 128, 256, 10**9]
FLAGS = [('off', None), ('on', '1')]

IMGS = {}
for p in PAIRS:
    ref = cv2.imread(f'data/benchmark_crops/{p}_reference.png', cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(f'data/benchmark_crops/{p}_source.png', cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None, p
    IMGS[p] = (ref, sec)


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_one(pair, quota, flag_val):
    CURRENT_QUOTA['q'] = quota
    REC['kp_counts'] = []
    REC['lowe_matches'] = None
    REC['engine'] = None
    if flag_val is None:
        os.environ.pop('CHANDRA_DEFORM_FIELD', None)
    else:
        os.environ['CHANDRA_DEFORM_FIELD'] = flag_val
    ref, sec = IMGS[pair]
    t0 = time.time()
    out = app._align_core(ref, sec)
    el = time.time() - t0
    jm = out['judge_metrics']
    df = jm.get('deform_field_stage') or {}
    tel = out.get('affine_telemetry') or {}
    gm = jm.get('global_metrics') or {}
    kc = REC['kp_counts']
    row = {
        'pair': pair,
        'quota': quota,
        'flag': 'on' if flag_val else 'off',
        'status_code': jm.get('status_code'),
        'n_inliers': gm.get('inlier_count'),
        'rmse_in_sample_px': jm.get('rmse_in_sample_px'),
        'rmse_gate_px': jm.get('rmse_gate_px'),
        'rmse_heldout_px': jm.get('rmse_heldout_px'),
        'gate_basis': jm.get('rmse_gate_basis'),
        'n_kp_ref': kc[0] if len(kc) > 0 else None,
        'n_kp_sec': kc[1] if len(kc) > 1 else None,
        'n_lowe_matches': REC['lowe_matches'],
        'n_hook_ransac_inliers': df.get('n_hook_ransac_inliers'),
        'df_applied': df.get('applied'),
        'df_reason': df.get('reason'),
        'df_lambda': df.get('lambda_chosen'),
        'df_rmse_before_px': df.get('rmse_before_px'),
        'df_rmse_after_px': df.get('rmse_after_px'),
        'df_heldout_px': df.get('heldout_rmse_px'),
        'rot_deg': tel.get('rotation_deg'),
        'scale_s': tel.get('scale_s'),
        'engine': REC['engine'],
        'elapsed_s': round(el, 1),
        'peak_rss_mb': round(peak_rss_mb(), 1),
    }
    return row


def main():
    rows = []
    for pair in PAIRS:
        for quota in QUOTAS:
            for flag_name, flag_val in FLAGS:
                r = run_one(pair, quota, flag_val)
                rows.append(r)
                print('%s quota=%s flag=%s -> %s inl=%s gate=%.4s applied=%s (%.1fs)' % (
                    pair, quota if quota < 10**9 else 'uncapped', flag_name,
                    r['status_code'], r['n_inliers'], r['rmse_gate_px'],
                    r['df_applied'], r['elapsed_s']), flush=True)
    # Determinism re-check: (ohrc_01, 64, off) must reproduce the first row.
    r0 = rows[0]
    rd = run_one('ohrc_01', 64, None)
    det_ok = (rd['status_code'] == r0['status_code']
              and rd['n_inliers'] == r0['n_inliers']
              and abs((rd['rmse_gate_px'] or 0) - (r0['rmse_gate_px'] or 0)) < 1e-9)
    print('determinism re-check (ohrc_01/64/off):', 'PASS' if det_ok else 'FAIL', flush=True)
    assert det_ok, 'non-deterministic pipeline!'
    with open('results/table_cap_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote results/table_cap_sweep.csv (%d rows)' % len(rows), flush=True)


if __name__ == '__main__':
    main()
