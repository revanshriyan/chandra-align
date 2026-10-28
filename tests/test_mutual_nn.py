"""Regression test for the _mutual_nn backward-set tuple order.

The backward knnMatch query indices live in des_b and its train indices in
des_a, so the mutual check must compare the forward pair swapped
(trainIdx, queryIdx) against the backward set. Checking the unswapped
pair only passes when the two indices coincide by accident, which made
SIFTMatcher return zero matches on real pairs.
"""

import numpy as np


def test_mutual_nn_returns_pairs_with_different_indices():
    import cv2

    from chandra_align.matcher import _mutual_nn

    # 2-D descriptors chosen so the two mutual pairs have deliberately
    # different indices on each side, and a third descriptor fails Lowe's
    # ratio test (ambiguous second neighbour) and must stay unmatched.
    des_a = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], np.float32)
    des_b = np.array([[30.0, 0.0], [0.2, 0.0], [10.2, 0.0]], np.float32)
    kp_a = [cv2.KeyPoint(1.0, 2.0, 1.0), cv2.KeyPoint(3.0, 4.0, 1.0),
            cv2.KeyPoint(5.0, 6.0, 1.0)]
    kp_b = [cv2.KeyPoint(7.0, 8.0, 1.0), cv2.KeyPoint(9.0, 10.0, 1.0),
            cv2.KeyPoint(11.0, 12.0, 1.0)]

    pts_a, pts_b, matches = _mutual_nn(des_a, des_b, kp_a, kp_b)

    # Mutual pairs are (A0, B1) and (A1, B2): no coincidental index
    # equality anywhere, so the pre-fix check returns zero matches.
    assert len(matches) == 2
    assert pts_a.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert pts_b.tolist() == [[9.0, 10.0], [11.0, 12.0]]
