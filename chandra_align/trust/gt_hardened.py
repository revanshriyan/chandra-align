"""Phase 10 — ground-truth hardening.

Wraps human/independent ground-truth point sets with structural assertions
so a GT evaluation cannot silently misuse them:

- pair + direction assertions (which image pair, which mapping direction)
- native-pixel declarations (coordinates live in the stated rasters' pixels)
- used_for_fitting=False asserted (GT never trains the transform)
- sha256 provenance of the GT file baked into every report
- per-point feature descriptions carried through
- bidirectional residuals: forward (src->ref via M) AND inverse
  (ref->src via M^-1); both reported, neither hidden.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HardenedGT:
    """A ground-truth point set with asserted provenance and direction."""

    def __init__(self, path, pair_id, direction="src_to_ref",
                 native_pixels=True, used_for_fitting=False,
                 descriptions=None):
        self.path = Path(path)
        self.pair_id = pair_id
        assert direction in ("src_to_ref", "ref_to_src"), \
            f"direction must be src_to_ref|ref_to_src, got {direction}"
        self.direction = direction
        self.native_pixels = bool(native_pixels)
        assert used_for_fitting is False, \
            "ground truth must never be used for fitting"
        self.used_for_fitting = False
        self.sha256 = sha256_of(self.path)
        self.ids, self.ref, self.src, self.desc = self._read(descriptions)

    def _read(self, descriptions):
        with self.path.open("r", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{self.path} has no rows")
        ids = [r["id"].strip() for r in rows]
        if len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError("GT ids must be unique and non-empty")
        ref = np.array([[float(r["x_ref"]), float(r["y_ref"])] for r in rows])
        src = np.array([[float(r["x_src"]), float(r["y_src"])] for r in rows])
        if not (np.isfinite(ref).all() and np.isfinite(src).all()):
            raise ValueError("GT coordinates must be finite")
        desc = {}
        for r, i in zip(rows, ids):
            d = (descriptions or {}).get(i, r.get("description", ""))
            desc[i] = d.strip()
        return ids, ref, src, desc

    def bidirectional_residuals(self, M):
        """Forward (src->ref via M) and inverse (ref->src via M^-1) RMSE.

        M must map src->ref when direction == 'src_to_ref'.
        Returns dict with rmse_forward_px, rmse_inverse_px, n_points,
        plus per-point residuals for inspection.
        """
        M = np.asarray(M, np.float64).reshape(2, 3)
        if self.direction == "ref_to_src":
            M = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2, :]
        fwd_pred = self.src @ M[:, :2].T + M[:, 2]
        fwd_res = np.hypot(fwd_pred[:, 0] - self.ref[:, 0],
                           fwd_pred[:, 1] - self.ref[:, 1])
        M_inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2, :]
        inv_pred = self.ref @ M_inv[:, :2].T + M_inv[:, 2]
        inv_res = np.hypot(inv_pred[:, 0] - self.src[:, 0],
                           inv_pred[:, 1] - self.src[:, 1])
        return {
            "pair_id": self.pair_id,
            "direction": self.direction,
            "native_pixels": self.native_pixels,
            "used_for_fitting": self.used_for_fitting,
            "sha256": self.sha256,
            "n_points": int(len(self.ids)),
            "rmse_forward_px": float(np.sqrt((fwd_res ** 2).mean())),
            "rmse_inverse_px": float(np.sqrt((inv_res ** 2).mean())),
            "per_point": [
                {"id": i, "fwd_res_px": float(fr), "inv_res_px": float(ir),
                 "description": self.desc[i]}
                for i, fr, ir in zip(self.ids, fwd_res, inv_res)
            ],
        }
