"""Render Phase 10's pixel verification grid as a heatmap overlay (SVG).

Builds on the shipped chandra_align.trust.area_check. Two panels from real
lunar texture (LROC NAC crop when available, deterministic synthetic scene
otherwise): the TRUE transform (cells verify) vs a WRONG transform (+60 px;
cells go dark) — the trust layer made visible.

Warp convention (matches area_check): M maps A-coords -> B-coords, so the
B image is built as warpAffine(A, invert(M)).

Run: PYTHONPATH=. python3 scripts/trust_map.py
Output: docs/images/trust-map-sample.svg
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from chandra_align.trust.area_check import area_check

BG, INK, MUTED, CARD = "#0b1220", "#e8eef7", "#9fb0c7", "#141d33"
STATE_COLOR = {"verified": "#34d399", "weak": "#fbbf24", "no_evidence": "#f87171"}
REAL_SRC = os.path.expanduser(
    "~/workspace/qa_lunar/real/M181058717LE_real.png")


def _load_scene():
    import cv2
    if os.path.exists(REAL_SRC):
        img = cv2.imread(REAL_SRC, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        print(f"using real LROC NAC crop from {REAL_SRC}")
        return img[h // 2 - 512:h // 2 + 512,
                   w // 2 - 512:w // 2 + 512].astype(np.float32)
    from chandra_align.testing import _base_scene
    print("real source missing; falling back to deterministic synthetic scene")
    return _base_scene(1024, 1024, seed=7).astype(np.float32)


def draw_panel(ax, img, states, grid, title, subtitle):
    h, w = np.asarray(img).shape[:2]
    ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255,
              extent=(0, w, h, 0))
    gh, gw = grid
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * h // gh, (gy + 1) * h // gh
            x0, x1 = gx * w // gw, (gx + 1) * w // gw
            st = states[gy * gw + gx]
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=STATE_COLOR[st], alpha=0.42,
                edgecolor="white", lw=0.8))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(title, color=INK, fontsize=11, weight="bold", pad=8)
    ax.text(w / 2, -14, subtitle, ha="center", va="top", color=MUTED,
            fontsize=9)


def main():
    import cv2
    scene = _load_scene()
    M = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, -6.0]], np.float64)
    M_inv = cv2.invertAffineTransform(M)
    h, w = scene.shape
    # M maps A->B coords: B[q] = A[M^-1 q]
    img_b = cv2.warpAffine(scene, M_inv.astype(np.float32), (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    grid = (6, 6)
    good = area_check(scene, img_b, M, grid=grid)

    M_wrong = M.copy()
    M_wrong[:, 2] += 60.0  # 60 px uniform error: plausible-looking, wrong
    bad = area_check(scene, img_b, M_wrong, grid=grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Per-cell trust map — Phase 10 pixel verification grid",
                 color=INK, fontsize=13, weight="bold", y=0.98)
    fig.text(0.5, 0.93,
             "36 independent cells; green = verified, amber = weak, red = no "
             "evidence. A 60 px-wrong transform is caught cell by cell.",
             ha="center", color=MUTED, fontsize=9)
    for ax in axes:
        ax.set_facecolor(BG)

    draw_panel(axes[0], img_b, good["cell_states"], grid,
               f"TRUE transform — {good['overall'].upper()}",
               f"{good['n_verified']}/36 cells verified")
    draw_panel(axes[1], img_b, bad["cell_states"], grid,
               f"WRONG transform (+60 px) — {bad['overall'].upper()}",
               f"{bad['n_verified']}/36 cells verified")

    for i, (st, col) in enumerate(STATE_COLOR.items()):
        fig.text(0.5 + (i - 1) * 0.16, 0.02, f"\u25a0 {st}",
                 color=col, fontsize=10, ha="center")

    plt.tight_layout(rect=[0, 0.04, 1, 0.9])
    out = "docs/images/trust-map-sample.svg"
    plt.savefig(out)
    print(f"wrote {out}: true={good['overall']} "
          f"({good['n_verified']}/36), wrong={bad['overall']} "
          f"({bad['n_verified']}/36)")


if __name__ == "__main__":
    main()
