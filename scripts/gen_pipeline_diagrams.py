"""Generate the committed pipeline workflow diagrams (docs/images/pipeline-*.png).

Run: python scripts/gen_pipeline_diagrams.py  (needs matplotlib)
Diagrams are drawn from the implemented code (app.py::match_pair_hf,
chandra_align/metrics/quadrant.py::validate_registration_gate), not from
aspirational blocks. Re-run after pipeline changes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "docs/images"
BG = "#0b1220"
INK = "#e8eef7"
MUTED = "#9fb0c7"
ACCENT = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
CARD = "#141d33"


def canvas(w=11, h=6.5, title="", subtitle=""):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, title, ha="center", va="center", fontsize=15,
            color=INK, weight="bold")
    if subtitle:
        ax.text(50, 90.5, subtitle, ha="center", va="center", fontsize=9, color=MUTED)
    return fig, ax


def box(ax, x, y, w, h, text, color=ACCENT, fs=8.5, textcolor=INK):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.4,rounding_size=1.2",
                           facecolor=CARD, edgecolor=color, linewidth=1.6)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=textcolor, weight="bold", linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, label="", color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 color=color, linewidth=1.4,
                                 mutation_scale=12, shrinkA=2, shrinkB=4))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.2, label, ha="center",
                fontsize=7.5, color=color, style="italic")


def cascade():
    fig, ax = canvas(title="CHANDRA-ALIGN — Matcher Cascade",
                     subtitle="app.py :: match_pair_hf — implemented order, fail-forward with diagnostics")
    steps = [
        (50, 80, 44, 11, "Input pair\n(uint8, 1–99% stretch)", ACCENT),
        (50, 62, 52, 11, "1  LightGlue / ALIKED  (GPU primary)\n≥ 4 correspondences?", GREEN),
        (50, 44, 52, 11, "2  RIFT2 — OPT-IN ONLY\nCHANDRA_ENABLE_RIFT2=1  ·  ≥ 4 correspondences?", AMBER),
        (50, 26, 52, 11, "3  SIFT + Brute-Force RANSAC  (CPU fallback)\nnfeatures=10000 · RANSAC-verified candidate", ACCENT),
        (50, 10, 44, 9, "Correspondences → RANSAC fit → Gate", INK),
    ]
    for x, y, w, h, t, c in steps:
        box(ax, x, y, w, h, t, color=c, fs=8)
    arrow(ax, 50, 74.5, 50, 67.5)
    arrow(ax, 50, 56.5, 50, 49.5, label="fewer than 4 / exception")
    arrow(ax, 50, 38.5, 50, 31.5, label="opt-in off or < 4")
    arrow(ax, 50, 20.5, 50, 14.5)
    ax.text(88, 62, "✓ use", color=GREEN, fontsize=8, weight="bold", va="center")
    ax.text(88, 44, "✓ use", color=GREEN, fontsize=8, weight="bold", va="center")
    for yy in (62, 44):
        ax.add_patch(FancyArrowPatch((76, yy), (84, yy), arrowstyle="-|>",
                                     color=GREEN, linewidth=1.4, mutation_scale=12))
    fig.savefig(f"{OUT}/pipeline-cascade.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)


def gates():
    fig, ax = canvas(w=12, h=6.5, title="CHANDRA-ALIGN — Fail-Closed Spatial Gate",
                     subtitle="chandra_align/metrics/quadrant.py :: validate_registration_gate — non-finite metrics fail closed")
    box(ax, 50, 82, 60, 10,
        "RANSAC fit: RMSE · inliers (≥ 8) · entropy · quadrant spread", ACCENT, fs=8.5)
    box(ax, 22, 58, 36, 20,
        "ACCEPT\nSub-pixel precision\n\nRMSE ≤ 0.50 px\ninliers ≥ 8\nentropy ≥ 0.75\n≥ 3 / 4 quadrants",
        GREEN, fs=8)
    box(ax, 50, 58, 36, 20,
        "COARSE ADVISORY\nRegional fit — not sub-pixel\n\nRMSE ≤ 2.50 px\ninliers ≥ 8\nentropy ≥ 0.50\n≥ 2 / 4 quadrants",
        AMBER, fs=8)
    box(ax, 80, 58, 32, 20,
        "REJECT\nDEGENERATE_FAILURE\n\nanything else fails\ntelemetry masked\nreasons reported",
        RED, fs=8)
    arrow(ax, 38, 77, 26, 68)
    arrow(ax, 50, 77, 50, 68)
    arrow(ax, 62, 77, 74, 68)
    box(ax, 50, 30, 88, 16,
        "Measured outcomes — synthetic: ACCEPT 0.37 px · real OHRC/TMC-2: COARSE 1.5–2.2 px · "
        "IIRS↔TMC-2: DEGENERATE_FAILURE · RIFT2: REJECTED (0–1 corr)",
        INK, fs=7.5)
    arrow(ax, 22, 48, 22, 38)
    arrow(ax, 50, 48, 50, 38)
    arrow(ax, 80, 48, 80, 38)
    ax.text(50, 12, "Gate verdicts are final: confidence scores never override them.",
            ha="center", fontsize=8, color=MUTED, style="italic")
    fig.savefig(f"{OUT}/pipeline-gates.png", dpi=100, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)


def overview():
    fig, ax = canvas(w=12, h=6.5, title="CHANDRA-ALIGN — End-to-End Data Flow",
                     subtitle="from Chandrayaan-2 products to verdict + exports")
    stages = [
        (12, 60, 20, 14, "PDS4 products\nOHRC · TMC-2 · IIRS\n+ XML labels", ACCENT),
        (36, 60, 20, 14, "Preprocess\ncrop / window\nuint8 stretch", ACCENT),
        (60, 60, 20, 14, "Matcher cascade\nLG/ALIKED → SIFT\n(RIFT2 opt-in)", GREEN),
        (84, 60, 20, 14, "RANSAC fit\npartial-affine\nrefit", ACCENT),
        (24, 30, 22, 14, "Fail-closed gate\nACCEPT / COARSE\n/ REJECT", AMBER),
        (50, 30, 22, 14, "Telemetry\nRMSE · MAE · entropy\nquadrants · confidence", ACCENT),
        (78, 30, 24, 14, "Exports\nCSV tables · overlays\nmasked on reject", INK),
    ]
    for x, y, w, h, t, c in stages:
        box(ax, x, y, w, h, t, color=c, fs=8)
    for (x1, _, _, _, _, _), (x2, _, _, _, _, _) in zip(stages[:3], stages[1:4]):
        arrow(ax, x1 + 10, 60, x2 - 10, 60)
    arrow(ax, 84, 53, 61, 37)
    arrow(ax, 35, 30, 39, 30)
    arrow(ax, 61, 30, 67, 30)
    ax.text(50, 12, "Every number on this page is measured — see results/table_canonical_v1.csv",
            ha="center", fontsize=8, color=MUTED, style="italic")
    fig.savefig(f"{OUT}/pipeline-overview.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    cascade()
    gates()
    overview()
    print("diagrams written to", OUT)
