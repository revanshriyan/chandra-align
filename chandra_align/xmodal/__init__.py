"""Phase 9 — cross-modal (IIRS <-> TMC-2/OHRC) matching toolkit.

Each module is a small, honest instrument: it does one thing and its
docstring states what it does NOT do. Nothing here claims to solve
cross-modal registration on its own; every candidate path runs behind
the same gates (Gate 3 conditioning, inlier/entropy/quadrant tiers)
and is kept only if held-out error improves.

Modules:
    wallis      Wallis local-contrast normalization (preprocessing branch).
    gsd         Common-GSD resampling with explicit unknown-GSD routing.
    tps         Smoothing thin-plate-spline warp (numpy-only solver).
    goa_nmi     Polarity-invariant detection (gradient-magnitude Harris)
                + NMI reranking for cross-modal candidate scoring.
    funnel      Stage-by-stage diagnosis of the detection/descriptor/RANSAC
                funnel (finds WHERE cross-modal matching starves).
    sift_rescue SIFT+Lowe+RANSAC rescue arm feeding a smoothing TPS,
                gated by Gate 3 conditioning and a >=13 inlier floor.
"""

from chandra_align.xmodal import funnel, goa_nmi, gsd, sift_rescue, tps, wallis

__all__ = ["wallis", "gsd", "tps", "goa_nmi", "funnel", "sift_rescue"]
