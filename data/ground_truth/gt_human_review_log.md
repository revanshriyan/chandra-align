# Ground-truth landmark human review log
Reviewer: independent visual review | Date: 2026-10-05
Method: each candidate's chip pair (reference left, source right) inspected at full chip-sheet resolution; questionable cells enlarged 3x. Criterion: same physical terrain feature visible in both chips at corresponding positions; no evidence of artifact-driven false match. Generator (Harris+NCC, no LightGlue/SIFT) is independent of the evaluated matchers; visual review is the manual confirmation step.
Result: 40/40 ACCEPTED (20 OHRC_pair, 20 TMC2_fore_nadir). No duplicates. Quadrant spread (ref coords): OHRC LT5/RT3/LB4/RB8; TMC2 LT5/RT3/LB6/RB6 — all >=3.
Caveats: review is at feature-correspondence level, not sub-pixel; GT points carry NCC localization noise (~0.5-1px). OHRC raw data shows detector striping (noted per-point); stripes are peripheral/co-located and terrain features correspond in all cases. TMC-2 fore/nadir are near-simultaneous so large shadows are stable tie features. Pair TMC2_fore_nadir-01 shows zero disparity (sub-pixel in the 640px proposal space); visual match is genuine.
Per-point notes:
- OHRC_pair-04: stripe artifacts peripheral; terrain corresponds
- OHRC_pair-05: horizontal striping co-located in both chips; crater field corresponds
- OHRC_pair-13: striping on left third; terrain features correspond
- OHRC_pair-14: striping on left 40%; dark slope feature corresponds
- OHRC_pair-16: striping on left 30%; dark crater anchor corresponds
- OHRC_pair-20: dense striping left 35%; crater field corresponds
- TMC2_fore_nadir-01: zero disparity (sub-pixel in proposal space); visual match genuine
- TMC2_fore_nadir-02: dark topographic shadow band in both; terrain corresponds
- TMC2_fore_nadir-03: large shadow (stable: fore/nadir near-simultaneous); terrain below corresponds
- TMC2_fore_nadir-12: shadow boundary stable; bright slope features correspond
- TMC2_fore_nadir-17: large shadow (stable); terrain corresponds
All other points: terrain correspondence verified, no anomalies.
