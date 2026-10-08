# CrossFeat Trial Plan — 2026-10-08

## What it is
CrossFeat (arXiv 2609.00272, ECCV 2026) learns a "crossing function" in
descriptor space that maps SIFT/ALIKED/SuperPoint/DISK descriptors from one
modality to a representation compatible with another. Geometry-appearance
disentanglement preserves geometric properties while adapting appearance.

Relevant to us: on the paper's satellite cross-modal task, CrossFeat was the
ONLY method with consistent success where dense matchers (MatchAnything,
MINIMA-LoFTR, MINIMA-RoMa) failed. Our IIRS↔TMC-2 (currently 1.31-1.69px
COARSE via LoFTR) is the same problem class.

Code: https://github.com/paulschneider01/CrossFeat (MIT, cloned to
~/workspace/crossfeat/).

## Critical caveat: training required
CrossFeat is NOT a pretrained drop-in. It requires training a
modality-conditioned VAE on ALIGNED image pairs from the two modalities.
The public pipeline:
1. Extract paired RootSIFT descriptors from registered 2D pairs
2. Fit shared PCA projection
3. Train VAE crosser (needs GPU for practical timelines)
4. Evaluate on held-out

## What we have
- 6 IIRS↔TMC-2 windows at COARSE (1.31-1.69px RMSE, 219-826 inliers each)
- These are ROUGHLY aligned but not pixel-perfect (CrossFeat wants "spatially
  aligned and identical dimensions")
- No GPU in this environment (CPU training possible but slow)
- 6 windows is thin for VAE training (paper used larger datasets)

## Trial arm design (frozen-gate discipline)
If pursued, the honest trial:
1. **Data prep**: Use the 6 COARSE windows as pseudo-aligned training pairs.
   Warp IIRS to TMC-2 frame using the recovered affine + deform field.
   Split: 4 train / 1 val / 1 test (thin, but honest about it).
2. **Training**: Train CrossFeat VAE on SIFT descriptors from the pairs.
   Needs GPU or patience (CPU).
3. **Evaluation**: Apply crossed descriptors to the held-out window;
   run through the FROZEN pipeline gate. Win = better than 1.31px COARSE
   baseline, or (ambitious) reaching COARSE with fewer inliers / on a
   harder window.
4. **Integration**: If the trial wins, wire as an opt-in matcher front-end
   (CHANDRA_CROSSFEAT=1), same additive pattern as the deform-field stage.

## Honest blockers
- Training data quality: 1.3-1.7px alignment error in "ground truth" pairs
  will limit what the VAE can learn.
- Data quantity: 6 windows is minimal; more IIRS windows needed for a
  serious attempt.
- Compute: VAE training on CPU will be slow; GPU strongly preferred.
- This is a days-to-weeks project, not a same-day lever.

## Recommendation
Defer to a dedicated sprint with GPU access. The 9-pair scoreboard stands
on validated levers; CrossFeat is the next research-direction bet for the
IIRS cross-modal gap, not an immediate pipeline change.
