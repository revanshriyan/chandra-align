# CrossFeat integration — clean negative (2026-10-15)

## What was tried
Trained a CrossFeat VAE (IIRS<->TMC-2, 19 pairs) on Colab: 100 epochs, no
divergence, train top-1 28.7%, val 27.8% (vs 0.05% chance). Wired as an
opt-in matcher arm (`CHANDRA_CROSSFEAT=1`) following the AnyMatch frontend
pattern: SIFT + RootSIFT, PCA, VAE crossing, mutual-NN + Lowe ratio.

## Result
**Clean negative.** On held-out W5/N13 through `_align_core`:
- W5: 0 matches (ratio 0.8); 3 matches without ratio, all 343–1799 px off
  on an aligned pair. Even at 2048 keypoints: max 8 matches, unusable.
- N13: 0 matches.
- Pre-registered bar (beat 1.31 px on W5): NOT MET.

## Diagnosis
The VAE was trained with paired sampling (descriptors at the *same*
locations in both modalities), so it learned cross-modal *retrieval*
but its descriptors are not discriminative for sparse matching with
*independently detected* keypoints. Training success did not transfer
to the matching task.

## Disposition
Wiring reverted (dead code in the matcher chain). Checkpoint preserved at
`~/workspace/crossfeat_results_2026-10-15/` (best_model.pt + pca.pkl) for
reference. Same class as the A25/MINIMA negatives: training succeeded,
task transfer failed.
