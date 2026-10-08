# Cross-Modal Alternatives to CrossFeat Training — 2026-10-08

Research into IIRS↔TMC-2 registration approaches beyond CrossFeat VAE training.
Current best: 1.31px COARSE via LoFTR+LK (6 windows, frozen gates).
CrossFeat status: code cloned (MIT), needs VAE training on aligned pairs —
see docs/crossfeat-trial-plan-2026-10-08.md.

This doc covers what the SOTA brief does NOT: methods needing no aligned
training pairs, synthetic data generation, non-VAE architectures, and hybrids.
MINIMA/GIM/RoMa are in the SOTA brief and not duplicated here.

---

## 1. Methods that DON'T need aligned training pairs

### 1a. MatchAnything — pretrained cross-modal matcher (weights available NOW)

- **What:** Universal cross-modality matching via large-scale pretraining
  (He et al., TPAMI 2026, arXiv:2501.07556). Single weight handles all
  cross-modal tasks without fine-tuning. DINOv2 backbone + LoFTR-style
  matcher, ELoFTR and RoMa variants.
- **Availability:** Code at github.com/zju3dv/matchanything; pretrained
  weights on HuggingFace (LittleFrog/MatchAnything space) and Google Drive.
  Inference code ready. No training needed.
- **Why it could help:** Zero-training-cost candidate arm. Drop into the
  existing harness behind frozen gates, same as the LoFTR arm.
- **Honest caveats:** The CrossFeat paper's satellite-task evaluation found
  MatchAnything (along with MINIMA variants) STRUGGLED to produce reliable
  correspondences where CrossFeat succeeded. This is a real negative signal
  on our exact problem class. Still worth a measured trial (an afternoon),
  but expect it may fail — report the result either way.
- **Cost to try:** An afternoon (download weights, adapt harness, run 6 windows).

### 1b. HOPC / CFOG — handcrafted cross-modal descriptors (code available)

- **What:** Histogram of Orientated Phase Congruency (HOPC, Ye et al.) and
  Channel Features of Orientated Gradients (CFOG). Dense structural
  descriptors invariant to nonlinear radiometric differences. HOPC_ncc uses
  NCC on phase-congruency orientation histograms; CFOG is a faster
  pixel-wise HOG extension with intensity-inversion handling.
- **Availability:** CFOG/HOPC code published by Ye et al. (referenced in
  their papers; implementation exists in the remote-sensing community).
  RIFT (related, PC-based) is already vendored in our repo.
- **Why it could help:** Our Phase-14 already does phase-congruency
  front-end work. HOPC is the natural dense-descriptor complement —
  it captures the same structural invariance but as a matchable descriptor
  rather than just a detector. CFOG specifically handles the intensity
  inversion between modalities (bright-in-one, dark-in-other), which is
  exactly the IIRS↔TMC-2 appearance gap.
- **Honest caveats:** Handcrafted methods plateau below learned methods on
  hard pairs; the MCCR paper (2026) corroborates PC+CFOG+TPS as a coherent
  recipe but on water surfaces, not lunar. May not beat LoFTR's 1.31px,
  but could provide a diverse second opinion for the gate.
- **Cost to try:** 1-2 days (implement CFOG descriptor, plug into matching
  harness, run 6 windows).

### 1c. Domain adaptation without pairs — UML paradigm

- **What:** "Better Together" (arXiv:2510.08492, Oct 2025) shows unpaired
  auxiliary-modality data improves unimodal representations via shared-parameter
  training. For us: train on unpaired IIRS + unpaired TMC-2 with a shared
  backbone, no cross-modal pairs needed.
- **Availability:** Paper + theory; no drop-in code for our task.
- **Why it could help:** We have abundant unpaired IIRS (full cube) and
  TMC-2 (full strips). Could learn modality-robust features without any
  aligned pairs.
- **Honest caveats:** This is representation learning, not a matcher.
  Would need significant integration work to turn into correspondences.
  Research-direction, not a near-term lever.
- **Cost to try:** Weeks (research project).

---

## 2. More training data without new acquisitions

### 2a. AnyMatch synthetic pair generation (ECCV 2026) — HIGHEST IMPACT

- **What:** AnyMatch (arXiv:2606.31077, ECCV 2026) synthesizes large-scale
  multi-modal pairs from SINGLE-VIEW images: monocular depth estimation →
  3D reprojection → inpainting → cross-modal translation (RGB→IR via
  diffusion, RGB→depth/normal/event). Fine-tuning LoFTR/RoMa on the
  synthetic "Any-syn" dataset achieves SOTA cross-modal matching with
  zero-shot generalization. On remote sensing: LoFTR_AnyMatch 24.36/39.76/
  58.77 vs LoFTR_MINIMA 20.93/33.96/54.30 (AUC @5/10/20°).
- **Availability:** Code at github.com/mnyangs/anymatch (pipeline scripts
  for view transformation + modality transformation). Uses public models
  (Moge depth, Stable Diffusion inpainting, DiffV2IR).
- **Why it could help:** This directly solves our "only 6 windows" problem.
  We can synthesize THOUSANDS of IIRS-like↔TMC-2-like pairs from our
  existing TMC-2 strips: take TMC-2 → synthesize IR modality via diffusion
  → train CrossFeat (or fine-tune LoFTR) on synthetic pairs → validate on
  real 6 windows behind frozen gates. The geometric consistency is
  guaranteed by construction (same source image).
- **Honest caveats:** Synthetic IR ≠ real IIRS hyperspectral (different
  physics). The paper's remote-sensing eval is VIS-IR, not hyperspectral.
  Risk of synthetic-to-real gap. But the AnyMatch numbers beat MINIMA on
  remote sensing, and it's the only approach that multiplies our training
  data 100× without new acquisitions.
- **Cost to try:** 2-4 days (run synthesis pipeline on TMC-2 strips,
  generate pairs, train CrossFeat or fine-tune LoFTR, validate).

### 2b. Diffusion-based IIRS synthesis (lunar-specific)

- **What:** LS2ODiff (Remote Sensing 2026, 18, 1587) — diffusion framework
  for lunar SAR-to-optical translation, trained on LRO NAC south-pole
  mosaics. CM-Diff (arXiv:2503.09514) — bidirectional IR↔visible diffusion.
- **Availability:** Papers published; code availability not confirmed.
- **Why it could help:** If we can translate TMC-2 → synthetic IIRS-like
  imagery, we get unlimited training pairs with perfect geometric alignment.
  LS2ODiff is lunar-specific (south pole NAC), closer to our domain than
  terrestrial IR models.
- **Honest caveats:** Code not confirmed available. SAR→optical is not
  IIRS→optical. Would need adaptation.
- **Cost to try:** Unknown (depends on code availability).

### 2c. Our 9 sub-pixel pairs as training signal

- **What:** The 9 OHRC/TMC-2 sub-pixel pairs (0.27–0.47px) are
  high-confidence alignments.
- **Why it could help (limited):** These are optical↔optical (same
  modality), so they don't teach cross-modal appearance mapping directly.
  BUT they can serve as: (a) validation that a cross-modal method doesn't
  break same-modal performance, (b) geometric priors for the deform-field
  stage, (c) hard negatives for contrastive training.
- **Honest caveats:** Not a substitute for cross-modal training pairs.
  Useful as guardrails, not as primary signal.

### 2d. Public lunar cross-modal datasets

- **What:** LRO NAC mosaics (USGS Astropedia, public), M3 hyperspectral
  data (public via PDS), LOLA DEMs.
- **Why it could help:** M3 (Moon Mineralogy Mapper) is hyperspectral like
  IIRS. M3↔NAC pairs could provide additional cross-modal training data
  beyond our 6 IIRS windows. The MDPI M3 paper describes co-registered
  M3 south-pole mosaics.
- **Honest caveats:** Different sensor characteristics than IIRS. Would
  need co-registration work. Data volume is large.
- **Cost to try:** Days (download, co-register, validate).

---

## 3. Better architectures than VAE for the crossing function

### 3a. Diffusion-based descriptor translation

- **What:** Instead of CrossFeat's VAE, use a diffusion model to translate
  descriptors across modalities. Diff-Reg and Diff²I2P show diffusion
  improving cross-modal (2D-3D) matching via iterative denoising refinement.
- **Availability:** Research direction; no drop-in code for descriptor
  translation specifically.
- **Why it could help:** Diffusion models capture multimodal distributions
  better than VAEs. Could handle the cases where VAE mode-collapses.
- **Honest caveats:** Slower inference than VAE. No existing implementation
  for our task. Research project.
- **Cost to try:** Weeks.

### 3b. Contrastive descriptor learning

- **What:** Train cross-modal descriptors with contrastive loss (like CLIP
  but for local features). The CrossKEY paper (medical MR-US) uses
  curriculum triplet loss with hard negative mining for 3D cross-modal
  descriptors, achieving 69.8% precision.
- **Availability:** CrossKEY code at github.com/morozovdd/crosskey
  (medical domain, but the loss formulation transfers).
- **Why it could help:** Contrastive learning may produce more discriminative
  cross-modal descriptors than VAE reconstruction. Could replace or
  complement CrossFeat's VAE.
- **Honest caveats:** Still needs training pairs. Medical→lunar transfer
  unproven.
- **Cost to try:** 1-2 weeks (adapt loss, train, validate).

---

## 4. Hybrid approaches

### 4a. CrossFeat for initialization + our pipeline for refinement

- **What:** Use CrossFeat-crossed descriptors ONLY to get an initial
  coarse alignment (replacing LoFTR in the front end), then run our
  existing pipeline (deform-field stage, L1 rescue, frozen gates) for
  refinement.
- **Why it could help:** Plays to each component's strength: CrossFeat
  solves the modality gap for coarse matching; our geometric stages solve
  the sub-pixel refinement. If CrossFeat gets us from 1.31px to ~1.0px
  coarse, the deform field might take it sub-pixel.
- **Cost to try:** Included in CrossFeat trial (no extra work if CrossFeat
  is trained).

### 4b. CrossFeat descriptors + deform-field stage (tight coupling)

- **What:** Feed CrossFeat-crossed SIFT descriptors directly into the
  deform-field pipeline as the matcher front-end, replacing LoFTR.
  The stage's TPS field then refines whatever CrossFeat produces.
- **Why it could help:** Our 9-pair success shows the deform field works
  when given reasonable correspondences. If CrossFeat provides denser or
  more accurate cross-modal correspondences than LoFTR, the field can
  exploit them.
- **Honest caveats:** Requires CrossFeat training first. The deform field
  needs inliers to work with; if CrossFeat produces garbage, the stage
  will (correctly) decline.
- **Cost to try:** 1 day integration after CrossFeat is trained.

### 4c. Ensemble: LoFTR + CrossFeat + HOPC voting

- **What:** Run three independent matcher arms (LoFTR baseline, CrossFeat
  if trained, HOPC handcrafted) and combine via the frozen gate —
  accept the best verdict, or require agreement.
- **Why it could help:** Diversity. Each method fails differently;
  ensemble reduces the chance all fail simultaneously.
- **Honest caveats:** 3× compute. Gate logic for ensembles needs careful
  design to avoid weakening (must not accept if ANY arm says SUCCESS
  without all passing individually — or define a principled voting rule).
- **Cost to try:** 2-3 days (after CrossFeat trained).

---

## Recommended execution order

1. **This week (no training):** MatchAnything trial (1a) — afternoon,
   weights available now. HOPC/CFOG implementation (1b) — 1-2 days.
   Both plug into existing harness behind frozen gates.

2. **Next (needs GPU):** CrossFeat training on maximized dataset (in
   progress). AnyMatch synthesis (2a) to multiply training data 100× —
   start after CrossFeat baseline is established.

3. **Research directions:** Diffusion descriptors (3a), contrastive
  learning (3b), UML unpaired training (1c) — defer to dedicated sprints.

4. **After CrossFeat trained:** Hybrids (4a/4b/4c) — 1-3 days integration.

## Bottom line

The single highest-leverage NO-TRAINING move is **MatchAnything** (weights
available now, afternoon to try). The highest-leverage TRAINING-DATA move
is **AnyMatch synthesis** (100× data multiplication from existing TMC-2).
CrossFeat remains the primary bet, but these ensure we don't miss out if
it stalls.
