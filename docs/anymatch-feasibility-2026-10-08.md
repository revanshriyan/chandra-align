# AnyMatch Feasibility Assessment — 2026-10-08

## What AnyMatch Is

AnyMatch (ECCV 2026, arXiv:2606.31077, Yang et al., Wuhan University) synthesizes
large-scale multi-modal training pairs from single-view images, then fine-tunes
LoFTR/EDM/RoMa on the synthetic data. Code cloned to `~/workspace/anymatch/`
(776 files, public repo).

Three modules:

1. **View Transformation** (`View_Transformation/NovelViewTransformation.py`):
   Single RGB image → MoGe monocular depth → 3D point cloud → random novel
   viewpoint → differentiable rendering → Stable Diffusion 2 inpainting for
   occlusions. Outputs geometrically consistent image pairs with pixel-level
   GT correspondences.

2. **Modality Transformation** (`Modality_Transformation/`):
   - RGB → IR via DiffV2IR (diffusion, LoRA-fine-tuned on RGB-IR pairs)
   - RGB → depth/normal via MoGe
   - RGB → event via ESIM simulator

3. **SGCV** (`SGVC/SGVC_pck.py`): Quality filter using pretrained RoMa to
   compute PCK; retains only samples with PCK@5px ≥ 0.6.

## The Two Paths

### Path A: Pretrained fine-tuned weights (fast)

The repo provides LoFTR/EDM/RoMa weights **already fine-tuned** on the Any-syn
synthetic dataset, via Google Drive (`weights/download.sh`, file ID
`1JCz52pM7On_YWCjBWr_VuTiXt1uOXPD_`).

These can be tried directly on IIRS↔TMC-2 pairs with no synthesis and no
training — same effort class as the MatchAnything trial.

**Caveat:** Fine-tuned on RGB↔IR/depth/normal/event (terrestrial). Our
modalities are IIRS hyperspectral VNIR (~311 m/px) ↔ TMC-2 visible (5 m/px).
The modality gap is different in kind, not just degree.

### Path B: Full synthesis pipeline (slow)

Generate custom training data, then fine-tune LoFTR ourselves.

**Blockers:**

1. **GPU hardcoded.** `NovelViewTransformation.py` hardcodes `cuda:3`
   throughout. Will not run on CPU here. The 5070 could run it, but the
   code would need device-parameter patches first.

2. **Wrong modality.** DiffV2IR translates RGB→terrestrial-IR. It does not
   produce IIRS-like output (hyperspectral VNIR composite at 311 m/px with
   destriping artifacts and 62× GSD gap vs TMC-2). Training a TMC-2→IIRS
   diffusion model would require the paired data we're trying to avoid
   collecting — defeating the purpose.

3. **Heavy dependencies.** The RGB2IR module alone needs BLIP, SAM, CLIP,
   and Stable Diffusion. The full pipeline is a multi-day setup even on GPU.

4. **View transformation doesn't help the cross-modal gap.** Novel-view
   synthesis of TMC-2 images gives geometric augmentation, but both views
   remain TMC-2 modality. It cannot synthesize the IIRS side.

## Assessment for the IIRS Problem

| Question | Answer |
|---|---|
| Can AnyMatch generate synthetic IIRS-like pairs from TMC-2? | **No.** No IIRS modality transform exists; training one needs the paired data we're trying to synthesize. |
| Can the view transform augment our 19 real pairs? | Geometrically yes, but it doesn't address the cross-modal gap. Marginal value. |
| Are the pretrained weights worth trying? | **Yes** — fast trial, same as MatchAnything. Expect similar marginal results (fine-tuned on terrestrial RGB-IR, not lunar hyperspectral). |
| Can the pipeline run here? | **No** — hardcoded `cuda:3`, no GPU on this VM. |
| Can it run on the 5070? | View transform yes (after device patches); modality transform for IIRS no (see above). |

## Recommendation

1. **Try the pretrained AnyMatch-LoFTR weights** on the W5 test pair when
   convenient — 30-minute trial, no training. Expect COARSE-level results
   similar to MatchAnything (1.57px).

2. **Do not pursue the synthesis pipeline** for IIRS. The "100× data
   multiplier" applies to RGB↔IR/depth/normal/event domains where the
   modality transforms exist. Our IIRS modality has no transform, and
   building one is circular.

3. **CrossFeat remains the primary path.** It learns the IIRS↔TMC-2 mapping
   directly from the 19 real pairs instead of requiring a hand-built
   modality transform. The 5070 training is the correct next step.

## Steps If Path A Is Wanted Later

```bash
cd ~/workspace/anymatch
pip install gdown
bash weights/download.sh          # downloads fine-tuned LoFTR/EDM/RoMa
# Weights are standard LoFTR checkpoints — load with kornia or the
# existing LoFTR harness in chandra-align, run on crossfeat_data/test/
```

No GitHub pushes made. No code modified.
