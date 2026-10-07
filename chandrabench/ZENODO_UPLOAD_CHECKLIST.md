# Zenodo upload checklist — ChandraBench v0.1

Do NOT attempt an API upload from an agent session. Zenodo needs a human
login (or a personal access token the human creates). Prepare everything
below, then do the upload in one sitting at https://zenodo.org/deposit.

## 0. Pre-flight (in this repo, already done)

- [x] `chandrabench/landmarks_ohrc_pair_v0.1.csv` — 20 rows, sha256
      `8de1613a70181dcfc286bea9d0168dc718de8579323483ef5c30f615cb8f2fa5`
- [x] `chandrabench/landmarks_tmc2_fore_nadir_v0.1.csv` — 20 rows, sha256
      `254d9d2c24b43d947dc15f3e9c1745ea238ac8b80f9d4774d8b0abf70a9a79c0`
- [x] `chandrabench/eval.py` — evaluator, self-test passes
- [x] `chandrabench/README.md` — task, metric, format, limitations
- [x] `chandrabench/LICENSE` — CC-BY-4.0 deed summary
- [x] `docs/chandrabench-tech-note.md` — 2-page tech note

Verify once more before upload: `python chandrabench/eval.py --self-test`
must print `SELF-TEST PASSED`.

## 1. Files to upload (as a .zip plus the loose files)

Upload these exact files (keep the `chandrabench/` directory structure):

- `chandrabench/landmarks_ohrc_pair_v0.1.csv`
- `chandrabench/landmarks_tmc2_fore_nadir_v0.1.csv`
- `chandrabench/eval.py`
- `chandrabench/README.md`
- `chandrabench/LICENSE`
- `docs/chandrabench-tech-note.md` (rename on upload to
  `chandrabench-tech-note-v0.1.md` so it sits next to the data)

Optional but recommended: also attach `results/table_issue02_ground_truth.csv`
as `reference-scores-v0.1.csv` so others can see the release-time baseline
numbers in context.

## 2. Metadata to enter on the deposit form

- **Title:** ChandraBench v0.1 — human-verified correspondences for lunar
  image registration (Chandrayaan-2 OHRC / TMC-2)
- **Authors:** CHANDRA-ALIGN team (add individual names as desired)
- **Description:** paste the first three sections of
  `docs/chandrabench-tech-note.md` (What / How collected / Metric), then add:
  "40 human-verified landmarks (20 per pair) for two Chandrayaan-2 pairs,
  with a fixed RMSE metric, a fail-closed circularity guard, and a one-command
  numpy-only evaluator. Feature-correspondence level; ~0.5–1 px localization
  noise; comparative benchmark, not an absolute accuracy claim."
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Keywords:** lunar, moon, image registration, Chandrayaan-2, OHRC, TMC-2,
  correspondences, benchmark, planetary
- **Version:** v0.1
- **Language:** eng
- **Resource type:** Dataset

## 3. DOI

- On the deposit page, click **Reserve DOI** BEFORE publishing. Copy the
  reserved DOI back into:
  - `chandrabench/README.md` (the `[reserved at upload]` line)
  - `docs/chandrabench-tech-note.md` (the citation block)
- Then publish. The DOI becomes live immediately.

## 4. After publishing

- [ ] Paste the DOI into the two files above and commit ("docs: ChandraBench
      Zenodo DOI live").
- [ ] Update the README benchmark section with the DOI badge/link.
- [ ] Announce the benchmark (the plan invites other teams to run it).
- [ ] File the follow-up issue for v0.2: absolute LROC/SLDEM controls.
