# RIFT2 License Research — Issue #14

Date: 2026-10-07. Read-only research; no repo changes made.

## (a) Upstream source and license

- **Canonical upstream:** `LJY-RS/RIFT2-multimodal-matching-rotation`
  (https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation) — the repo the
  RIFT2 paper itself points to ("The source code will be made publicly available
  in https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation").
- **Paper:** Li, Xu, Hu, Zhang, "RIFT2: Speeding-up RIFT with A New
  Rotation-Invariance Technique," arXiv:2303.00319 (2023).
- **License found in upstream:** **None.** The repository root contains MATLAB
  sources, demo scripts, and image folders, but no LICENSE, COPYING, or NOTICE
  file, and no license statement in its README. Verified 2026-10-07 via the
  GitHub repo file listing.
- **Vendored copy provenance:** `vendored/rift2/` matches the layout of the
  third-party Python port `canyagmur/RIFT2-MULTIMODAL-MATCHING-ROTATION-PYTHON`
  (`src/RIFT2.py`, `src/phase_congruency/phasecong.py`, `src/phase_congruency/tools.py`).
  That port likewise ships no license file.
- **Per-file headers in the vendored tree:**
  - `src/phase_congruency/phasecong.py` — MIT license header, attribution to
    Peter Kovesi (original MATLAB) and Alistair Muldal (Python translation).
    This permission covers **the phase-congruency component only**.
  - `src/RIFT2.py` — **no license or copyright header.**
  - `src/phase_congruency/tools.py` — **no license or copyright header.**
- Note: one unrelated third-party Python port claims MIT for its own rewrite in
  its README. That is a third party's claim about their own port, not a grant
  from the RIFT2 authors, and it does not cover the code vendored here.

## (b) Redistribution verdict: NOT PERMITTED (as things stand)

Under default copyright law, a public repository with **no license** means all
rights reserved — "publicly visible" is not "freely redistributable." None of
the following grant redistribution rights for the RIFT2 implementation:

- The arXiv paper (describes the algorithm; does not license the code).
- GitHub's Terms of Service (permit viewing and forking on GitHub, not
  redistribution into an independent project).
- The MIT header in `phasecong.py` (covers only that file's phase-congruency
  code, per its own attribution to Kovesi/Muldal — not `RIFT2.py` or `tools.py`).
- A third-party port's MIT claim (does not bind the original authors and does
  not apply to this vendored copy).

The repo's own `docs/vendor-license-review.md` reached the same conclusion and
recommended asking the upstream maintainers for a written license grant, or
removing the tree if none is forthcoming. No such grant is on record.

## (c) Runtime dependents (what breaks if the tree is removed)

Code that touches the vendored tree at runtime:

1. `chandra_align/matcher/__init__.py`
   - `_VENDORED_RIFT2` path constant; `_import_rift2()` inserts
     `vendored/rift2` into `sys.path` and imports `src.RIFT2.RIFT2`.
   - `RIFT2Matcher` class (wrapper; `name = "rift2"`).
   - Matcher factory: `name == "rift2"` requires `rift2_opt_in is True`, else
     raises `ValueError("RIFT2 is disabled unless matcher.rift2_opt_in is
     explicitly true")`.
2. `app.py`
   - Line ~421: `rift2_opt_in` derived from `CHANDRA_ENABLE_RIFT2` env var
     (default: disabled).
   - Lines ~443–454: opt-in-gated attempt → `RIFT2Matcher(npt=2048).match(...)`,
     records `diagnostics["rift2_error"]` on failure.
   - Line ~1975: help text mentioning `CHANDRA_ENABLE_RIFT2=1`.

Non-runtime references (docs/tests/config naming the engine) would need
companion edits but nothing else imports the tree. Removal is a contained,
coordinated change: delete `vendored/rift2/`, remove `RIFT2Matcher` +
factory branch + env-flag wiring, update docs/tests that name it.

## (d) Recommendation: REMOVE

1. **License:** No redistribution right exists for the RIFT2 implementation.
   Keeping unlicensed third-party code in a public repo is an open legal risk;
   the prior "keep" decision predates this analysis.
2. **Function:** The audit found RIFT2 yields 0 correspondences on every
   engine-reported run — it is non-functional here, and already disabled by
   default behind `CHANDRA_ENABLE_RIFT2=1`. Nothing of value is lost.
3. **Clean removal:** Only two runtime touchpoints (§c); no other module imports
   the tree. Phase 14's phase-congruency front-end
   (`chandra_align/xmodal/phase_congruency.py`) was reimplemented from scratch
   and is fully independent of the vendored files, so the modality-gap work
   does not depend on keeping the tree.
4. **Alternative considered and rejected:** keeping only MIT-licensed
   `phasecong.py` — pointless, since its only consumer is the RIFT2 port itself.

Suggested sequence: (1) attempt one written license request to the upstream
maintainers and file the response; (2) if no clear grant within a fixed window,
remove `vendored/rift2/` plus the wiring in §c in a single commit, and close
#14 noting the attempt. Do not claim the phase-congruency MIT notice covers
the RIFT2 implementation.
