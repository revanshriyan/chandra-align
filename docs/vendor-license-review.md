# Vendored RIFT2 license review

## Finding

The repository does not currently have enough licensing evidence to add a license notice for the entire `vendored/rift2/` tree. The upstream project is [LJY-RS/RIFT2-multimodal-matching-rotation](https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation), which contains Python and MATLAB implementations and cites the RIFT2 paper, but no explicit license for the RIFT2 implementation was located in its repository metadata or the checked source files. The RIFT2 paper is [arXiv:2303.00319](https://arxiv.org/abs/2303.00319). A public copy of Peter Kovesi's phase-congruency algorithm includes permission to redistribute with attribution, and the Python `phasecong.py` file in this repository identifies Kovesi as the original MATLAB author and Alistair Muldal as the Python translator. That permission applies to the phase-congruency component; it does not establish a license for RIFT2 itself or every file in this vendor subtree.

## Local attribution and license inventory

- `vendored/rift2/src/RIFT2.py`: no license or copyright header found; imports the phase-congruency implementation.
- `vendored/rift2/src/phase_congruency/phasecong.py`: includes an MIT license statement and attribution to Peter Kovesi and Alistair Muldal. The local header should be preserved. A copy of Kovesi's permission text is also published in the [NIH MIPAV algorithm license page](https://mipav.cit.nih.gov/license/CopyrightInfo.html), and the upstream Python translation's header is visible in [phasepack](https://github.com/alimuldal/phasepack/blob/master/phasepack/phasecong.py).
- `vendored/rift2/src/tools.py`: no explicit license or copyright header found.
- No top-level `LICENSE` or `NOTICE` covering the vendored RIFT2 implementation was found.

Because the core license is unresolved, this review deliberately does not add a new `LICENSE` or `NOTICE` and does not remove source files.

## What depends on this tree

Removing `vendored/rift2/` alone would break the import/path setup and RIFT2 class in `chandra_align/matcher/__init__.py`. The app's matcher routing and fallback in `app.py`, engine enums/metadata in `chandra_align/pipeline/router.py`, `src/api/main.py`, `src/exporters/dossier.py`, and `src/schemas/models.py`, plus tier configuration under `config/`, expose or select RIFT2. Tests, scripts, and docs also name the engine. The matcher is already documented as non-functional in current validation; removal is feasible only as a coordinated deprecation that removes the import and UI/config/API references and updates the associated tests and result descriptions.

## Recommendation

Ask the upstream maintainers to identify the license and provenance for the RIFT2 Python source and helper files, and retain that written response in this repository. Until that is resolved, treat redistribution of the RIFT2 implementation as an open risk and do not claim that the phase-congruency MIT notice licenses the full tree. If maintainers do not provide clear permission, remove the non-functional RIFT2 implementation in a separate change after unwiring all references listed above. Keep the separately licensed phase-congruency file only if its attribution and license are preserved and its continued use is reviewed independently.

## Sources checked

- [RIFT2 upstream repository](https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation)
- [RIFT2 paper](https://arxiv.org/abs/2303.00319)
- [Kovesi phase-congruency permission text](https://mipav.cit.nih.gov/license/CopyrightInfo.html)
- [Python phasecong MIT header](https://github.com/alimuldal/phasepack/blob/master/phasepack/phasecong.py)
