# Judge Q&A: where the team ran the measurements

> The Issue #1 matcher comparison ran locally on an NVIDIA RTX 5070 with PyTorch 2.11.0+cu128. The app also has a CPU SIFT + RANSAC fallback; that fallback produced the earlier CPU benchmark results. The repository contains a Kaggle-ready notebook, but its cells have no execution counts or saved outputs, and we found no Kaggle run logs or Colab execution artifacts. We therefore have no repository evidence that the reported comparison was run on Kaggle or Colab.

Evidence checked: `results/table_issue01_gpu_validation.csv` and `results/run_issue01_gpu_validation.py`; `notebooks/chandra_align_kaggle.ipynb` (4 cells, no executed code cells or outputs); repository search for Kaggle credentials/run outputs and Colab artifacts. No `kaggle.json` was found in the repository.
