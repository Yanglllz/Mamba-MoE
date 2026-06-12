# Manuscript Core Code

This directory contains the manuscript-facing core entry points:

- `model/FreqMamba_MoE_SharedHead.py`: model entry point for the shared-reconstruction-head Mamba-MoE graph.
- `train_main_120k.py`: main 120,000-step training launcher for local paired MRI/CT/PET restoration data.
- `evaluate_strict.py`: strict checkpoint evaluation script for PSNR/SSIM summaries and per-slice CSV export.

Large datasets, pretrained checkpoints, saved predictions, and case-level
statistical artifacts are not stored in Git. Point these scripts to local
benchmark paths and checkpoint files.

The reported manuscript setting uses one 120,000-iteration checkpoint, 76
intermediate H-MoE sites, deterministic known-modality routing, and one shared
reconstruction head. No 4,000-step refinement checkpoint is used for the
reported main results.
