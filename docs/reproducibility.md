# Reproducibility Notes

This document records the experimental protocol described in the manuscript. The public repository includes the core model, main training entry point, strict checkpoint evaluator, and local paired-file data loader. Large datasets, trained checkpoints, saved predictions, and case-level statistical artifacts are tracked outside Git.

## Benchmark

Experiments use the public All-in-One medical image restoration benchmark released with AMIR:

- MRI super-resolution
- CT denoising
- PET restoration

The strict test evaluation in the manuscript uses:

| Modality | Test slices | Independent cases | Case-level inference |
| --- | ---: | ---: | --- |
| MRI | 11,400 | 114 | yes |
| CT | 211 | 1 | descriptive only |
| PET | 2,044 | 29 | yes |

CT case-level significance is omitted because the available test grouping contains one independent CT case.

## Model Configuration

- Backbone width: 48 channels
- MambaIR blocks: 2-2-4-2-2
- RIN dictionary size: 16
- Instruction dimension: 256
- SRM experts: 4
- SRM top-k: 2
- H-MoE ranks:
  - MRI: 128
  - CT: 64
  - PET: 32
- Reconstruction head: one shared `3x3` reconstruction layer

## Training Protocol

The reported model uses:

- 120,000-step base training
- AdamW optimizer
- Maximum learning rate: `2e-4`
- Minimum learning rate: `1e-7`
- Cosine annealing
- Batch size: 12
- Gradient accumulation: 2
- Patch size: `128 x 128`
- Mixed-precision training
- Gradient clipping with max norm 1.0
- Balanced MRI/CT/PET modality exposure
- Random seed: 42

No additional 4,000-step or low-learning-rate refinement checkpoint is used for the reported main results. All main-table values correspond to the single consistently trained 120,000-iteration checkpoint.

## Evaluation Protocol

All methods are evaluated under a strict saved-prediction protocol:

1. Save predictions for each method and modality.
2. Denormalize predictions and references using the benchmark transforms.
3. Truncate predictions to the modality-specific physical intensity range.
4. Compute restoration metrics from saved predictions.

Primary main-table metrics:

- PSNR
- SSIM

Secondary diagnostic and proxy metrics:

- PET lesion Dice
- PET SUVmax bias
- PET SUVmean bias
- diagnostic high-frequency summaries for routing/module diagnostics and CT sanity evaluation

Slice-level confidence intervals use 5,000 bootstrap resamples with the percentile method. Statistical tests use case-level aggregation and paired Wilcoxon signed-rank tests when multiple independent cases are available.

## Hardware and Software

The manuscript reports inference cost on an NVIDIA GeForce RTX 5090 GPU using PyTorch 2.7.0 and CUDA 12.8. The public code release does not require the exact same GPU for model inspection or quick inference checks.
