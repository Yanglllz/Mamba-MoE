# Metrics

This document summarizes the metrics used in the Mamba-MoE manuscript and how they should be interpreted in medical image restoration.

This file provides metric definitions. The public saved-prediction script
`scripts/evaluate.py` implements PSNR and SSIM summaries. Benchmark-specific PET proxy metrics,
CT sanity evaluation, routing diagnostics, module diagnostics, and statistical
testing scripts will be released with the full artifact package.

## Restoration Metrics

### PSNR

Peak signal-to-noise ratio (PSNR) measures pixel-level fidelity between the restored image and the reference image. Higher PSNR is better.

In the manuscript, PSNR is computed after:

1. inverting the benchmark-provided modality-specific normalization;
2. truncating predictions to the modality-specific physical intensity range;
3. computing the metric in the restored physical intensity space.

Because intensity ranges differ across MRI, CT, and PET, PSNR should be interpreted within each modality rather than as a direct cross-modality magnitude comparison.

### SSIM

Structural similarity (SSIM) measures local luminance, contrast, and structural agreement between prediction and reference. Higher SSIM is better.

SSIM is computed under the same denormalization and truncation protocol as PSNR, using the postprocessed physical intensity range as the data range.

### Diagnostic HF-RMSE

Diagnostic high-frequency RMSE is used only for controlled ablation and case-level diagnostic analysis. It applies a Laplace high-pass filter to prediction and reference images and measures RMSE between the filtered images. Lower diagnostic HF-RMSE is better.

Diagnostic HF-RMSE is not part of the main cross-model restoration table and should be interpreted only within the diagnostic protocol where it is reported.

## Downstream Proxy Metrics

### PET Lesion Dice

PET lesion Dice measures overlap between predicted and reference lesion masks. In the manuscript, lesion masks are generated using a fixed SUV threshold of `2.5`. Higher lesion Dice is better.

This metric evaluates lesion-level uptake consistency under a fixed thresholding rule. It is not a substitute for physician annotation or clinical endpoint validation.

### SUVmax Bias

SUVmax bias measures the relative error of maximum standardized uptake value inside the ground-truth lesion region. Lower absolute bias is better.

### SUVmean Bias

SUVmean bias measures the relative error of mean standardized uptake value inside the ground-truth lesion region. Lower absolute bias is better.

SUVmax and SUVmean biases are complementary to lesion Dice. A method can improve lesion overlap while still showing mixed uptake quantification behavior.

## Statistical Testing

Slice-level metrics are summarized with bootstrap 95% confidence intervals over test slices. These intervals are descriptive because neighboring slices from the same case are not independent.

For statistical inference, slices are first aggregated by independent case. Paired Wilcoxon signed-rank tests are then applied to case-level means when at least two independent cases are available.

Under the benchmark grouping used in the manuscript:

- MRI: 114 independent cases
- CT: 1 independent case
- PET: 29 independent cases

CT case-level significance is omitted because only one independent CT case is available.
