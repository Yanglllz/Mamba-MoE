# Dataset Preparation

Mamba-MoE is evaluated on the public **All-in-One medical image restoration benchmark** released with AMIR:

- Benchmark repository: <https://github.com/Yaziwel/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing>
- Original MRI source referenced by the benchmark: IXI dataset, <https://brain-development.org/ixi-dataset/>
- Original CT source referenced by the benchmark: AAPM Low-Dose CT Grand Challenge, <https://www.aapm.org/GrandChallenge/LowDoseCT/>

The AMIR benchmark repository provides preprocessing instructions and dataset download links through Baidu Netdisk and Google Drive. Please follow the original benchmark license and access requirements.

## Expected Layout

The full training and evaluation release will expect the benchmark data under a layout similar to:

```text
data/
  MRI/
    train/
    test/
  CT/
    train/
    test/
  PET/
    train/
    test/
```

The exact file naming convention follows the released AMIR benchmark files. This repository does not redistribute medical images.

## Evaluation Notes

All reported metrics in the manuscript are computed after:

1. the benchmark-provided modality-specific normalization is inverted;
2. predictions are truncated to the modality-specific physical intensity range;
3. metrics are computed under the strict saved-prediction protocol.

Case-level statistical testing is performed only when at least two independent cases are available. Under the benchmark grouping used in the paper, MRI has 114 cases, PET has 29 cases, and CT is reported descriptively because the CT test split contains one independent case.

