# Dataset Preparation

Mamba-MoE is evaluated on the public **All-in-One medical image restoration benchmark** released with AMIR:

- Benchmark repository: <https://github.com/Yaziwel/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing>
- Original MRI source referenced by the benchmark: IXI dataset, <https://brain-development.org/ixi-dataset/>
- Original CT source referenced by the benchmark: AAPM Low-Dose CT Grand Challenge, <https://www.aapm.org/GrandChallenge/LowDoseCT/>

The AMIR benchmark repository provides preprocessing instructions and dataset download links through Baidu Netdisk and Google Drive. Please follow the original benchmark license and access requirements.

## Lightweight Paired-File Layout

This repository includes `mamba_moe.data.PairedRestorationDataset` for local paired-file experiments. It expects degraded inputs and reference targets under matching filename stems:

```text
data/
  MRI/
    input/
      case001.npy
      case002.npy
    gt/
      case001.npy
      case002.npy
  CT/
    input/
    gt/
  PET/
    input/
    gt/
```

A split-aware layout is also supported:

```text
data/
  MRI/
    train/
      input/
      gt/
    val/
      input/
      gt/
```

Supported formats are `.npy`, `.npz`, `.png`, `.tif`, and `.tiff`. Arrays are loaded as single-channel or explicit-channel tensors in `C x H x W` format. Integer images are converted to floating point in `[0, 1]`; floating-point arrays are preserved as `float32`.

Example:

```bash
python scripts/train.py --data_root /path/to/data --split train --batch_size 1 --steps 100
```

## Benchmark Layout

The full benchmark-specific release will follow the original AMIR benchmark organization and preprocessing conventions. This repository does not redistribute medical images.

## Evaluation Notes

All reported metrics in the manuscript are computed after:

1. the benchmark-provided modality-specific normalization is inverted;
2. predictions are truncated to the modality-specific physical intensity range;
3. metrics are computed under the strict saved-prediction protocol.

Case-level statistical testing is performed only when at least two independent cases are available. Under the benchmark grouping used in the paper, MRI has 114 cases, PET has 29 cases, and CT is reported descriptively because the CT test split contains one independent case.
