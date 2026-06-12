# Mamba-MoE

Official implementation of **Mamba-MoE: Deterministic Intermediate Expert Isolation With a Shared Reconstruction Head for All-in-One Medical Image Restoration**.

<p align="center">
  <img src="assets/fig1.png" width="95%">
</p>

<p align="center">
  <em>Deterministic intermediate expert isolation with a shared reconstruction head for all-in-one medical image restoration.</em>
</p>

Mamba-MoE is an all-in-one medical image restoration framework for MRI super-resolution, CT denoising, and PET restoration. It builds on an AMIR-style instruction-guided Mamba encoder-decoder and injects deterministic modality-matched residual experts into intermediate convolutional operators. MRI uses a spatial expert for stride-1 wrapped convolutions, whereas CT and PET use compact channel experts. The final reconstruction layer is shared across modalities.

> **Release status.** This repository provides the core model, a project-compatible VSSBlock implementation, a paired-file restoration dataloader, lightweight training/evaluation utilities, and prediction export scripts. Full benchmark-specific dataloaders, pretrained checkpoints, full saved predictions, and case-level statistical analysis scripts will be released upon publication.
> This lightweight release mirrors the shared-reconstruction-head design used in the manuscript, but it is not yet the complete 120k benchmark reproduction package. Checkpoint-compatible construction details, checkpoint hashes, benchmark-specific evaluators, and full statistical artifacts will be released with the full artifact package.

## Key Features

- **All-in-one restoration:** one model handles MRI, CT, and PET restoration tasks.
- **Deterministic expert isolation:** the known modality identity activates one matched expert branch without learned soft expert mixing.
- **Heterogeneous experts:** MRI uses spatial residual experts, while CT and PET use channel-oriented residual experts.
- **Intermediate injection:** H-MoE wraps selected intermediate `Conv2d` operators and computes `z = C(u) + E_m(u)`.
- **Shared reconstruction head:** one shared `3x3` reconstruction layer is used for all modalities with a global input residual.
- **Paired-file dataloader:** a lightweight loader supports local paired `input`/`gt` restoration files for MRI, CT, and PET.

## Repository Structure

```text
Mamba-MoE/
  README.md
  LICENSE
  requirements.txt
  environment.yml
  assets/
    fig1.png
    fig2.png
  docs/
    metrics.md
    release_manifest.md
    reproducibility.md
  configs/
    mamba_moe_sharedhead.yaml
  dataset/
    README.md
  mamba_moe/
    __init__.py
    data.py
    model.py
    vmamba.py
  scripts/
    evaluate.py
    export_predictions.py
    run_minimal_inference.py
    train.py
```

## Installation

```bash
conda create -n mamba_moe python=3.10
conda activate mamba_moe
pip install -r requirements.txt
```

The model uses the project-compatible VSSBlock implementation provided in `mamba_moe/vmamba.py`. The import path in `mamba_moe/model.py` matches the implementation used in our experiments so the architecture remains aligned with the manuscript.

`mamba-ssm` and `causal-conv1d` usually require a CUDA/PyTorch build that matches the local GPU environment. CPU-only environments are suitable for source inspection and data-loader checks, but Mamba-backed inference requires these packages to be installed successfully.

## Minimal Usage

```python
import torch
from mamba_moe import DeterministicHMoE, build_mamba_moe_sharedhead

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_mamba_moe_sharedhead(device=device)

x = torch.randn(1, 1, 128, 128, device=device)
DeterministicHMoE.CURRENT_TASK = "MRI"  # one of: MRI, CT, PET

with torch.no_grad():
    y, router_logits = model(x)
print(y.shape)
```

You can also run the minimal inference entry point:

```bash
python scripts/run_minimal_inference.py --task MRI --height 128 --width 128
```

## Dataset Preparation

This work uses the public All-in-One medical image restoration benchmark released with AMIR. Please see [`dataset/README.md`](dataset/README.md) for dataset links, access notes, and local directory examples.

For lightweight local experiments, the included `PairedRestorationDataset` expects paired degraded/reference files under this layout:

```text
data/
  MRI/
    input/
      case001.npy
    gt/
      case001.npy
  CT/
    input/
    gt/
  PET/
    input/
    gt/
```

Supported paired-file formats are `.npy`, `.npz`, `.png`, `.tif`, and `.tiff`. Files are paired by filename stem. If a split is provided, the loader also supports `data/<modality>/<split>/input` and `data/<modality>/<split>/gt`.

No dataset files are redistributed in this repository.

## Qualitative Results

<p align="center">
  <img src="assets/fig2.png" width="95%">
</p>

Representative MRI, CT, and PET restoration examples are shown under the same saved-prediction protocol used in the manuscript. Quantitative interpretation should rely on the full test-set and case-level summaries reported in the paper.

## Training

The repository includes a lightweight training skeleton in `scripts/train.py`. It can run either on a placeholder random dataset for wiring checks or on paired local restoration files through `--data_root`.

This script is intended for wiring checks and local adaptation experiments. It is not the full 120,000-iteration benchmark training launcher used for the manuscript.

Placeholder wiring check:

```bash
python scripts/train.py --steps 10 --batch_size 1
```

Paired-file training example:

```bash
python scripts/train.py \
  --data_root /path/to/data \
  --split train \
  --batch_size 1 \
  --steps 100
```

`DeterministicHMoE` uses one modality context per forward pass, so mixed-modality batches require a modality-grouped sampler. For simple local experiments, use `--batch_size 1`.

The manuscript reports the main model from:

- one 120,000-iteration checkpoint;
- balanced MRI/CT/PET modality exposure;
- deterministic modality context during inference.

No additional 4,000-step or low-learning-rate refinement checkpoint is used for the reported main results.

The key model configuration is summarized in [`configs/mamba_moe_sharedhead.yaml`](configs/mamba_moe_sharedhead.yaml).

## Evaluation

The repository includes lightweight utilities for saved-prediction evaluation and prediction export:

- strict saved-prediction evaluation;
- PSNR and SSIM computation after denormalization and modality-specific truncation;
- prediction export with a fixed modality context.

The included `scripts/evaluate.py` summarizes PSNR and SSIM only; it does not generate manuscript confidence intervals, PET proxy metrics, CT sanity results, routing diagnostics, or module diagnostics. Metric definitions are summarized in [`docs/metrics.md`](docs/metrics.md). Full benchmark-specific evaluation wrappers, PET lesion Dice and SUV bias, CT sanity metrics, diagnostic high-frequency summaries, and case-level paired significance testing scripts will be released upon publication.

## Checkpoints

Pretrained checkpoints will be released upon publication.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{mambamoe2026,
  title={Mamba-MoE: Deterministic Intermediate Expert Isolation With a Shared Reconstruction Head for All-in-One Medical Image Restoration},
  author={Liu, Yang and Man, Ranran and Peng, Yanjun and Sun, Jindong and Yang, Guang},
  journal={Machine Intelligence Research},
  year={2026},
  note={Under review}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
