# Mamba-MoE

Official implementation of **Mamba-MoE: Deterministic Expert Isolation With a Shared Output Head for All-in-One Medical Image Restoration**.

<p align="center">
  <img src="assets/fig1.png" width="95%">
</p>

<p align="center">
  <em>Deterministic intermediate expert isolation with a shared output head for all-in-one medical image restoration.</em>
</p>

Mamba-MoE is an all-in-one medical image restoration framework for MRI super-resolution, CT denoising, and PET restoration/synthesis. It builds on an AMIR-style instruction-guided Mamba encoder-decoder and injects deterministic modality-matched residual experts into intermediate convolutional operators. MRI uses a spatial expert for stride-1 wrapped convolutions, whereas CT and PET use compact channel experts. The final reconstruction layer is shared across modalities.

> **Release status.** This repository provides the core model, lightweight training/evaluation utilities, and prediction export scripts. Full benchmark-specific dataloaders, pretrained checkpoints, full saved predictions, and case-level statistical analysis scripts will be released upon publication.

## Key Features

- **All-in-one restoration:** one model handles MRI, CT, and PET restoration tasks.
- **Deterministic expert isolation:** the known modality identity activates one matched expert branch without learned soft expert mixing.
- **Heterogeneous experts:** MRI uses spatial residual experts, while CT and PET use channel-oriented residual experts.
- **Intermediate injection:** H-MoE wraps selected intermediate `Conv2d` operators and computes `z = C(u) + E_m(u)`.
- **Shared output head:** one shared `3x3` reconstruction layer is used for all modalities with a global input residual.
- **Lightweight release:** the current code isolates the core architecture without large checkpoints or dataset files.

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
    reproducibility.md
  configs/
    mamba_moe_sharedhead.yaml
  dataset/
    README.md
  mamba_moe/
    __init__.py
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

This work uses the public All-in-One medical image restoration benchmark released with AMIR. Please see [`dataset/README.md`](dataset/README.md) for dataset links and the expected local directory layout.

No dataset files are redistributed in this repository.

## Qualitative Results

<p align="center">
  <img src="assets/fig2.png" width="95%">
</p>

Representative MRI, CT, and PET restoration examples are shown under the same saved-prediction protocol used in the manuscript. Quantitative interpretation should rely on the full test-set and case-level summaries reported in the paper.

## Training

The repository includes a minimal training skeleton in `scripts/train.py`. This script is intended to verify the optimization path and model wiring; it uses a placeholder dataset and is not the full All-in-One benchmark dataloader. The manuscript uses:

- 120,000-step base training;
- 4,000-step shared-head refinement;
- balanced MRI/CT/PET modality exposure;
- deterministic modality context during inference.

The key model configuration is summarized in [`configs/mamba_moe_sharedhead.yaml`](configs/mamba_moe_sharedhead.yaml).

## Evaluation

The repository includes lightweight utilities for saved-prediction evaluation and prediction export:

- strict saved-prediction evaluation;
- PSNR, SSIM, and HFEN computation after denormalization and modality-specific truncation;
- prediction export with a fixed modality context.

Metric definitions are summarized in [`docs/metrics.md`](docs/metrics.md). Full benchmark-specific evaluation wrappers, MRI/CT edge Dice, PET lesion Dice and SUV bias, and case-level paired significance testing scripts will be released upon publication.

## Checkpoints

Pretrained checkpoints will be released upon publication.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{mambamoe2026,
  title={Mamba-MoE: Deterministic Expert Isolation With a Shared Output Head for All-in-One Medical Image Restoration},
  author={Liu, Yang and Man, Ranran and Peng, Yanjun and Sun, Jindong and Yang, Guang},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  note={Under review}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
