# Mamba-MoE

Official implementation of **Mamba-MoE: Deterministic Expert Isolation With a Shared Output Head for All-in-One Medical Image Restoration**.

Mamba-MoE is an all-in-one medical image restoration framework for MRI super-resolution, CT denoising, and PET restoration/synthesis. It builds on an AMIR-style instruction-guided Mamba encoder-decoder and injects deterministic modality-matched residual experts into intermediate convolutional operators. MRI uses a spatial expert for stride-1 wrapped convolutions, whereas CT and PET use compact channel experts. The final reconstruction layer is shared across modalities.

> **Release status.** This repository currently provides a lightweight core implementation for paper review and reproducibility inspection. Full training scripts, pretrained checkpoints, saved-prediction evaluation scripts, and case-level statistical analysis scripts will be released with the full version.

## Overview

![Mamba-MoE framework](assets/fig1.png)

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
  assets/
    fig1.png
  configs/
    mamba_moe_sharedhead.yaml
  dataset/
    README.md
  mamba_moe/
    __init__.py
    model.py
  scripts/
    run_minimal_inference.py
```

## Installation

```bash
conda create -n mamba_moe python=3.10
conda activate mamba_moe
pip install -r requirements.txt
```

The model expects a VMamba/VSSBlock implementation. In the full release, `mamba_moe/vmamba.py` will be populated by the project-compatible VSSBlock implementation. For now, `mamba_moe/model.py` keeps the same import path used in our experiments so the architecture remains aligned with the manuscript.

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

## Training

The full training pipeline will be released after publication. The manuscript uses:

- 120,000-step base training;
- 4,000-step shared-head refinement;
- balanced MRI/CT/PET modality exposure;
- deterministic modality context during inference.

The key model configuration is summarized in [`configs/mamba_moe_sharedhead.yaml`](configs/mamba_moe_sharedhead.yaml).

## Evaluation

The full release will include scripts for:

- strict saved-prediction evaluation;
- PSNR, SSIM, and HFEN computation after denormalization and modality-specific truncation;
- MRI/CT edge Dice;
- PET lesion Dice and SUV bias;
- case-level paired significance testing.

## Checkpoints

Pretrained checkpoints will be released upon publication.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{mambamoe2026,
  title={Mamba-MoE: Deterministic Expert Isolation With a Shared Output Head for All-in-One Medical Image Restoration},
  author={Anonymous},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  note={Under review}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

