# Mamba-MoE

Minimal PyTorch implementation for **Mamba-MoE: Deterministic Expert Isolation With a Shared Output Head for All-in-One Medical Image Restoration**.

This repository contains the core model components needed to reproduce the proposed architecture:

- an AMIR-style instruction-guided Mamba encoder-decoder;
- deterministic intermediate H-MoE expert injection;
- MRI spatial expert and CT/PET channel experts;
- one shared output head with global input residual.

The repository is intentionally lightweight. Dataset preparation, trained checkpoints, saved-prediction evaluation scripts, and case-level statistical analysis scripts will be released with the full version.

## Installation

```bash
conda create -n mamba_moe python=3.10
conda activate mamba_moe
pip install -r requirements.txt
```

The model expects a VMamba/VSSBlock implementation. In the full release, `mamba_moe/vmamba.py` will be populated by the project-compatible VSSBlock implementation. For now, `mamba_moe/model.py` keeps the same import path used in our experiments.

## Minimal Usage

```python
import torch
from mamba_moe import build_mamba_moe_sharedhead, DeterministicHMoE

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_mamba_moe_sharedhead(device=device)

x = torch.randn(1, 1, 128, 128, device=device)
DeterministicHMoE.CURRENT_TASK = "MRI"  # one of: MRI, CT, PET
with torch.no_grad():
    y, router_logits = model(x)
```

## Deterministic Routing

The H-MoE selector does not learn input-adaptive probabilities. It uses the known modality identity:

- `MRI`: spatial expert for stride-1 wrapped convolutions, channel fallback for strided convolutions;
- `CT`: channel expert;
- `PET`: channel expert.

Each wrapped convolution computes

```text
z = C(u) + E_m(u)
```

where `C` is the frozen base convolution and `E_m` is the selected modality-matched residual expert.

## Citation

Citation information will be added after publication.

