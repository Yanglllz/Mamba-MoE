import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from mamba_moe import DeterministicHMoE, PairedRestorationDataset, build_mamba_moe_sharedhead


class PlaceholderRestorationDataset(Dataset):
    """Small synthetic dataset for installation and wiring checks.

    The real paired-file loader is ``PairedRestorationDataset`` and can be
    enabled with ``--data_root``. The expected item is
    (degraded_image, reference_image, modality_name).
    """

    def __init__(self, length: int = 16, image_size: int = 128) -> None:
        self.length = length
        self.image_size = image_size
        self.modalities = ["MRI", "CT", "PET"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        x = torch.rand(1, self.image_size, self.image_size)
        y = x.clone()
        modality = self.modalities[index % len(self.modalities)]
        return x, y, modality


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def _resolve_batch_task(modalities) -> str:
    if isinstance(modalities, str):
        return modalities
    if isinstance(modalities, (list, tuple)):
        task = str(modalities[0])
        if any(str(item) != task for item in modalities):
            raise ValueError(
                "Mixed-modality batch detected. DeterministicHMoE uses one modality "
                "context per forward pass; use --batch_size 1 or a modality-grouped sampler."
            )
        return task
    return str(modalities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick-start Mamba-MoE training utility.")
    parser.add_argument("--data_root", type=Path, default=None, help="Optional paired restoration dataset root.")
    parser.add_argument("--split", type=str, default=None, help="Optional split name, e.g. train or val.")
    parser.add_argument("--modalities", nargs="+", default=["MRI", "CT", "PET"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=128, help="Synthetic dataset image size.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mamba_moe_sharedhead(device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    if args.data_root is None:
        dataset = PlaceholderRestorationDataset(image_size=args.image_size)
        print("Using synthetic random data. Pass --data_root to use paired restoration files.")
    else:
        dataset = PairedRestorationDataset(args.data_root, modalities=args.modalities, split=args.split)
        print(f"Loaded {len(dataset)} paired samples from {args.data_root}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    while step < args.steps:
        for x, y, modalities in loader:
            task = _resolve_batch_task(modalities)
            DeterministicHMoE.CURRENT_TASK = task
            x = x.to(device)
            y = y.to(device)

            pred, _ = model(x)
            loss = charbonnier_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            print({"step": step, "task": task, "loss": float(loss.item())})
            if step >= args.steps:
                break

    torch.save({"model": model.state_dict()}, args.checkpoint_dir / "mamba_moe_minimal.pt")


if __name__ == "__main__":
    main()
