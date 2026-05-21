import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mamba_moe import DeterministicHMoE, build_mamba_moe_sharedhead


class PlaceholderRestorationDataset(Dataset):
    """Minimal placeholder dataset.

    Replace this class with the All-in-One benchmark loader in the full release.
    The expected item is (degraded_image, reference_image, modality_name).
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Mamba-MoE training skeleton.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mamba_moe_sharedhead(device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    dataset = PlaceholderRestorationDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    while step < args.steps:
        for x, y, modalities in loader:
            task = modalities[0] if isinstance(modalities, (list, tuple)) else modalities
            DeterministicHMoE.CURRENT_TASK = str(task)
            x = x.to(device)
            y = y.to(device)

            pred, _ = model(x)
            loss = charbonnier_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            print({"step": step, "task": DeterministicHMoE.CURRENT_TASK, "loss": float(loss.item())})
            if step >= args.steps:
                break

    torch.save({"model": model.state_dict()}, args.checkpoint_dir / "mamba_moe_minimal.pt")


if __name__ == "__main__":
    main()

