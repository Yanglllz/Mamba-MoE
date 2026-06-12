"""Main Mamba-MoE training entry point for paired MRI/CT/PET restoration data.

The script is intentionally dataset-path agnostic. It expects local paired files
under:

    data_root/MRI/train/input, data_root/MRI/train/gt
    data_root/CT/train/input,  data_root/CT/train/gt
    data_root/PET/train/input, data_root/PET/train/gt

Each optimization step samples one batch from each modality and applies the
corresponding deterministic H-MoE route before the forward pass. This matches
the paper's known-modality routing assumption while keeping the launcher usable
outside the original server filesystem.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from mamba_moe import DeterministicHMoE, PairedRestorationDataset, build_mamba_moe_sharedhead


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def build_modality_loaders(args: argparse.Namespace) -> Dict[str, Iterable]:
    loaders = {}
    for modality in args.modalities:
        dataset = PairedRestorationDataset(
            args.data_root,
            modalities=[modality],
            split=args.split,
            input_dir_name=args.input_dir,
            target_dir_name=args.target_dir,
        )
        loaders[modality] = cycle(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
        )
        print(f"Loaded {len(dataset)} {modality} training pairs.")
    return loaders


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, output_dir / f"step_{step:06d}.pth")
    torch.save(ckpt, output_dir / "latest.pth")


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = build_mamba_moe_sharedhead(device=device, base_ckpt=args.resume_model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.min_lr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    loaders = build_modality_loaders(args)
    modality_order = list(args.modalities)
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    for step in range(1, args.steps + 1):
        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum_steps):
            for modality in modality_order:
                x, y, _ = next(loaders[modality])
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).float()

                DeterministicHMoE.CURRENT_TASK = modality
                with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                    pred, _ = model(x)
                    loss = charbonnier_loss(pred, y)
                    loss = loss / max(1, len(modality_order) * args.grad_accum_steps)
                scaler.scale(loss).backward()
                step_loss += float(loss.item())

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print({"step": step, "loss": step_loss, "lr": lr})

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.output_dir / "checkpoints", step, model, optimizer, scheduler, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main 120k Mamba-MoE training launcher.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/mamba_moe_120k"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--input_dir", type=str, default="input")
    parser.add_argument("--target_dir", type=str, default="gt")
    parser.add_argument("--modalities", nargs="+", default=["MRI", "CT", "PET"])
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-modality micro-batch size.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_model", type=str, default=None, help="Optional model state_dict checkpoint.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
