"""Strict checkpoint evaluator for paired MRI/CT/PET restoration files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from mamba_moe import DeterministicHMoE, PairedRestorationDataset, build_mamba_moe_sharedhead


def load_state_dict(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "ema", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return {k.replace("module.", "", 1): v for k, v in checkpoint.items()}


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float().numpy()
    if x.ndim == 4:
        x = x[:, 0]
    return x


def evaluate_modality(args: argparse.Namespace, model: torch.nn.Module, modality: str, device: str) -> List[Dict[str, float]]:
    dataset = PairedRestorationDataset(
        args.data_root,
        modalities=[modality],
        split=args.split,
        input_dir_name=args.input_dir,
        target_dir_name=args.target_dir,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    DeterministicHMoE.CURRENT_TASK = modality
    rows: List[Dict[str, float]] = []

    with torch.no_grad():
        for index, (x, y, _) in enumerate(loader):
            x = x.to(device).float()
            y = y.to(device).float()
            pred, _ = model(x)
            pred_np = tensor_to_numpy(pred)[0]
            gt_np = tensor_to_numpy(y)[0]
            pred_np = np.clip(pred_np, 0.0, args.data_range)
            gt_np = np.clip(gt_np, 0.0, args.data_range)
            rows.append(
                {
                    "modality": modality,
                    "index": index,
                    "PSNR": float(peak_signal_noise_ratio(gt_np, pred_np, data_range=args.data_range)),
                    "SSIM": float(structural_similarity(gt_np, pred_np, data_range=args.data_range)),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Mamba-MoE checkpoint evaluator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--input_dir", type=str, default="input")
    parser.add_argument("--target_dir", type=str, default="gt")
    parser.add_argument("--modalities", nargs="+", default=["MRI", "CT", "PET"])
    parser.add_argument("--data_range", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = build_mamba_moe_sharedhead(device=device)
    missing, unexpected = model.load_state_dict(load_state_dict(args.checkpoint, device), strict=False)
    if missing:
        print({"missing_keys": len(missing)})
    if unexpected:
        print({"unexpected_keys": len(unexpected)})
    model.eval()

    all_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    for modality in args.modalities:
        rows = evaluate_modality(args, model, modality, device)
        all_rows.extend(rows)
        summary_rows.append(
            {
                "modality": modality,
                "PSNR": float(np.mean([r["PSNR"] for r in rows])),
                "SSIM": float(np.mean([r["SSIM"] for r in rows])),
                "n": len(rows),
            }
        )
        print(summary_rows[-1])

    write_csv(args.save_dir / "per_slice_metrics.csv", all_rows)
    write_csv(args.save_dir / "summary.csv", summary_rows)


if __name__ == "__main__":
    main()
