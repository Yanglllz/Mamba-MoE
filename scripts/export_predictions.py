import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread

from mamba_moe import DeterministicHMoE, build_mamba_moe_sharedhead


def load_image(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {arr.shape} for {path}")
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def iter_inputs(input_dir: Path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(input_dir.rglob(f"*{suffix}"))
    return sorted(files)


def save_prediction(pred: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    np.save(output_path, arr)


def resolve_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "generator", "netG", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if isinstance(checkpoint, dict):
        return {str(k).replace("module.", "", 1): v for k, v in checkpoint.items()}
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export saved predictions for the strict evaluation protocol.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["MRI", "CT", "PET"])
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--suffixes", nargs="+", default=[".npy", ".png", ".tif", ".tiff"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mamba_moe_sharedhead(device=device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(resolve_state_dict(checkpoint), strict=False)
    model.eval()

    DeterministicHMoE.CURRENT_TASK = args.task
    input_files = iter_inputs(args.input_dir, args.suffixes)
    if not input_files:
        raise RuntimeError(f"No input files found in {args.input_dir}")

    with torch.no_grad():
        for input_path in input_files:
            rel = input_path.relative_to(args.input_dir)
            output_path = (args.output_dir / rel).with_suffix(".npy")
            x = load_image(input_path).to(device)
            pred, _ = model(x)
            save_prediction(pred, output_path)
            print(f"{input_path} -> {output_path}")


if __name__ == "__main__":
    main()

