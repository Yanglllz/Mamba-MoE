import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def iter_pairs(pred_dir: Path, gt_dir: Path, suffixes: Iterable[str]):
    gt_files = []
    for suffix in suffixes:
        gt_files.extend(gt_dir.rglob(f"*{suffix}"))
    for gt_path in sorted(gt_files):
        rel = gt_path.relative_to(gt_dir)
        pred_path = pred_dir / rel
        if pred_path.exists():
            yield pred_path, gt_path


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, data_range: float) -> Dict[str, float]:
    return {
        "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=data_range)),
        "ssim": float(structural_similarity(gt, pred, data_range=data_range)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight saved-prediction evaluator.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument("--data_range", type=float, default=1.0)
    parser.add_argument("--suffixes", nargs="+", default=[".npy", ".png", ".tif", ".tiff"])
    args = parser.parse_args()

    scores = []
    for pred_path, gt_path in iter_pairs(args.pred_dir, args.gt_dir, args.suffixes):
        pred = load_image(pred_path)
        gt = load_image(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {pred_path} {pred.shape} vs {gt_path} {gt.shape}")
        scores.append(evaluate_pair(pred, gt, args.data_range))

    if not scores:
        raise RuntimeError("No matched prediction/reference pairs were found.")

    keys = scores[0].keys()
    summary = {key: float(np.mean([item[key] for item in scores])) for key in keys}
    summary["n"] = len(scores)
    print(summary)


if __name__ == "__main__":
    main()
