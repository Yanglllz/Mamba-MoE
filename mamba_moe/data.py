from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = (".npy", ".npz", ".png", ".tif", ".tiff")


def _array_from_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            first_key = sorted(data.files)[0]
            array = data[first_key]
    elif suffix in {".png", ".tif", ".tiff"}:
        array = io.imread(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")
    return np.asarray(array)


def _to_chw_float(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        # Accept either CHW or HWC. Medical slices are expected to be single-channel;
        # multi-channel images are preserved when the channel axis is explicit.
        if array.shape[0] in (1, 3, 4):
            pass
        elif array.shape[-1] in (1, 3, 4):
            array = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(
                "Expected a 2D image or a 3D image with an explicit channel axis, "
                f"got shape {array.shape}."
            )
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(info.max)
    else:
        array = array.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(array))


class PairedRestorationDataset(Dataset):
    """Paired-file dataset for lightweight restoration experiments.

    Expected layout by default::

        root/
          MRI/input/*.npy|*.png|*.tif|*.tiff
          MRI/gt/*.npy|*.png|*.tif|*.tiff
          CT/input/...
          CT/gt/...
          PET/input/...
          PET/gt/...

    If ``split`` is provided, the loader first looks for
    ``root/<modality>/<split>/input`` and ``root/<modality>/<split>/gt``.
    File pairs are matched by filename stem.
    """

    def __init__(
        self,
        root: str | Path,
        modalities: Iterable[str] = ("MRI", "CT", "PET"),
        split: str | None = None,
        input_dir_name: str = "input",
        target_dir_name: str = "gt",
    ) -> None:
        self.root = Path(root)
        self.modalities = tuple(modalities)
        self.split = split
        self.input_dir_name = input_dir_name
        self.target_dir_name = target_dir_name
        self.samples: list[tuple[Path, Path, str]] = []

        for modality in self.modalities:
            base = self.root / modality
            if split is not None and (base / split / input_dir_name).exists():
                base = base / split
            input_dir = base / input_dir_name
            target_dir = base / target_dir_name
            if not input_dir.exists() or not target_dir.exists():
                continue

            inputs = self._collect(input_dir)
            targets = {p.stem: p for p in self._collect(target_dir)}
            for input_path in inputs:
                target_path = targets.get(input_path.stem)
                if target_path is not None:
                    self.samples.append((input_path, target_path, modality))

        if not self.samples:
            raise FileNotFoundError(
                "No paired restoration samples found. Expected files under "
                "root/<modality>/input and root/<modality>/gt, or under "
                "root/<modality>/<split>/input and gt when split is provided."
            )

    @staticmethod
    def _collect(directory: Path) -> list[Path]:
        files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        input_path, target_path, modality = self.samples[index]
        x = _to_chw_float(_array_from_file(input_path))
        y = _to_chw_float(_array_from_file(target_path))
        if x.shape != y.shape:
            raise ValueError(
                f"Input and target shapes do not match for {input_path.name}: "
                f"{tuple(x.shape)} vs {tuple(y.shape)}."
            )
        return x, y, modality
