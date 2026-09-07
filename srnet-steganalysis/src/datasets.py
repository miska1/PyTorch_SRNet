"""Paired cover/stego dataset discovery, loading, and deterministic sampling."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".pgm",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
LABEL_COVER = 0
LABEL_STEGO = 1


@dataclass(frozen=True)
class PairRecord:
    key: str
    cover_path: Path
    stego_path: Path


def _index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(directory)
        key = relative.with_suffix("").as_posix().casefold()
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path

    if duplicates:
        examples = "; ".join(
            f"{key}: {', '.join(str(path) for path in paths)}"
            for key, paths in list(duplicates.items())[:3]
        )
        raise ValueError(
            "Duplicate relative names (ignoring extension and case) were found: " + examples
        )
    if not index:
        raise ValueError(f"No supported images found under {directory}")
    return index


def discover_pairs(
    split_root: Path, cover_dir: str = "cover", stego_dir: str = "stego"
) -> list[PairRecord]:
    """Match images within one already-separated train/val/test directory."""
    covers = _index_images(split_root / cover_dir)
    stegos = _index_images(split_root / stego_dir)
    cover_only = sorted(covers.keys() - stegos.keys())
    stego_only = sorted(stegos.keys() - covers.keys())
    if cover_only or stego_only:
        details: list[str] = []
        if cover_only:
            details.append(f"cover-only examples: {cover_only[:5]}")
        if stego_only:
            details.append(f"stego-only examples: {stego_only[:5]}")
        raise ValueError(
            f"Every cover in {split_root} must have a stego image with the same "
            "relative name (extension may differ). " + "; ".join(details)
        )
    return [PairRecord(key, covers[key], stegos[key]) for key in sorted(covers)]


def discover_data_splits(
    data_root: Path,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    cover_dir: str = "cover",
    stego_dir: str = "stego",
) -> dict[str, list[PairRecord]]:
    """Load predefined splits without random redistribution."""
    split_dirs = {"train": train_split, "val": val_split, "test": test_split}
    return {
        name: discover_pairs(data_root / split_dir, cover_dir, stego_dir)
        for name, split_dir in split_dirs.items()
    }


def pair_keys_by_split(
    splits: Mapping[str, Sequence[PairRecord]],
) -> dict[str, list[str]]:
    return {name: [pair.key for pair in pairs] for name, pairs in splits.items()}


def verify_saved_pair_keys(
    current: Mapping[str, Sequence[PairRecord]],
    saved: Mapping[str, Sequence[str]] | None,
) -> None:
    """Prevent a resumed run from silently using changed split membership."""
    if saved is None:
        return
    current_keys = pair_keys_by_split(current)
    for split_name in ("train", "val", "test"):
        expected = list(saved.get(split_name, []))
        actual = current_keys[split_name]
        if expected != actual:
            raise ValueError(
                f"The {split_name!r} dataset differs from the saved checkpoint. "
                "Resume with the original data or start a new output directory."
            )


def find_cross_split_cover_duplicates(
    splits: Mapping[str, Sequence[PairRecord]],
) -> list[tuple[str, str, Path, Path]]:
    """Return byte-identical cover images that occur in different splits."""
    seen: dict[str, tuple[str, Path]] = {}
    duplicates: list[tuple[str, str, Path, Path]] = []
    for split_name in ("train", "val", "test"):
        for pair in splits[split_name]:
            digest = hashlib.sha256(pair.cover_path.read_bytes()).hexdigest()
            previous = seen.get(digest)
            if previous is not None and previous[0] != split_name:
                duplicates.append((previous[0], split_name, previous[1], pair.cover_path))
            else:
                seen[digest] = (split_name, pair.cover_path)
    return duplicates


class PairedImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Return one cover/stego pair with an optional shared lossless transform."""

    def __init__(
        self,
        pairs: Sequence[PairRecord],
        image_size: int = 256,
        augment: bool = False,
        seed: int = 1337,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            if image.size != (self.image_size, self.image_size):
                raise ValueError(
                    f"{path} has size {image.size}; expected "
                    f"{self.image_size}x{self.image_size}. Images are not resized "
                    "because interpolation can alter the stego signal."
                )
            pixels = np.asarray(image.convert("L"), dtype=np.float32).copy()
        return torch.from_numpy(pixels).unsqueeze(0).div_(255.0).sub_(0.5)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        images = torch.stack(
            [self._load_image(pair.cover_path), self._load_image(pair.stego_path)]
        )
        if self.augment:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            images = torch.rot90(images, rng.randrange(4), dims=(-2, -1))
            if rng.random() < 0.5:
                images = torch.flip(images, dims=(-1,))
            if rng.random() < 0.5:
                images = torch.flip(images, dims=(-2,))
        labels = torch.tensor([LABEL_COVER, LABEL_STEGO], dtype=torch.long)
        return images, labels, pair.key


class EpochShuffleSampler(Sampler[int]):
    """Epoch-seeded pair shuffle that supports exact mid-epoch restart."""

    def __init__(self, dataset_size: int, seed: int) -> None:
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = 1
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= self.dataset_size:
            raise ValueError("Sampler start index is outside the dataset.")
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self) -> Iterable[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return self.dataset_size - self.start_index
