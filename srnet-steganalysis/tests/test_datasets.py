from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets import (
    EpochShuffleSampler,
    PairedImageDataset,
    discover_data_splits,
    discover_pairs,
    discover_recursive_pairs,
    find_cross_split_cover_duplicates,
)


def write_pair(split_root: Path, name: str, value: int) -> None:
    (split_root / "cover").mkdir(parents=True, exist_ok=True)
    (split_root / "stego").mkdir(parents=True, exist_ok=True)
    cover = np.full((32, 32), value, dtype=np.uint8)
    stego = cover.copy()
    stego[value % 32, (value * 3) % 32] ^= 1
    Image.fromarray(cover).save(split_root / "cover" / f"{name}.png")
    Image.fromarray(stego).save(split_root / "stego" / f"{name}.png")


class DatasetTests(unittest.TestCase):
    def test_predefined_splits_are_loaded_without_redistribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4):
                write_pair(root / "train", f"train_{index}", index + 1)
            for index in range(2):
                write_pair(root / "val", f"val_{index}", index + 20)
                write_pair(root / "test", f"test_{index}", index + 40)
            splits = discover_data_splits(root)
            self.assertEqual([pair.key for pair in splits["train"]], [
                "train_0", "train_1", "train_2", "train_3"
            ])
            self.assertEqual(len(splits["val"]), 2)
            self.assertEqual(len(splits["test"]), 2)

    def test_pair_augmentation_is_shared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            split_root = Path(temporary) / "train"
            write_pair(split_root, "sample", 7)
            dataset = PairedImageDataset(
                discover_pairs(split_root), image_size=32, augment=True, seed=11
            )
            dataset.set_epoch(3)
            images, labels, key = dataset[0]
            self.assertEqual(images.shape, (2, 1, 32, 32))
            self.assertTrue(torch.equal(labels, torch.tensor([0, 1])))
            self.assertEqual(key, "sample")
            self.assertEqual(int((images[0] != images[1]).sum()), 1)

    def test_recursive_discovery_combines_all_pair_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pair(root / "train", "same_name", 5)
            write_pair(root / "val", "same_name", 15)
            write_pair(root / "test", "same_name", 25)
            pairs, groups = discover_recursive_pairs(root)
            self.assertEqual(groups, {"test": 1, "train": 1, "val": 1})
            self.assertEqual(
                [pair.key for pair in pairs],
                ["test/same_name", "train/same_name", "val/same_name"],
            )

    def test_recursive_discovery_supports_a_direct_pair_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pair(root, "sample", 5)
            pairs, groups = discover_recursive_pairs(root)
            self.assertEqual(groups, {".": 1})
            self.assertEqual([pair.key for pair in pairs], ["sample"])

    def test_recursive_discovery_rejects_an_incomplete_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "orphan" / "cover").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "without both cover and stego"):
                discover_recursive_pairs(root)

    def test_grayscale_values_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            split_root = Path(temporary) / "train"
            write_pair(split_root, "sample", 7)
            dataset = PairedImageDataset(discover_pairs(split_root), image_size=32)
            images, _, _ = dataset[0]
            expected = torch.tensor(7.0).div(255.0).sub(0.5)
            self.assertTrue(torch.allclose(images[0, 0, 0, 0], expected))

    def test_missing_pair_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            split_root = Path(temporary) / "train"
            write_pair(split_root, "complete", 1)
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(
                split_root / "cover" / "unmatched.png"
            )
            with self.assertRaisesRegex(ValueError, "cover-only"):
                discover_pairs(split_root)

    def test_duplicate_cover_content_across_splits_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pair(root / "train", "a", 9)
            write_pair(root / "val", "b", 9)
            write_pair(root / "test", "c", 30)
            duplicates = find_cross_split_cover_duplicates(discover_data_splits(root))
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(duplicates[0][:2], ("train", "val"))

    def test_epoch_sampler_can_resume_at_dataset_end(self) -> None:
        sampler = EpochShuffleSampler(dataset_size=10, seed=5)
        sampler.set_epoch(epoch=3, start_index=10)
        self.assertEqual(list(sampler), [])


if __name__ == "__main__":
    unittest.main()
