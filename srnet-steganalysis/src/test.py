"""Evaluate one SRNet checkpoint on the predefined test split."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .datasets import PairedImageDataset, discover_pairs
from .srnet import SRNet
from .utils import (
    configure_logging,
    evaluate_model,
    format_duration,
    load_torch_file,
    model_state_from_checkpoint,
    resolve_device,
    seed_everything,
    seed_worker,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an SRNet checkpoint on data/test.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--cover-dir", default="cover")
    parser.add_argument("--stego-dir", default="stego")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8, help="Pairs per batch.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--verify-checkpoint-test-set",
        action="store_true",
        help=(
            "Require the evaluated pair names to match the original test set "
            "recorded in the checkpoint. By default, external test sets are allowed."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=None,
        help="ROC graph path (default: beside the metrics JSON).",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def verify_checkpoint_test_set(
    checkpoint: object,
    current_test_keys: Sequence[str],
    required: bool,
) -> None:
    """Optionally require the current pairs to be the checkpoint's original test set."""
    if not required or not isinstance(checkpoint, Mapping):
        return
    dataset_keys = checkpoint.get("dataset_keys")
    if not isinstance(dataset_keys, Mapping):
        return
    saved_test_keys = list(dataset_keys.get("test", []))
    if saved_test_keys and saved_test_keys != list(current_test_keys):
        raise ValueError(
            "Test data differs from the dataset recorded in the checkpoint. "
            "Remove --verify-checkpoint-test-set to evaluate an external dataset."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and worker count cannot be negative.")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    data_root = args.data_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / "standalone_test_metrics.json"
    )
    roc_output_path = (
        args.roc_output.expanduser().resolve()
        if args.roc_output is not None
        else output_path.with_name(f"{output_path.stem}_roc_curve.png")
    )
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else checkpoint_path.parent / "testing.log"
    )
    logger = configure_logging(log_file, "srnet.test")
    run_start = time.perf_counter()
    logger.info("RUN start command=%s", " ".join(sys.argv))
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    pairs = discover_pairs(data_root / args.test_split, args.cover_dir, args.stego_dir)
    checkpoint = load_torch_file(checkpoint_path)
    verify_checkpoint_test_set(
        checkpoint,
        [pair.key for pair in pairs],
        required=args.verify_checkpoint_test_set,
    )

    dataset = PairedImageDataset(pairs, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    model = SRNet().to(device)
    model.load_state_dict(model_state_from_checkpoint(checkpoint))
    criterion = nn.CrossEntropyLoss()
    if isinstance(checkpoint, Mapping) and "next_epoch" in checkpoint:
        epoch_label = str(max(int(checkpoint["next_epoch"]) - 1, 0))
    else:
        epoch_label = "weights"
    logger.info(
        "TEST start checkpoint=%s pairs=%d device=%s amp=%s",
        checkpoint_path,
        len(pairs),
        device,
        use_amp,
    )
    metrics = evaluate_model(
        model,
        loader,
        criterion,
        device,
        use_amp,
        "TEST",
        epoch_label,
        logger,
        roc_curve_path=roc_output_path,
    )
    write_json(
        output_path,
        {
            "checkpoint": str(checkpoint_path),
            "pairs": len(pairs),
            "roc_curve": str(roc_output_path),
            "test": metrics,
        },
    )
    logger.info(
        "RUN complete metrics=%s roc_curve=%s duration=%s",
        output_path,
        roc_output_path,
        format_duration(time.perf_counter() - run_start),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.getLogger("srnet.test").exception("FATAL %s", error)
        raise
