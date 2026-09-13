"""Evaluate every epoch_NNNN.pt checkpoint on a predefined split."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    write_metrics_csv,
)


def _epoch_number(path: Path) -> int:
    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Unexpected epoch checkpoint name: {path.name}")
    return int(match.group(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all epoch checkpoints on data/val or data/test."
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("runs/srnet"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--cover-dir", default="cover")
    parser.add_argument("--stego-dir", default="stego")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8, help="Pairs per batch.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints_dir = args.checkpoints_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    checkpoints = sorted(checkpoints_dir.glob("epoch_*.pt"), key=_epoch_number)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_NNNN.pt files found in {checkpoints_dir}")
    split_dir = args.val_split if args.split == "val" else args.test_split
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoints_dir / f"epoch_metrics_{args.split}.csv"
    )
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else checkpoints_dir / "epoch_evaluation.log"
    )
    logger = configure_logging(log_file, "srnet.eval_epochs")
    run_start = time.perf_counter()
    logger.info("RUN start command=%s", " ".join(sys.argv))
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    pairs = discover_pairs(data_root / split_dir, args.cover_dir, args.stego_dir)
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
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, Any]] = []
    logger.info(
        "EVALUATION start checkpoints=%d split=%s pairs=%d device=%s",
        len(checkpoints),
        args.split,
        len(pairs),
        device,
    )

    expected_keys = [pair.key for pair in pairs]
    for checkpoint_path in checkpoints:
        epoch = _epoch_number(checkpoint_path)
        checkpoint = load_torch_file(checkpoint_path)
        if isinstance(checkpoint, Mapping) and "dataset_keys" in checkpoint:
            saved_keys = list(checkpoint["dataset_keys"].get(args.split, []))
            if saved_keys and saved_keys != expected_keys:
                raise ValueError(f"{checkpoint_path} was trained with different {args.split} data.")
        model.load_state_dict(model_state_from_checkpoint(checkpoint))
        metrics = evaluate_model(
            model,
            loader,
            criterion,
            device,
            use_amp,
            args.split.upper(),
            str(epoch),
            logger,
        )
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "error_rate": metrics["error_rate"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "false_alarm_rate": metrics["false_alarm_rate"],
                "missed_detection_rate": metrics["missed_detection_rate"],
                "detection_probability": metrics["detection_probability"],
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tp": int(metrics["tp"]),
            }
        )

    write_metrics_csv(output_path, rows)
    best = min(rows, key=lambda row: row["loss"])
    logger.info(
        "EVALUATION complete best_epoch=%d best_loss=%.6f output=%s duration=%s",
        best["epoch"],
        best["loss"],
        output_path,
        format_duration(time.perf_counter() - run_start),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.getLogger("srnet.eval_epochs").exception("FATAL %s", error)
        raise
