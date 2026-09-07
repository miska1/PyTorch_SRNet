"""Train SRNet using predefined train/val/test cover-stego directories."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .datasets import (
    EpochShuffleSampler,
    PairedImageDataset,
    discover_data_splits,
    find_cross_split_cover_duplicates,
    pair_keys_by_split,
    verify_saved_pair_keys,
)
from .srnet import SRNet
from .utils import (
    ClassificationMeter,
    MinuteProgressReporter,
    atomic_torch_save,
    capture_rng_state,
    configure_logging,
    create_grad_scaler,
    create_optimizer,
    evaluate_model,
    flatten_pair_batch,
    format_duration,
    format_metrics,
    load_torch_file,
    model_state_from_checkpoint,
    resolve_device,
    resolve_resume_path,
    restore_rng_state,
    seed_everything,
    seed_worker,
    write_json,
)


@dataclass
class RuntimeProgress:
    epoch: int = 1
    next_batch: int = 0
    train_meter_state: dict[str, float | int] | None = None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    dataset: PairedImageDataset,
    sampler: EpochShuffleSampler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    gradient_clip: float,
    epoch: int,
    total_batches: int,
    batch_size: int,
    start_batch: int,
    initial_meter_state: Mapping[str, float | int] | None,
    runtime: RuntimeProgress,
    logger: logging.Logger,
) -> dict[str, float]:
    model.train()
    dataset.set_epoch(epoch)
    pair_offset = min(start_batch * batch_size, len(dataset))
    sampler.set_epoch(epoch, start_index=pair_offset)
    meter = ClassificationMeter()
    if initial_meter_state:
        meter.load_state_dict(initial_meter_state)

    runtime.epoch = epoch
    runtime.next_batch = start_batch
    runtime.train_meter_state = meter.state_dict()
    epoch_start = time.perf_counter()
    reporter = MinuteProgressReporter(
        logger=logger,
        epoch=epoch,
        total_batches=total_batches,
        initial_completed=start_batch,
        session_start=epoch_start,
    )

    try:
        for batch_number, (pair_images, pair_labels, _) in enumerate(
            loader, start=start_batch + 1
        ):
            batch_start = time.perf_counter()
            logger.info(
                "TRAIN batch start epoch=%d batch=%d/%d pairs=%d images=%d lr=%.8g",
                epoch,
                batch_number,
                total_batches,
                pair_images.shape[0],
                pair_images.shape[0] * 2,
                optimizer.param_groups[0]["lr"],
            )
            images, labels = flatten_pair_batch(pair_images, pair_labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            cpu_logits = logits.detach().cpu()
            cpu_labels = labels.detach().cpu()
            batch_loss = loss.detach().item()
            meter.update(cpu_logits, cpu_labels, batch_loss)
            runtime.next_batch = batch_number
            runtime.train_meter_state = meter.state_dict()
            reporter.record_completed(batch_number)
            reporter.start_after_first_batch(batch_number)
            batch_accuracy = (cpu_logits.argmax(1) == cpu_labels).float().mean().item()
            logger.info(
                "TRAIN batch complete epoch=%d batch=%d/%d loss=%.6f "
                "accuracy=%.4f duration=%s",
                epoch,
                batch_number,
                total_batches,
                batch_loss,
                batch_accuracy,
                format_duration(time.perf_counter() - batch_start),
            )
    finally:
        reporter.stop()

    metrics = meter.compute()
    logger.info(
        "TRAIN epoch complete epoch=%d %s duration=%s",
        epoch,
        format_metrics(metrics),
        format_duration(time.perf_counter() - epoch_start),
    )
    return metrics


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Any,
    args: argparse.Namespace,
    dataset_keys: Mapping[str, Sequence[str]],
    next_epoch: int,
    next_batch: int,
    train_meter_state: Mapping[str, float | int] | None,
    best_val_loss: float,
    no_improvement_epochs: int,
    train_metrics: Mapping[str, float] | None = None,
    validation_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "model_name": "SRNet",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "next_epoch": next_epoch,
        "next_batch": next_batch,
        "train_meter_state": dict(train_meter_state) if train_meter_state else None,
        "best_val_loss": best_val_loss,
        "no_improvement_epochs": no_improvement_epochs,
        "train_metrics": dict(train_metrics) if train_metrics else None,
        "validation_metrics": dict(validation_metrics) if validation_metrics else None,
        "dataset_keys": {name: list(keys) for name, keys in dataset_keys.items()},
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rng_state": capture_rng_state(),
        "saved_at_unix": time.time(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SRNet on predefined data/train, data/val, and data/test splits."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--cover-dir", default="cover")
    parser.add_argument("--stego-dir", default="stego")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/srnet"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of cover/stego pairs per batch (twice as many images).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augmentation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--check-split-overlap",
        action="store_true",
        help="Hash cover files and fail if identical data occurs across splits.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Full checkpoint path, plain state_dict path, or 'latest'.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else args.output_dir / "training.log"
    )
    logger = configure_logging(log_file, "srnet.train")
    run_start = time.perf_counter()
    logger.info("RUN start command=%s", " ".join(sys.argv))

    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    logger.info(
        "SYSTEM device=%s amp=%s torch=%s log_file=%s",
        device,
        use_amp,
        torch.__version__,
        log_file,
    )

    resume_path = resolve_resume_path(args.resume, args.output_dir)
    resume_value = load_torch_file(resume_path) if resume_path else None
    if isinstance(resume_value, Mapping) and "split_keys" in resume_value:
        raise ValueError(
            "This is a legacy random-split checkpoint. Start a new run with the "
            "predefined train/val/test dataset layout."
        )

    splits = discover_data_splits(
        args.data_root,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        cover_dir=args.cover_dir,
        stego_dir=args.stego_dir,
    )
    verify_saved_pair_keys(
        splits,
        resume_value.get("dataset_keys")
        if isinstance(resume_value, Mapping) and "model_state" in resume_value
        else None,
    )
    dataset_keys = pair_keys_by_split(splits)
    logger.info(
        "DATA train_pairs=%d val_pairs=%d test_pairs=%d image_size=%dx%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        args.image_size,
        args.image_size,
    )
    if args.check_split_overlap:
        logger.info("DATA split-overlap check start")
        duplicates = find_cross_split_cover_duplicates(splits)
        if duplicates:
            first = duplicates[0]
            raise ValueError(
                f"Found {len(duplicates)} byte-identical cover images across splits; "
                f"first duplicate is {first[2]} ({first[0]}) and {first[3]} ({first[1]})."
            )
        logger.info("DATA split-overlap check complete duplicates=0")

    train_dataset = PairedImageDataset(
        splits["train"], args.image_size, augment=args.augmentation, seed=args.seed
    )
    val_dataset = PairedImageDataset(splits["val"], args.image_size)
    test_dataset = PairedImageDataset(splits["test"], args.image_size)
    train_sampler = EpochShuffleSampler(len(train_dataset), args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    total_train_batches = math.ceil(len(train_dataset) / args.batch_size)

    model = SRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    scaler = create_grad_scaler(use_amp)
    logger.info(
        "MODEL name=SRNet trainable_parameters=%d optimizer=Adamax",
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )

    next_epoch = 1
    next_batch = 0
    resume_meter_state: Mapping[str, float | int] | None = None
    best_val_loss = math.inf
    no_improvement_epochs = 0
    if resume_value is not None:
        model.load_state_dict(model_state_from_checkpoint(resume_value))
        if isinstance(resume_value, Mapping) and "model_state" in resume_value:
            optimizer.load_state_dict(resume_value["optimizer_state"])
            scheduler.load_state_dict(resume_value["scheduler_state"])
            scaler.load_state_dict(resume_value.get("scaler_state", {}))
            next_epoch = int(resume_value.get("next_epoch", 1))
            next_batch = int(resume_value.get("next_batch", 0))
            resume_meter_state = resume_value.get("train_meter_state")
            best_val_loss = float(resume_value.get("best_val_loss", math.inf))
            no_improvement_epochs = int(resume_value.get("no_improvement_epochs", 0))
            restore_rng_state(resume_value.get("rng_state"))
            logger.info(
                "RESUME full_checkpoint=%s next_epoch=%d next_batch=%d",
                resume_path,
                next_epoch,
                next_batch,
            )
        else:
            logger.info("RESUME plain_model_weights=%s starting_epoch=1", resume_path)

    if not 0 <= next_batch <= total_train_batches:
        raise ValueError(
            f"Checkpoint next_batch={next_batch} is invalid for {total_train_batches} batches."
        )

    runtime = RuntimeProgress(
        epoch=next_epoch,
        next_batch=next_batch,
        train_meter_state=dict(resume_meter_state) if resume_meter_state else None,
    )
    last_train_metrics: Mapping[str, float] | None = None
    last_val_metrics: Mapping[str, float] | None = None
    last_completed_epoch = next_epoch - 1

    try:
        for epoch in range(next_epoch, args.epochs + 1):
            start_batch = next_batch if epoch == next_epoch else 0
            initial_meter = resume_meter_state if epoch == next_epoch else None
            logger.info(
                "EPOCH start epoch=%d/%d resume_batch=%d total_batches=%d",
                epoch,
                args.epochs,
                start_batch,
                total_train_batches,
            )
            last_train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                dataset=train_dataset,
                sampler=train_sampler,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                gradient_clip=args.gradient_clip,
                epoch=epoch,
                total_batches=total_train_batches,
                batch_size=args.batch_size,
                start_batch=start_batch,
                initial_meter_state=initial_meter,
                runtime=runtime,
                logger=logger,
            )
            last_val_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                "VALIDATION",
                str(epoch),
                logger,
            )
            scheduler.step(last_val_metrics["loss"])
            improved = last_val_metrics["loss"] < best_val_loss - args.min_delta
            if improved:
                best_val_loss = last_val_metrics["loss"]
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                dataset_keys,
                next_epoch=epoch + 1,
                next_batch=0,
                train_meter_state=None,
                best_val_loss=best_val_loss,
                no_improvement_epochs=no_improvement_epochs,
                train_metrics=last_train_metrics,
                validation_metrics=last_val_metrics,
            )
            epoch_path = args.output_dir / f"epoch_{epoch:04d}.pt"
            atomic_torch_save(payload, epoch_path)
            atomic_torch_save(payload, args.output_dir / "last.pt")
            if improved:
                atomic_torch_save(payload, args.output_dir / "best.pt")
            last_completed_epoch = epoch
            runtime = RuntimeProgress(epoch=epoch + 1)
            logger.info(
                "CHECKPOINT epoch=%d saved=%s best=%s best_val_loss=%.6f "
                "epochs_without_improvement=%d",
                epoch,
                epoch_path,
                improved,
                best_val_loss,
                no_improvement_epochs,
            )
            logger.info(
                "EPOCH complete epoch=%d/%d train_loss=%.6f val_loss=%.6f lr=%.8g",
                epoch,
                args.epochs,
                last_train_metrics["loss"],
                last_val_metrics["loss"],
                optimizer.param_groups[0]["lr"],
            )
            next_batch = 0
            resume_meter_state = None
            if no_improvement_epochs >= args.early_stopping_patience:
                logger.info(
                    "EARLY STOP epoch=%d patience=%d best_val_loss=%.6f",
                    epoch,
                    args.early_stopping_patience,
                    best_val_loss,
                )
                break
    except KeyboardInterrupt:
        interrupted_path = args.output_dir / "interrupted.pt"
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                dataset_keys,
                next_epoch=runtime.epoch,
                next_batch=runtime.next_batch,
                train_meter_state=runtime.train_meter_state,
                best_val_loss=best_val_loss,
                no_improvement_epochs=no_improvement_epochs,
                train_metrics=last_train_metrics,
                validation_metrics=last_val_metrics,
            ),
            interrupted_path,
        )
        logger.warning(
            "INTERRUPTED saved=%s next_epoch=%d next_batch=%d",
            interrupted_path,
            runtime.epoch,
            runtime.next_batch,
        )
        logger.info("RUN interrupted duration=%s", format_duration(time.perf_counter() - run_start))
        return 130

    best_path = args.output_dir / "best.pt"
    if best_path.is_file():
        best_checkpoint = load_torch_file(best_path)
        model.load_state_dict(model_state_from_checkpoint(best_checkpoint))
        logger.info("TEST using_best_checkpoint=%s", best_path)
    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        use_amp,
        "TEST",
        str(last_completed_epoch),
        logger,
    )
    write_json(
        args.output_dir / "test_metrics.json",
        {
            "test": test_metrics,
            "best_validation_loss": best_val_loss,
            "last_completed_epoch": last_completed_epoch,
        },
    )
    logger.info("RUN complete duration=%s", format_duration(time.perf_counter() - run_start))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.getLogger("srnet.train").exception("FATAL %s", error)
        raise
