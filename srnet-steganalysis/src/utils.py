"""Shared metrics, logging, checkpoint, device, and evaluation helpers."""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


LABEL_COVER = 0
LABEL_STEGO = 1


class ClassificationMeter:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.count = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        predictions = logits.argmax(dim=1)
        sample_count = targets.numel()
        self.loss_sum += loss * sample_count
        self.count += sample_count
        self.tp += int(((predictions == LABEL_STEGO) & (targets == LABEL_STEGO)).sum())
        self.tn += int(((predictions == LABEL_COVER) & (targets == LABEL_COVER)).sum())
        self.fp += int(((predictions == LABEL_STEGO) & (targets == LABEL_COVER)).sum())
        self.fn += int(((predictions == LABEL_COVER) & (targets == LABEL_STEGO)).sum())

    def state_dict(self) -> dict[str, float | int]:
        return {
            "loss_sum": self.loss_sum,
            "count": self.count,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }

    def load_state_dict(self, state: Mapping[str, float | int]) -> None:
        self.loss_sum = float(state.get("loss_sum", 0.0))
        self.count = int(state.get("count", 0))
        self.tp = int(state.get("tp", 0))
        self.tn = int(state.get("tn", 0))
        self.fp = int(state.get("fp", 0))
        self.fn = int(state.get("fn", 0))

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise RuntimeError("Cannot compute metrics for an empty data split.")

        def safe_div(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator else 0.0

        accuracy = safe_div(self.tp + self.tn, self.count)
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        specificity = safe_div(self.tn, self.tn + self.fp)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        return {
            "loss": self.loss_sum / self.count,
            "accuracy": accuracy,
            "error_rate": 1.0 - accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "f1": f1,
            "tp": float(self.tp),
            "tn": float(self.tn),
            "fp": float(self.fp),
            "fn": float(self.fn),
        }


class MinuteProgressReporter:
    """Log one-minute batch throughput after the first completed batch."""

    def __init__(
        self,
        logger: logging.Logger,
        epoch: int,
        total_batches: int,
        initial_completed: int,
        session_start: float,
    ) -> None:
        self.logger = logger
        self.epoch = epoch
        self.total_batches = total_batches
        self.initial_completed = initial_completed
        self.session_start = session_start
        self.completed = initial_completed
        self.previous_completed = initial_completed
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start_after_first_batch(self, completed: int) -> None:
        if self._thread is not None:
            return
        with self._lock:
            self.completed = completed
            self.previous_completed = completed
        self._thread = threading.Thread(
            target=self._run,
            name=f"epoch-{self.epoch}-minute-progress",
            daemon=True,
        )
        self._thread.start()

    def record_completed(self, completed: int) -> None:
        with self._lock:
            self.completed = completed

    def _run(self) -> None:
        while not self._stop.wait(60.0):
            now = time.perf_counter()
            with self._lock:
                completed = self.completed
                in_last_minute = completed - self.previous_completed
                self.previous_completed = completed

            processed = completed - self.initial_completed
            elapsed = max(now - self.session_start, 1e-9)
            rate = processed / elapsed
            if rate > 0:
                estimated_duration = self.total_batches / rate
                eta = max((self.total_batches - completed) / rate, 0.0)
                estimate = (
                    f"estimated_epoch_duration={format_duration(estimated_duration)} "
                    f"eta={format_duration(eta)}"
                )
            else:
                estimate = "estimated_epoch_duration=unknown eta=unknown"
            self.logger.info(
                "MINUTE epoch=%d batches_in_last_minute=%d progress=%d/%d %s",
                self.epoch,
                in_last_minute,
                completed,
                self.total_batches,
                estimate,
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_metrics(metrics: Mapping[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.6f} accuracy={metrics['accuracy']:.4f} "
        f"error_rate={metrics['error_rate']:.4f} f1={metrics['f1']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"confusion=(tn={int(metrics['tn'])}, fp={int(metrics['fp'])}, "
        f"fn={int(metrics['fn'])}, tp={int(metrics['tp'])})"
    )


def configure_logging(log_file: Path, logger_name: str = "srnet") -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def create_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.Adamax(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def flatten_pair_batch(
    pair_images: torch.Tensor, pair_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return pair_images.flatten(0, 1), pair_labels.flatten(0, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    phase: str,
    epoch_label: str,
    logger: logging.Logger,
) -> dict[str, float]:
    model.eval()
    meter = ClassificationMeter()
    phase_start = time.perf_counter()
    for batch_number, (pair_images, pair_labels, _) in enumerate(loader, start=1):
        batch_start = time.perf_counter()
        logger.info(
            "%s batch start epoch=%s batch=%d/%d pairs=%d images=%d",
            phase,
            epoch_label,
            batch_number,
            len(loader),
            pair_images.shape[0],
            pair_images.shape[0] * 2,
        )
        images, labels = flatten_pair_batch(pair_images, pair_labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_loss = loss.detach().item()
        meter.update(logits.detach().cpu(), labels.detach().cpu(), batch_loss)
        logger.info(
            "%s batch complete epoch=%s batch=%d/%d loss=%.6f duration=%s",
            phase,
            epoch_label,
            batch_number,
            len(loader),
            batch_loss,
            format_duration(time.perf_counter() - batch_start),
        )
    metrics = meter.compute()
    logger.info(
        "%s complete epoch=%s %s duration=%s",
        phase,
        epoch_label,
        format_metrics(metrics),
        format_duration(time.perf_counter() - phase_start),
    )
    return metrics


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_state_from_checkpoint(value: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(value, Mapping) and "model_state" in value:
        return value["model_state"]
    if isinstance(value, Mapping) and value and all(
        isinstance(item, torch.Tensor) for item in value.values()
    ):
        return value
    raise ValueError("File is neither a full SRNet checkpoint nor a plain state_dict.")


def resolve_resume_path(resume: str | None, output_dir: Path) -> Path | None:
    if resume is None:
        return None
    if resume != "latest":
        path = Path(resume).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    candidates = list(output_dir.glob("epoch_*.pt"))
    candidates.extend(
        path
        for path in (output_dir / "last.pt", output_dir / "interrupted.pt")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metrics table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
