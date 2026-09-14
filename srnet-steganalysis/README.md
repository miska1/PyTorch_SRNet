# SRNet paired steganalysis trainer

PyTorch training, validation, and testing for the original 12-layer SRNet
architecture. Input images are grayscale 256x256 cover/stego pairs that are
already separated into training, validation, and test datasets.

## Project layout

```text
srnet-steganalysis/
├── data/
│   ├── train/
│   │   ├── cover/
│   │   └── stego/
│   ├── val/
│   │   ├── cover/
│   │   └── stego/
│   └── test/
│       ├── cover/
│       └── stego/
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── eval_epochs.py
│   ├── srnet.py
│   ├── test.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_datasets.py
│   ├── test_srnet.py
│   ├── test_test_cli.py
│   └── test_utils.py
├── README.md
└── requirements.txt
```

If the validation directory is named `validation` instead of `val`, pass
`--val-split validation` to training and epoch evaluation.

## Pair naming

Pairing occurs independently inside every split. Relative paths without file
extensions are matched case-insensitively:

```text
data/train/cover/camera_a/000001.png
data/train/stego/camera_a/000001.tif
```

Every cover must have exactly one matching stego image. Images are converted to
one channel but never resized, because interpolation can alter the weak stego
signal. `--batch-size 8` means 8 pairs and therefore 16 images.

## Install

Python 3.10 or newer is recommended:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch appropriate for the local
CUDA driver.

## Train

Run commands from the project root:

```bash
python -m src.train --data-root data --output-dir runs/srnet
```

Example with common options:

```bash
python -m src.train \
  --data-root data \
  --output-dir runs/srnet \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --early-stopping-patience 12 \
  --check-split-overlap
```

`--check-split-overlap` hashes cover files and stops if byte-identical images
occur in different splits. This check is optional because hashing a large
dataset adds startup time.

Training uses only `data/train`, model selection uses only `data/val`, and the
best validation checkpoint is evaluated once on `data/test` after training.

## Logging and overtraining controls

The trainer logs to both the console and `OUTPUT_DIR/training.log`:

- timestamped start and completion of every epoch;
- timestamped start and completion of every train/validation/test batch;
- loss, accuracy, confusion counts, F1, balanced accuracy, false-alarm rate,
  missed-detection rate, detection probability, and durations;
- one-minute batch throughput and estimated epoch duration, starting after the
  first completed training batch.

Overtraining is controlled with validation-loss early stopping, validation-
driven learning-rate reduction, L2 weight decay, gradient clipping, paired
lossless rotation/mirroring, and final testing with the best validation model.

## Checkpoints, Ctrl+C, and resume

Every completed epoch creates a separate full checkpoint:

```text
runs/srnet/epoch_0001.pt
runs/srnet/epoch_0002.pt
runs/srnet/last.pt
runs/srnet/best.pt
```

Full checkpoints include model, optimizer, scheduler, AMP scaler, early-
stopping state, RNG state, next batch position, configuration, and exact pair
keys for all predefined datasets.

Press **Ctrl+C** once during training. The program writes
`runs/srnet/interrupted.pt`. Resume from it with:

```bash
python -m src.train \
  --data-root data \
  --output-dir runs/srnet \
  --resume runs/srnet/interrupted.pt
```

Or select the most recently modified checkpoint automatically:

```bash
python -m src.train --data-root data --output-dir runs/srnet --resume latest
```

`--epochs` is the final epoch number, not the number of additional epochs.
Only load checkpoint files you trust because PyTorch checkpoints can contain
pickled objects.

## Standalone testing

Evaluate one checkpoint recursively on every paired group below `--data-root`:

```bash
python -m src.test \
  --data-root data \
  --checkpoint runs/srnet/best.pt
```

For the standard project layout, this combines all images from:

```text
data/test/cover/   data/test/stego/
data/train/cover/  data/train/stego/
data/val/cover/    data/val/stego/
```

The metrics JSON records every discovered group and its pair count. Pair keys
are prefixed with their relative group path, so identical filenames in `train`,
`val`, and `test` do not collide.

Results are written to `runs/srnet/standalone_test_metrics.json` and the console
and file log. The JSON includes false-alarm rate, missed-detection rate,
detection probability, and ROC-AUC. A 300-DPI ROC graph is written to
`runs/srnet/standalone_test_metrics_roc_curve.png`. The standalone tester
accepts any paired dataset; it does not need
to be the test set that was present during training. For example:

```bash
python -m src.test \
  --data-root /path/to/another-dataset \
  --test-split test \
  --checkpoint runs/srnet/best.pt \
  --output runs/srnet/another_dataset_metrics.json \
  --roc-output runs/srnet/another_dataset_roc.png
```

The external dataset must retain the paired `test/cover` and `test/stego`
structure. To evaluate only one subdirectory instead of the complete recursive
tree, pass it explicitly:

```bash
python -m src.test \
  --data-root data \
  --test-split test \
  --checkpoint runs/srnet/best.pt
```

Add `--verify-checkpoint-test-set` only with the single original test split when
you intentionally want to require its checkpoint-recorded membership.

Combining `train`, `val`, and `test` is useful for diagnostics, but the combined
result is not an unbiased held-out test measurement because the model has
already learned from the training images and was selected using validation.

At the end of training, the same metrics are saved in `test_metrics.json`, and
the test ROC graph is saved as `test_roc_curve.png` in the training output
directory. Rates and probabilities are stored as proportions from 0 to 1.

## Evaluate every epoch

Evaluate all `epoch_NNNN.pt` files on validation data and create a CSV table:

```bash
python -m src.eval_epochs \
  --data-root data \
  --checkpoints-dir runs/srnet \
  --split val
```

The default output is `runs/srnet/epoch_metrics_val.csv`. To compare checkpoints
on the test split instead, use `--split test`; avoid doing this repeatedly while
tuning the model because it turns the test set into another validation set.

## Unit tests

```bash
python -m unittest discover -s tests -v
```

The tests cover grayscale value preservation, split discovery, pair matching,
shared pair augmentation, duplicate detection, resumable shuffling, metrics,
and SRNet output shape.

## Architecture reference

The model follows M. Boroumand, M. Chen, and J. Fridrich, “Deep Residual Network
for Steganalysis of Digital Images,” *IEEE Transactions on Information
Forensics and Security*, 14(5), 2019. The model is isolated in `src/srnet.py`.
