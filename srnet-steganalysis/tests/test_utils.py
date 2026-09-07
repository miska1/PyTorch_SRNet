from __future__ import annotations

import unittest

import torch

from src.utils import ClassificationMeter, format_duration


class UtilityTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]])
        targets = torch.tensor([0, 1, 0, 1])
        meter = ClassificationMeter()
        meter.update(logits, targets, loss=0.25)
        metrics = meter.compute()
        self.assertAlmostEqual(metrics["loss"], 0.25)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["tn"], 1.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)

    def test_duration_format(self) -> None:
        self.assertEqual(format_duration(65), "01:05")
        self.assertEqual(format_duration(3661), "01:01:01")


if __name__ == "__main__":
    unittest.main()
