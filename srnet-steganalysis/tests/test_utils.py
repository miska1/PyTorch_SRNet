from __future__ import annotations

import unittest

import torch

from src.utils import ClassificationMeter, compute_binary_roc, format_duration


class UtilityTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        logits = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]]
        )
        targets = torch.tensor([0, 0, 1, 1, 1])
        meter = ClassificationMeter()
        meter.update(logits, targets, loss=0.25)
        metrics = meter.compute()
        self.assertAlmostEqual(metrics["loss"], 0.25)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertEqual(metrics["tp"], 2.0)
        self.assertEqual(metrics["tn"], 1.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertAlmostEqual(metrics["false_alarm_rate"], 0.5)
        self.assertAlmostEqual(metrics["missed_detection_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["detection_probability"], 2.0 / 3.0)

    def test_binary_roc_auc(self) -> None:
        false_positive_rates, true_positive_rates, auc = compute_binary_roc(
            targets=[0, 0, 1, 1],
            scores=[0.1, 0.4, 0.35, 0.8],
        )
        self.assertAlmostEqual(auc, 0.75)
        self.assertEqual(false_positive_rates[0], 0.0)
        self.assertEqual(true_positive_rates[0], 0.0)
        self.assertEqual(false_positive_rates[-1], 1.0)
        self.assertEqual(true_positive_rates[-1], 1.0)

    def test_duration_format(self) -> None:
        self.assertEqual(format_duration(65), "01:05")
        self.assertEqual(format_duration(3661), "01:01:01")


if __name__ == "__main__":
    unittest.main()
