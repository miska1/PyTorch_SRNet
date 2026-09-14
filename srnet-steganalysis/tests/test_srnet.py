from __future__ import annotations

import unittest

import torch

from src.srnet import RGB_CHANNELS, SRNet, validate_checkpoint_input_channels


class SRNetTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = SRNet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, RGB_CHANNELS, 32, 32))
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(model.layer1.conv.in_channels, RGB_CHANNELS)

    def test_grayscale_checkpoint_is_rejected(self) -> None:
        grayscale_state = {"layer1.conv.weight": torch.zeros(64, 1, 3, 3)}
        with self.assertRaisesRegex(ValueError, "expects 1 input channel"):
            validate_checkpoint_input_channels(grayscale_state)


if __name__ == "__main__":
    unittest.main()
